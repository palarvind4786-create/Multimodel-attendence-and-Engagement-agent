import cv2
import time
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete

from app.vision.detector import PersonDetector
from app.vision.tracker import MultiPersonTracker
from app.vision.face_recognizer import LocalFaceRecognizer
from app.vision.head_pose import HeadPoseEstimator
from app.audio.audio_analyzer import AggregateAudioAnalyzer
from app.services.attendance_service import AttendanceService
from app.models.db_models import EngagementWindow, Session, Student
from app.core.config import settings

class MultimodalPipeline:
    def __init__(self):
        self.detector = PersonDetector()
        self.tracker = MultiPersonTracker()
        self.face_recognizer = LocalFaceRecognizer()
        self.head_pose = HeadPoseEstimator()
        self.audio_analyzer = AggregateAudioAnalyzer()

    def calculate_frame_motion(self, prev_gray: Optional[np.ndarray], curr_gray: np.ndarray) -> float:
        """Calculates normalized motion intensity (0.0 to 1.0) via frame differencing."""
        if prev_gray is None or curr_gray is None:
            return 0.0
        diff = cv2.absdiff(prev_gray, curr_gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion_ratio = float(np.count_nonzero(thresh)) / float(thresh.size)
        return min(1.0, motion_ratio * 5.0)

    async def process_video_file(
        self,
        db: AsyncSession,
        session_id: int,
        video_filepath: str,
        enrolled_cache: List[Tuple[int, str, np.ndarray]],  # (student_id, student_name, vector)
        progress_callback: Optional[Callable[[float, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> Dict:
        """
        Asynchronously processes an uploaded video file with:
        - Person detection & tracking
        - Biometric face matching against consented students
        - Head pose orientation calculation
        - Motion intensity calculation
        - 30-second window aggregation saved to SQLite
        """
        cap = cv2.VideoCapture(video_filepath)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_filepath}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0.0

        # Run audio analysis across 30-sec windows
        audio_windows = self.audio_analyzer.extract_from_video_and_analyze_windows(
            video_filepath=video_filepath,
            total_duration_sec=duration_sec,
            window_size_sec=float(settings.ENGAGEMENT_WINDOW_SECONDS)
        )
        audio_map = {w["window_index"]: w for w in audio_windows}

        # Sampling step to optimize CPU/GPU processing (e.g. process ~3 fps)
        step = max(1, int(fps / 3.0))
        frame_idx = 0
        prev_gray = None

        # Window aggregation accumulators
        window_size_sec = float(settings.ENGAGEMENT_WINDOW_SECONDS)
        num_windows = max(1, int(np.ceil(duration_sec / window_size_sec)))
        window_stats: Dict[int, Dict] = {
            w: {
                "counts": [],
                "forward_ratios": [],
                "motions": [],
                "inactivity_counts": []
            }
            for w in range(num_windows)
        }

        try:
            while cap.isOpened():
                if cancel_check and cancel_check():
                    print("[MultimodalPipeline] Processing cancelled by user.")
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % step == 0:
                    curr_time_sec = frame_idx / fps
                    curr_window_idx = min(num_windows - 1, int(curr_time_sec // window_size_sec))

                    # 1. Motion
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    motion = self.calculate_frame_motion(prev_gray, gray)
                    prev_gray = gray

                    # 2. Person Detection
                    detections = self.detector.detect(frame)
                    
                    # 3. Person Tracking
                    active_tracks = self.tracker.update(detections)

                    # 4. Face Recognition & Head Pose per person
                    forward_count = 0
                    total_faces_checked = 0

                    for track in active_tracks:
                        x1, y1, x2, y2 = track.bbox
                        person_crop = frame[y1:y2, x1:x2]
                        if person_crop.size == 0:
                            continue

                        # Detect face inside person crop
                        face_crop = self.face_recognizer.detect_face(person_crop)
                        if face_crop is not None:
                            total_faces_checked += 1
                            
                            # Estimate observable head orientation
                            is_fwd, yaw, pitch, roll = self.head_pose.estimate_orientation(face_crop)
                            if is_fwd:
                                forward_count += 1

                            # Try matching face if not already matched
                            if track.matched_student_id is None and enrolled_cache:
                                emb = self.face_recognizer.extract_embedding(face_crop)
                                if emb is not None:
                                    s_id, s_name, score = self.face_recognizer.match_face(emb, enrolled_cache)
                                    if s_id is not None:
                                        track.matched_student_id = s_id
                                        track.matched_student_name = s_name
                                        track.face_confidence = score

                        # Update attendance record for this sighting
                        await AttendanceService.register_or_update_detection(
                            db=db,
                            session_id=session_id,
                            track_id=track.track_id,
                            matched_student_id=track.matched_student_id,
                            confidence=track.face_confidence if track.matched_student_id else track.score,
                            timestamp_sec=curr_time_sec,
                            frame_interval_sec=float(step) / fps
                        )

                    # Inactivity calculation: persons with movement < 0.05
                    inactivity_ratio = 1.0 if motion < 0.02 else 0.0
                    fwd_ratio = (forward_count / total_faces_checked) if total_faces_checked > 0 else 0.5

                    # Accumulate into current window
                    window_stats[curr_window_idx]["counts"].append(len(active_tracks))
                    window_stats[curr_window_idx]["forward_ratios"].append(fwd_ratio)
                    window_stats[curr_window_idx]["motions"].append(motion)
                    window_stats[curr_window_idx]["inactivity_counts"].append(inactivity_ratio)

                    # Update progress
                    if progress_callback and total_frames > 0:
                        progress = min(1.0, float(frame_idx) / float(total_frames))
                        progress_callback(progress, f"Analyzing frame {frame_idx}/{total_frames} ({curr_time_sec:.1f}s)")

                frame_idx += 1

        finally:
            cap.release()

        # Clear any prior engagement windows for this session to avoid duplicates on re-analysis
        await db.execute(delete(EngagementWindow).where(EngagementWindow.session_id == session_id))

        # Build and store EngagementWindow records
        created_windows = []
        for w_idx in range(num_windows):
            w_start = w_idx * window_size_sec
            w_end = min(duration_sec, (w_idx + 1) * window_size_sec)
            stat = window_stats[w_idx]

            avg_persons = float(np.mean(stat["counts"])) if stat["counts"] else 0.0
            avg_fwd = float(np.mean(stat["forward_ratios"])) if stat["forward_ratios"] else 0.5
            avg_motion = float(np.mean(stat["motions"])) if stat["motions"] else 0.0
            avg_inactivity = float(np.mean(stat["inactivity_counts"])) if stat["inactivity_counts"] else 0.0

            aud = audio_map.get(w_idx, {"is_audio_available": False, "speech_active_ratio": 0.0})
            speech_ratio = aud.get("speech_active_ratio", 0.0)

            # Composite observable activity index (0 - 100)
            composite_score = (
                (avg_fwd * 35.0) +
                (min(1.0, avg_motion * 3.0) * 35.0) +
                (speech_ratio * 30.0)
            )
            composite_score = round(max(0.0, min(100.0, composite_score)), 1)

            window_obj = EngagementWindow(
                session_id=session_id,
                window_index=w_idx,
                start_seconds=w_start,
                end_seconds=w_end,
                person_count_avg=round(avg_persons, 1),
                forward_facing_ratio=round(avg_fwd, 2),
                movement_intensity=round(avg_motion, 2),
                prolonged_inactivity_ratio=round(avg_inactivity, 2),
                speech_active_ratio=round(speech_ratio, 2),
                interaction_activity_score=composite_score,
                is_audio_available=aud.get("is_audio_available", False),
                is_video_available=True,
                notes="Observable behavioral signals"
            )
            db.add(window_obj)
            created_windows.append(window_obj)

        await db.commit()

        return {
            "session_id": session_id,
            "processed_frames": frame_idx,
            "duration_seconds": duration_sec,
            "windows_count": len(created_windows)
        }
