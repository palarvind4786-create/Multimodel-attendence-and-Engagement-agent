import os
import uuid
import asyncio
import cv2
import numpy as np
from typing import Dict
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal, get_db
from app.core.config import settings
from app.services.student_service import StudentService
from app.services.session_service import SessionService
from app.services.attendance_service import AttendanceService
from app.services.suggestion_service import SuggestionService
from app.vision.pipeline import MultimodalPipeline
from app.models.db_models import Session

router = APIRouter()
pipeline = MultimodalPipeline()
suggestion_service = SuggestionService()

# In-memory job tracker for video upload background processing
ACTIVE_JOBS: Dict[str, Dict] = {}

async def run_video_processing_task(job_id: str, session_id: int, video_path: str):
    ACTIVE_JOBS[job_id]["status"] = "PROCESSING"
    ACTIVE_JOBS[job_id]["progress"] = 0.05
    ACTIVE_JOBS[job_id]["message"] = "Loading face models and enrolled student profiles..."

    async with AsyncSessionLocal() as db:
        try:
            # Load consented student embeddings strictly for this session's enrolled cohort
            session_obj = await SessionService.get_by_id(db, session_id)
            target_class = session_obj.target_class if session_obj else None
            enrolled = await StudentService.load_all_consented_embeddings(db, course_name=target_class)
            enrolled_cache = [(s.id, s.full_name, emb) for s, emb in enrolled]

            def update_progress(prog: float, msg: str):
                if job_id in ACTIVE_JOBS:
                    ACTIVE_JOBS[job_id]["progress"] = round(prog, 2)
                    ACTIVE_JOBS[job_id]["message"] = msg

            def cancel_check() -> bool:
                return ACTIVE_JOBS.get(job_id, {}).get("cancelled", False)

            ACTIVE_JOBS[job_id]["message"] = "Processing video frames..."
            result = await pipeline.process_video_file(
                db=db,
                session_id=session_id,
                video_filepath=video_path,
                enrolled_cache=enrolled_cache,
                progress_callback=update_progress,
                cancel_check=cancel_check
            )

            # Auto-generate teaching suggestions from actual processed windows
            try:
                await suggestion_service.generate_and_save_for_session(db, session_id)
            except Exception as e:
                print(f"[VideoProcessing] Notice generating suggestions: {e}")

            # Mark session as completed
            await SessionService.update_status(db, session_id, "COMPLETED")

            ACTIVE_JOBS[job_id]["status"] = "COMPLETED"
            ACTIVE_JOBS[job_id]["progress"] = 1.0
            ACTIVE_JOBS[job_id]["message"] = "Video processing completed successfully."
            ACTIVE_JOBS[job_id]["result"] = result

        except Exception as e:
            ACTIVE_JOBS[job_id]["status"] = "FAILED"
            ACTIVE_JOBS[job_id]["message"] = f"Processing error: {str(e)}"
        finally:
            # Privacy rule: Clean up uploaded video file if not keeping raw media
            if not settings.RAW_MEDIA_PERSISTENCE and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except OSError:
                    pass

@router.post("/upload-and-process")
async def upload_and_process_video(
    background_tasks: BackgroundTasks,
    session_id: int = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    session = await SessionService.get_by_id(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{ext}'. Allowed: .mp4, .avi, .mov, .mkv, .webm"
        )

    # Save video to temporary upload path
    job_id = str(uuid.uuid4())
    temp_filename = f"video_{session_id}_{job_id}{ext}"
    temp_path = settings.UPLOAD_DIR / temp_filename

    contents = await file.read()
    with open(temp_path, "wb") as f:
        f.write(contents)

    # Update session status
    await SessionService.update_status(db, session_id, "LIVE")

    ACTIVE_JOBS[job_id] = {
        "job_id": job_id,
        "session_id": session_id,
        "status": "QUEUED",
        "progress": 0.0,
        "message": "Video queued for analysis",
        "cancelled": False
    }

    # Launch processing in background worker
    background_tasks.add_task(run_video_processing_task, job_id, session_id, str(temp_path))

    return {
        "job_id": job_id,
        "session_id": session_id,
        "status": "QUEUED",
        "message": "Video uploaded and analysis task launched in background."
    }

@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    job = ACTIVE_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job

@router.post("/cancel/{job_id}")
async def cancel_job(job_id: str):
    job = ACTIVE_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    job["cancelled"] = True
    job["status"] = "CANCELLED"
    job["message"] = "Job cancelled by instructor."
    return {"status": "success", "message": "Job cancellation signal sent."}

@router.post("/live-frame/{session_id}")
async def process_live_frame(
    session_id: int,
    file: UploadFile = File(...),
    timestamp_sec: float = Form(0.0),
    db: AsyncSession = Depends(get_db)
):
    """Processes a single live webcam frame sent by the frontend during a live session."""
    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid frame image format.")

    session_obj = await SessionService.get_by_id(db, session_id)
    target_class = session_obj.target_class if session_obj else None
    enrolled = await StudentService.load_all_consented_embeddings(db, course_name=target_class)
    enrolled_cache = [(s.id, s.full_name, emb) for s, emb in enrolled]

    # Detect
    detections = pipeline.detector.detect(frame)
    tracks = pipeline.tracker.update(detections)

    recognized_names = []
    forward_count = 0

    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            face_crop = pipeline.face_recognizer.detect_face(crop)
            if face_crop is not None:
                is_fwd, yaw, pitch, roll = pipeline.head_pose.estimate_orientation(face_crop)
                if is_fwd:
                    forward_count += 1
                
                if track.matched_student_id is None and enrolled_cache:
                    emb = pipeline.face_recognizer.extract_embedding(face_crop)
                    if emb is not None:
                        s_id, s_name, score = pipeline.face_recognizer.match_face(emb, enrolled_cache)
                        if s_id is not None:
                            track.matched_student_id = s_id
                            track.matched_student_name = s_name
                            track.face_confidence = score

        await AttendanceService.register_or_update_detection(
            db=db,
            session_id=session_id,
            track_id=track.track_id,
            matched_student_id=track.matched_student_id,
            confidence=track.face_confidence if track.matched_student_id else track.score,
            timestamp_sec=timestamp_sec,
            frame_interval_sec=0.5
        )

        recognized_names.append(track.matched_student_name or f"Person #{track.track_id}")

    return {
        "status": "success",
        "active_persons": len(tracks),
        "recognized_identities": recognized_names,
        "forward_facing_count": forward_count,
        "timestamp_sec": timestamp_sec
    }
