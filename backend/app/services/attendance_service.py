from typing import List, Optional
import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import joinedload
from fastapi import HTTPException, status

from app.models.db_models import AttendanceRecord, Student, Session
from app.schemas.attendance import AttendanceUpdate
from app.core.config import settings

class AttendanceService:
    @staticmethod
    async def get_by_session(db: AsyncSession, session_id: int) -> List[dict]:
        query = (
            select(AttendanceRecord)
            .options(joinedload(AttendanceRecord.student))
            .where(AttendanceRecord.session_id == session_id)
            .order_by(AttendanceRecord.id.asc())
        )
        result = await db.execute(query)
        records = result.scalars().all()
        
        output = []
        for r in records:
            output.append({
                "id": r.id,
                "session_id": r.session_id,
                "student_id": r.student_id,
                "student_name": r.student.full_name if r.student else (r.detected_name or "Unknown Person"),
                "student_code": r.student.student_id if r.student else None,
                "tracking_track_id": r.tracking_track_id,
                "detected_name": r.detected_name,
                "status": r.status,
                "first_seen_seconds": r.first_seen_seconds,
                "last_seen_seconds": r.last_seen_seconds,
                "sustained_duration_seconds": round(r.sustained_duration_seconds, 1),
                "observation_count": r.observation_count,
                "avg_confidence": round(r.avg_confidence, 2),
                "is_manual_override": r.is_manual_override,
                "notes": r.notes,
                "created_at": r.created_at,
                "updated_at": r.updated_at
            })
        return output

    @staticmethod
    async def manual_override(db: AsyncSession, record_id: int, update_data: AttendanceUpdate) -> AttendanceRecord:
        result = await db.execute(select(AttendanceRecord).where(AttendanceRecord.id == record_id))
        record = result.scalars().first()
        if not record:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Attendance record not found.")

        record.status = update_data.status
        record.is_manual_override = True
        if update_data.notes:
            record.notes = (record.notes or "") + f" [Override: {update_data.notes}]"
        
        await db.commit()
        await db.refresh(record)
        return record

    @staticmethod
    def evaluate_attendance_status(
        first_seen_sec: float,
        last_seen_sec: float,
        sustained_duration_sec: float,
        observation_count: int,
        session_elapsed_sec: float,
        min_observations: int = settings.MIN_ATTENDANCE_OBSERVATIONS,
        late_threshold_sec: float = settings.LATE_ARRIVAL_MINUTES * 60.0
    ) -> str:
        """
        Pure rule-based attendance determination ensuring:
        - Never marks PRESENT on a single observation.
        - Requires observation_count >= min_observations.
        - Evaluates arrival timing (LATE) and departure (LEFT_EARLY).
        """
        if observation_count < min_observations:
            return "UNVERIFIED"

        # Check late arrival
        if first_seen_sec > late_threshold_sec:
            return "LATE"

        # Check premature departure if session has substantial duration (> 15 mins)
        if session_elapsed_sec > 900 and (session_elapsed_sec - last_seen_sec) > 600:
            if sustained_duration_sec < (session_elapsed_sec * 0.4):
                return "LEFT_EARLY"

        return "PRESENT"

    @staticmethod
    async def register_or_update_detection(
        db: AsyncSession,
        session_id: int,
        track_id: int,
        matched_student_id: Optional[int],
        confidence: float,
        timestamp_sec: float,
        frame_interval_sec: float = 0.5
    ) -> AttendanceRecord:
        """
        Incrementally accumulates tracker sightings and face matches.
        Enforces no duplicate counting for the same student in a session.
        """
        record: Optional[AttendanceRecord] = None

        if matched_student_id is not None:
            # Check if this student already has a record in this session
            res = await db.execute(
                select(AttendanceRecord).where(
                    AttendanceRecord.session_id == session_id,
                    AttendanceRecord.student_id == matched_student_id
                )
            )
            record = res.scalars().first()

            # Clean up prior unverified placeholder for this track_id if it existed
            unv_res = await db.execute(
                select(AttendanceRecord).where(
                    AttendanceRecord.session_id == session_id,
                    AttendanceRecord.tracking_track_id == track_id,
                    AttendanceRecord.student_id == None
                )
            )
            unv_record = unv_res.scalars().first()
            if unv_record:
                if record:
                    record.observation_count += unv_record.observation_count
                    record.sustained_duration_seconds += unv_record.sustained_duration_seconds
                await db.delete(unv_record)
        else:
            # Check if this track_id already has an unverified record
            res = await db.execute(
                select(AttendanceRecord).where(
                    AttendanceRecord.session_id == session_id,
                    AttendanceRecord.tracking_track_id == track_id,
                    AttendanceRecord.student_id == None
                )
            )
            record = res.scalars().first()

        now_utc = datetime.datetime.utcnow()

        if not record:
            record = AttendanceRecord(
                session_id=session_id,
                student_id=matched_student_id,
                tracking_track_id=track_id,
                detected_name=None,
                status="UNVERIFIED" if matched_student_id is None else "PENDING",
                first_seen_seconds=timestamp_sec,
                last_seen_seconds=timestamp_sec,
                sustained_duration_seconds=frame_interval_sec,
                observation_count=1,
                avg_confidence=confidence,
                created_at=now_utc,
                updated_at=now_utc
            )
            db.add(record)
        else:
            if not record.is_manual_override:
                # Update observation metrics
                record.observation_count += 1
                if record.first_seen_seconds is None:
                    record.first_seen_seconds = timestamp_sec
                record.last_seen_seconds = timestamp_sec
                record.sustained_duration_seconds += frame_interval_sec
                
                # Rolling average confidence
                record.avg_confidence = (
                    (record.avg_confidence * (record.observation_count - 1) + confidence)
                    / record.observation_count
                )

                # Link track ID if updated
                record.tracking_track_id = track_id

                # Evaluate verified status if student is known
                if record.student_id is not None:
                    record.status = AttendanceService.evaluate_attendance_status(
                        first_seen_sec=record.first_seen_seconds,
                        last_seen_sec=record.last_seen_seconds,
                        sustained_duration_sec=record.sustained_duration_seconds,
                        observation_count=record.observation_count,
                        session_elapsed_sec=timestamp_sec
                    )
                else:
                    record.status = "UNVERIFIED"
                record.updated_at = now_utc

        await db.commit()
        await db.refresh(record)
        return record
