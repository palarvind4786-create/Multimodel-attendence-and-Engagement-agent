import os
import datetime
import numpy as np
from typing import List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from fastapi import HTTPException, status

from app.models.db_models import Student, ConsentLog
from app.schemas.student import StudentCreate, StudentUpdate, StudentConsentUpdate
from app.core.config import settings

class StudentService:
    @staticmethod
    async def get_all(db: AsyncSession, include_archived: bool = False) -> List[Student]:
        query = select(Student)
        if not include_archived:
            query = query.where(Student.is_archived == False)
        result = await db.execute(query.order_by(Student.id.desc()))
        return list(result.scalars().all())

    @staticmethod
    async def get_by_id(db: AsyncSession, student_pk: int) -> Optional[Student]:
        result = await db.execute(select(Student).where(Student.id == student_pk))
        return result.scalars().first()

    @staticmethod
    async def get_by_student_id(db: AsyncSession, student_id: str) -> Optional[Student]:
        result = await db.execute(select(Student).where(Student.student_id == student_id))
        return result.scalars().first()

    @staticmethod
    async def create(db: AsyncSession, data: StudentCreate) -> Student:
        existing = await StudentService.get_by_student_id(db, data.student_id)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Student ID '{data.student_id}' is already registered."
            )

        student = Student(
            student_id=data.student_id,
            full_name=data.full_name,
            course_name=data.course_name,
            email=data.email,
            has_consented=data.has_consented,
            consent_date=datetime.datetime.utcnow() if data.has_consented else None
        )
        db.add(student)
        await db.flush()

        # Log consent action if consented at creation
        if data.has_consented:
            consent_log = ConsentLog(
                student_id=student.id,
                consent_action="GRANTED",
                notes="Initial consent granted during student registration"
            )
            db.add(consent_log)

        await db.commit()
        await db.refresh(student)
        return student

    @staticmethod
    async def update(db: AsyncSession, student_pk: int, data: StudentUpdate) -> Student:
        student = await StudentService.get_by_id(db, student_pk)
        if not student:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found.")

        update_dict = data.model_dump(exclude_unset=True)
        for key, value in update_dict.items():
            setattr(student, key, value)

        await db.commit()
        await db.refresh(student)
        return student

    @staticmethod
    async def update_consent(db: AsyncSession, student_pk: int, consent_data: StudentConsentUpdate) -> Student:
        student = await StudentService.get_by_id(db, student_pk)
        if not student:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found.")

        student.has_consented = consent_data.has_consented
        student.consent_date = datetime.datetime.utcnow() if consent_data.has_consented else None

        # If consent is revoked, securely delete stored facial embedding
        if not consent_data.has_consented and student.embedding_path:
            if os.path.exists(student.embedding_path):
                try:
                    os.remove(student.embedding_path)
                except OSError:
                    pass
            student.embedding_path = None

        consent_log = ConsentLog(
            student_id=student.id,
            consent_action="GRANTED" if consent_data.has_consented else "REVOKED",
            notes=consent_data.notes or "Consent status updated by instructor"
        )
        db.add(consent_log)
        await db.commit()
        await db.refresh(student)
        return student

    @staticmethod
    async def store_face_embedding(db: AsyncSession, student_pk: int, embedding: np.ndarray, image_bytes: Optional[bytes] = None) -> Student:
        student = await StudentService.get_by_id(db, student_pk)
        if not student:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found.")

        if not student.has_consented:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Biometric consent has not been granted for this student. Face enrollment is prohibited."
            )

        # Save normalized embedding to secure local directory
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        emb_filename = f"student_{student.id}_{student.student_id}.npy"
        emb_path = settings.EMBEDDINGS_DIR / emb_filename
        np.save(str(emb_path), embedding.astype(np.float32))

        student.embedding_path = str(emb_path)

        if image_bytes:
            img_filename = f"avatar_{student.id}.jpg"
            img_path = settings.UPLOAD_DIR / img_filename
            with open(str(img_path), "wb") as f:
                f.write(image_bytes)
            student.profile_image_path = f"/api/v1/students/{student.id}/photo"

        await db.commit()
        await db.refresh(student)
        return student

    @staticmethod
    async def delete(db: AsyncSession, student_pk: int) -> bool:
        student = await StudentService.get_by_id(db, student_pk)
        if not student:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found.")

        # Clean up embedding file
        if student.embedding_path and os.path.exists(student.embedding_path):
            try:
                os.remove(student.embedding_path)
            except OSError:
                pass

        await db.delete(student)
        await db.commit()
        return True

    @staticmethod
    async def load_all_consented_embeddings(db: AsyncSession, course_name: Optional[str] = None) -> List[Tuple[Student, np.ndarray]]:
        """Loads valid embeddings in memory for high-speed local cosine matching during sessions."""
        query = select(Student).where(
            Student.is_archived == False,
            Student.has_consented == True,
            Student.embedding_path != None
        )
        if course_name:
            query = query.where(Student.course_name == course_name)
        result = await db.execute(query)
        students = result.scalars().all()
        
        enrolled_cache = []
        for s in students:
            if s.embedding_path and os.path.exists(s.embedding_path):
                try:
                    vec = np.load(s.embedding_path)
                    enrolled_cache.append((s, vec))
                except Exception:
                    continue
        return enrolled_cache
