import pytest
import pytest_asyncio
import numpy as np
import cv2
import tempfile
import os
import uuid
import asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy import select

from app.main import app
from app.core.database import Base, AsyncSessionLocal, engine
from app.models.db_models import Student, Session, AttendanceRecord, EngagementWindow, TeachingSuggestion
from app.services.student_service import StudentService
from app.services.session_service import SessionService
from app.services.attendance_service import AttendanceService
from app.vision.pipeline import MultimodalPipeline

def create_synthetic_test_video(filepath: str, duration_sec: float = 3.0, fps: float = 10.0):
    """Generates a synthetic mp4 test video with synthetic human-like silhouettes."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w, h = 320, 240
    out = cv2.VideoWriter(filepath, fourcc, fps, (w, h))

    total_frames = int(duration_sec * fps)
    for i in range(total_frames):
        frame = np.full((h, w, 3), 200, dtype=np.uint8)
        # Draw a synthetic body & head
        cv2.rectangle(frame, (120, 80), (200, 230), (50, 50, 50), -1)  # Torso
        cv2.circle(frame, (160, 50), 30, (180, 150, 130), -1)          # Head
        cv2.circle(frame, (150, 45), 4, (30, 30, 30), -1)               # Left eye
        cv2.circle(frame, (170, 45), 4, (30, 30, 30), -1)               # Right eye
        cv2.circle(frame, (160, 55), 3, (50, 50, 50), -1)               # Nose
        cv2.line(frame, (152, 65), (168, 65), (40, 40, 40), 2)          # Mouth
        out.write(frame)

    out.release()

@pytest_asyncio.fixture
async def client():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

@pytest.mark.asyncio
async def test_cohort_isolation_prevents_outside_student_confirmation(client):
    """
    Critical Correctness: Students outside the session cohort cannot appear as confirmed attendees.
    """
    uid = uuid.uuid4().hex[:6]
    # Student in Math Cohort
    stu_math = await client.post("/api/v1/students", json={
        "student_id": f"MATH-{uid}",
        "full_name": f"Math Student {uid}",
        "course_name": f"Math-Cohort-{uid}",
        "has_consented": True
    })
    math_id = stu_math.json()["id"]

    # Student in Physics Cohort
    stu_phys = await client.post("/api/v1/students", json={
        "student_id": f"PHYS-{uid}",
        "full_name": f"Physics Student {uid}",
        "course_name": f"Physics-Cohort-{uid}",
        "has_consented": True
    })
    phys_id = stu_phys.json()["id"]

    # Create Session for Math Cohort ONLY
    sess = await client.post("/api/v1/sessions", json={
        "course_title": f"Calculus {uid}",
        "instructor_name": "Dr. Euler",
        "target_class": f"Math-Cohort-{uid}"
    })
    session_id = sess.json()["id"]

    # Verify initial attendance has only Math student
    att_res = await client.get(f"/api/v1/attendance/session/{session_id}")
    records = att_res.json()
    assert len(records) == 1
    assert records[0]["student_id"] == math_id

    # Verify embeddings loaded for this session do NOT include Physics student
    async with AsyncSessionLocal() as db:
        enrolled_math = await StudentService.load_all_consented_embeddings(db, course_name=f"Math-Cohort-{uid}")
        enrolled_student_ids = [s.id for s, _ in enrolled_math]
        assert phys_id not in enrolled_student_ids

@pytest.mark.asyncio
async def test_absent_assigned_only_after_completion(client):
    """
    Critical Correctness: ABSENT is assigned only after completed session,
    never during incomplete or pending processing.
    """
    uid = uuid.uuid4().hex[:6]
    cohort = f"Bio-{uid}"

    # Create 2 enrolled students
    s1 = await client.post("/api/v1/students", json={
        "student_id": f"BIO-1-{uid}",
        "full_name": "Bio Student 1",
        "course_name": cohort,
        "has_consented": True
    })
    s2 = await client.post("/api/v1/students", json={
        "student_id": f"BIO-2-{uid}",
        "full_name": "Bio Student 2",
        "course_name": cohort,
        "has_consented": True
    })

    # Create Session (starts SCHEDULED)
    sess = await client.post("/api/v1/sessions", json={
        "course_title": f"Biology {uid}",
        "instructor_name": "Dr. Darwin",
        "target_class": cohort
    })
    session_id = sess.json()["id"]

    # Initial state must be PENDING, never ABSENT
    att1 = (await client.get(f"/api/v1/attendance/session/{session_id}")).json()
    for rec in att1:
        assert rec["status"] == "PENDING", f"Expected PENDING in scheduled session, got {rec['status']}"

    # Transition to LIVE
    await client.put(f"/api/v1/sessions/{session_id}/status", json={"status": "LIVE"})
    att2 = (await client.get(f"/api/v1/attendance/session/{session_id}")).json()
    for rec in att2:
        assert rec["status"] != "ABSENT", "Students must not be ABSENT while session is LIVE"

    # Transition to COMPLETED
    await client.put(f"/api/v1/sessions/{session_id}/status", json={"status": "COMPLETED"})
    att3 = (await client.get(f"/api/v1/attendance/session/{session_id}")).json()
    for rec in att3:
        assert rec["status"] == "ABSENT", "Unobserved pending students must be ABSENT after COMPLETED"

@pytest.mark.asyncio
async def test_revoked_consent_prevents_face_matching(client):
    """
    Critical Correctness: Revoked biometric consent immediately deletes embedding
    and prevents inclusion in active face matching cache.
    """
    uid = uuid.uuid4().hex[:6]
    cohort = f"Chem-{uid}"

    # Create student with consent
    stu_res = await client.post("/api/v1/students", json={
        "student_id": f"CHEM-{uid}",
        "full_name": f"Chem Student {uid}",
        "course_name": cohort,
        "has_consented": True
    })
    stu_id = stu_res.json()["id"]

    # Store fake embedding
    async with AsyncSessionLocal() as db:
        await StudentService.store_face_embedding(db, stu_id, np.random.randn(512).astype(np.float32))

        # Check it is loaded
        loaded1 = await StudentService.load_all_consented_embeddings(db, course_name=cohort)
        assert any(s.id == stu_id for s, _ in loaded1)

        # Revoke consent
        await StudentService.update_consent(db, stu_id, StudentConsentUpdate(has_consented=False))

        # Must no longer be in loaded cache
        loaded2 = await StudentService.load_all_consented_embeddings(db, course_name=cohort)
        assert not any(s.id == stu_id for s, _ in loaded2)

from app.schemas.student import StudentConsentUpdate

@pytest.mark.asyncio
async def test_e2e_video_processing_flow(client):
    """
    PHASE 2 Full End-to-End Test:
    1. Create Student & Consent
    2. Enroll Face
    3. Create Session with cohort
    4. Upload Video and process in background
    5. Verify SQLite records & API responses
    6. Verify CSV export
    """
    uid = uuid.uuid4().hex[:6]
    cohort = f"E2E-{uid}"

    # 1. Create Student
    stu_res = await client.post("/api/v1/students", json={
        "student_id": f"E2E-STU-{uid}",
        "full_name": f"E2E Attendee {uid}",
        "course_name": cohort,
        "has_consented": True
    })
    assert stu_res.status_code == 201
    stu_id = stu_res.json()["id"]

    # 2. Store face embedding
    fake_emb = np.random.randn(512).astype(np.float32)
    async with AsyncSessionLocal() as db:
        await StudentService.store_face_embedding(db, stu_id, fake_emb)

    # 3. Create Session
    sess_res = await client.post("/api/v1/sessions", json={
        "course_title": f"E2E Vision Class {uid}",
        "instructor_name": "Prof. Vision",
        "room_number": "Lab 1",
        "target_class": cohort,
        "source_type": "VIDEO_UPLOAD"
    })
    assert sess_res.status_code == 201
    session_id = sess_res.json()["id"]

    # 4. Generate synthetic test video file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
        temp_video_path = tmp_video.name

    try:
        create_synthetic_test_video(temp_video_path, duration_sec=2.0, fps=10.0)
        assert os.path.exists(temp_video_path)
        assert os.path.getsize(temp_video_path) > 1000

        # Upload video to endpoint
        with open(temp_video_path, "rb") as vf:
            upload_res = await client.post(
                "/api/v1/video/upload-and-process",
                data={"session_id": str(session_id)},
                files={"file": ("test_lecture.mp4", vf, "video/mp4")}
            )
        assert upload_res.status_code == 200
        job_data = upload_res.json()
        job_id = job_data["job_id"]

        # Wait for background task to complete (or poll status)
        for _ in range(30):
            await asyncio.sleep(0.5)
            status_res = await client.get(f"/api/v1/video/status/{job_id}")
            if status_res.status_code == 200:
                s_data = status_res.json()
                if s_data["status"] in ["COMPLETED", "FAILED"]:
                    break

        # 5. Verify session completed and windows generated
        sess_check = (await client.get(f"/api/v1/sessions/{session_id}")).json()
        assert sess_check["status"] == "COMPLETED"

        # 6. Verify Analytics & Windows
        analytics_res = await client.get(f"/api/v1/analytics/session/{session_id}")
        assert analytics_res.status_code == 200
        analytics_data = analytics_res.json()
        assert analytics_data["windows_count"] >= 1
        assert len(analytics_data["timeline"]) >= 1

        # 7. Verify Teaching Suggestions
        sug_res = await client.get(f"/api/v1/suggestions/session/{session_id}")
        assert sug_res.status_code == 200
        sug_data = sug_res.json()
        assert len(sug_data) >= 1

        # 8. Verify CSV export contains session UID and target class
        csv_res = await client.get(f"/api/v1/analytics/session/{session_id}/export-csv")
        assert csv_res.status_code == 200
        assert sess_check["session_uid"] in csv_res.text

    finally:
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except OSError:
                pass

@pytest.mark.asyncio
async def test_job_cancellation(client):
    """
    Critical Correctness: Cancelled jobs release resources and set CANCELLED status.
    """
    # Create session
    sess_res = await client.post("/api/v1/sessions", json={
        "course_title": "Cancel Test",
        "instructor_name": "Prof. Cancel",
        "target_class": "Cancel-101",
        "source_type": "VIDEO_UPLOAD"
    })
    session_id = sess_res.json()["id"]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
        temp_video_path = tmp_video.name

    try:
        create_synthetic_test_video(temp_video_path, duration_sec=4.0, fps=10.0)

        with open(temp_video_path, "rb") as vf:
            upload_res = await client.post(
                "/api/v1/video/upload-and-process",
                data={"session_id": str(session_id)},
                files={"file": ("long_video.mp4", vf, "video/mp4")}
            )
        job_id = upload_res.json()["job_id"]

        # Send immediate cancellation signal
        cancel_res = await client.post(f"/api/v1/video/cancel/{job_id}")
        assert cancel_res.status_code == 200

        # Verify job marked CANCELLED
        status_res = await client.get(f"/api/v1/video/status/{job_id}")
        assert status_res.json()["status"] == "CANCELLED"
    finally:
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except OSError:
                pass
