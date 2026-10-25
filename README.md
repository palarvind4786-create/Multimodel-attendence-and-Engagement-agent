# Multimodal Attendance & Engagement Agent

An offline-first, privacy-preserving classroom intelligence application that combines automated attendance verification, observable engagement signal analysis, session telemetry, and context-aware pedagogical suggestions.

---

## 🚀 Live Streamlit Web App (1-Click Deployment)

This repository includes a native, full-featured **Streamlit Web Application** ready for free 1-click cloud deployment on **Streamlit Community Cloud** or local execution!

### ☁️ Deploy on Streamlit Community Cloud:
1. Go to **[share.streamlit.io](https://share.streamlit.io/)** and sign in with GitHub.
2. Click **Create app** > **I already have an app**.
3. Select your repository: `palarvind4786-create/Multimodel-attendence-and-Engagement-agent`
4. Set Branch: `main`
5. Set Main file path: `streamlit_app.py`
6. Click **Deploy!** 🎉

### 💻 Run Streamlit App Locally:
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 1. System Architecture

```
                                 +------------------------------+
                                 |   React + Vite UI Frontend   |
                                 |  (Tailwind CSS, Recharts)    |
                                 +--------------+---------------+
                                                |
                                                | HTTP REST / Polling
                                                v
                                 +------------------------------+
                                 |    FastAPI Python Backend    |
                                 |   (Async SQLAlchemy, SQLite) |
                                 +--------------+---------------+
                                                |
               +--------------------------------+--------------------------------+
               |                                |                                |
               v                                v                                v
     [ Computer Vision ]               [ Audio Analytics ]             [ AI Teaching Advisor ]
 - YOLOv8 Person Detection           - RMS Energy Tracking            - Local Ollama LLM (optional)
 - Multi-Person Spatial Tracking    - Zero-Crossing VAD Rate         - Deterministic Rule Engine
 - FaceNet 512-dim Embeddings       - Aggregate Speech Windows       - Observable Evidence Logs
 - solvePnP 3D Head Orientation      - (No Speaker Diarization)       - Educational Interventions
```

---

## 2. Key Modules & Capabilities

1. **Dashboard Overview**: Displays actual database telemetry including total enrolled students, sessions conducted, sustained attendance percentages, average interaction scores, and comparative trends over recent sessions.
2. **Student Management & Consent Workflow**: Explicit biometric consent must be granted before facial feature extraction. Embeddings are stored locally in isolated `.npy` vectors and are never exposed over public APIs.
3. **Attendance Verification**:
   - Enforces the **sustained-presence rule**: Never marks present on a single frame (requires $\ge 5$ observations).
   - Accurately distinguishes `PRESENT`, `LATE` (arriving after configured threshold), `LEFT_EARLY`, `UNVERIFIED` (unknown person or match below $0.65$ cosine similarity), `PENDING`, and `ABSENT`.
   - Instructor manual override workflow with audit notes.
4. **Observable Engagement Signals**:
   - 3D Head-Pose Estimation using Perspective-n-Point (`solvePnP`) to calculate yaw, pitch, and roll angles to determine forward orientation.
   - Movement intensity via normalized frame differencing.
   - Aggregate voice-activity detection (VAD).
   - Metrics are aggregated across 30-second time windows as observable behavioral indicators (not internal mental state).
5. **Session Analytics & CSV Export**: Interactive multi-series charts (Recharts) with downloadable CSV reports.
6. **AI Teaching Suggestions**: Synthesizes session metrics into actionable pedagogical check-ins via local Ollama or a deterministic rule-based fallback engine.
7. **Dual-Modality Ingestion**: Supports live webcam input and background recorded-video processing with progress bars and cancellation controls.

---

## 3. Technology Stack

- **Backend**: Python 3.11+ / 3.13, FastAPI, Uvicorn, SQLAlchemy (Async SQLite), Ultralytics (YOLOv8), Keras-FaceNet, OpenCV, PyTorch (CUDA acceleration enabled), NumPy, SciPy, HTTPX, Pytest.
- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS, Recharts, Lucide Icons.
- **Privacy & Storage**: SQLite, local vector storage, zero cloud dependencies.

---

## 4. Setup & Running Instructions

### Prerequisites
- Python 3.11+ (verified compatible on Python 3.13)
- Node.js v18+ & npm
- (Optional) NVIDIA GPU with CUDA drivers for accelerated neural inference (CPU fallback supported)

### Step 1: Backend Setup
Open a terminal in the project root:
```bash
cd backend

# Install Python dependencies
python -m pip install -r requirements.txt

# (Optional) Seed realistic demo classroom data
python -m app.utils.synthetic_seeding

# Run automated tests to verify your environment
python -m pytest tests/

# Start the FastAPI backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
The backend API documentation is interactively accessible at `http://127.0.0.1:8000/docs`.

### Step 2: Frontend Setup
Open a second terminal:
```bash
cd frontend

# Install npm dependencies
npm install

# Start the Vite development server
npm run dev
```
Open your browser and navigate to:
```
http://127.0.0.1:5173
```

---

## 5. Automated Test Suite

Run unit and integration tests covering database initialization, student CRUD, consent verification, attendance threshold logic, API endpoints, head-pose calculation, and audio analysis:
```bash
cd backend
python -m pytest tests/ -v
```

Verified Test Results:
- `test_single_frame_never_marks_present`: PASSED
- `test_sustained_presence_marks_present`: PASSED
- `test_late_arrival_detection`: PASSED
- `test_left_early_detection`: PASSED
- `test_student_creation_and_duplicate_prevention`: PASSED
- `test_consent_enforcement_for_face_embedding`: PASSED
- `test_session_creation_and_isolation`: PASSED
- `test_health_endpoint`: PASSED
- `test_dashboard_metrics_empty_state`: PASSED
- `test_student_and_session_api_flow`: PASSED
- `test_teaching_suggestions_generation`: PASSED
- `test_head_pose_estimator`: PASSED
- `test_audio_analyzer_energy`: PASSED

---

## 6. API Route Summary

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Hardware acceleration and model telemetry |
| `GET` | `/api/v1/dashboard/metrics` | Live metrics, recent sessions, and signal trends |
| `GET` | `/api/v1/students` | List enrolled students |
| `POST` | `/api/v1/students` | Create new student record |
| `POST` | `/api/v1/students/{id}/enroll-face` | Extract & store FaceNet embedding from image |
| `PUT` | `/api/v1/students/{id}/consent` | Grant or revoke biometric consent |
| `GET` | `/api/v1/sessions` | List sessions with attendance summaries |
| `POST` | `/api/v1/sessions` | Create and isolate a new session cohort |
| `PUT` | `/api/v1/sessions/{id}/status` | Update status (LIVE, PAUSED, COMPLETED) |
| `GET` | `/api/v1/attendance/session/{id}` | Retrieve session attendance records |
| `PUT` | `/api/v1/attendance/{record_id}/override` | Instructor manual attendance correction |
| `GET` | `/api/v1/analytics/session/{id}` | 30s window telemetry & notable changes |
| `GET` | `/api/v1/analytics/session/{id}/export-csv` | Download complete attendance CSV |
| `GET` | `/api/v1/suggestions/session/{id}` | Retrieve pedagogical suggestions |
| `POST` | `/api/v1/suggestions/session/{id}/generate` | Generate context-aware suggestions |
| `POST` | `/api/v1/video/upload-and-process` | Queue video file for background processing |
| `GET` | `/api/v1/video/status/{job_id}` | Poll background processing progress |
| `POST` | `/api/v1/video/cancel/{job_id}` | Cancel active video processing job |
| `POST` | `/api/v1/video/live-frame/{session_id}` | Process single webcam frame in live mode |

---

## 7. Responsible Educational AI & Privacy Guarantees

1. **Zero Cloud Offloading**: Video feeds, audio streams, and biometric embeddings are processed strictly on localhost.
2. **Ephemeral Media Policy**: Uploaded video files and temporary audio extracts are automatically deleted from disk following telemetry extraction.
3. **No Mental State Claims**: Observable signals (orientation, movement, audio activity) are strictly described as behavioral indicators. The software never claims to measure internal thoughts, comprehension, or student intelligence.
4. **Educator Primacy**: Automated suggestions and attendance decisions are advisory. Final decisions and manual adjustments always reside with the educator.
