# Signova

AI Sign Language Learning & Detection Platform built with a Next.js frontend and a FastAPI backend.

It combines live ASL detection, quiz-based practice, learning content, and progress tracking in one app.

## Highlights

- Live camera detection with both static image prediction and dynamic sequence prediction.
- Sentence controls for building captured output with Enter, Delete, and Clear actions.
- Learning pages for alphabets, numbers, and common words.
- Quiz flow with beginner, medium, and hard levels.
- Authenticated dashboard and progress summaries.

## Tech Stack

- Frontend: Next.js 16, React 19, TypeScript, Tailwind CSS v4, shadcn/ui.
- Backend: FastAPI, SQLAlchemy, SQLite, Pydantic v2.
- ML: PyTorch, MediaPipe, OpenCV, NumPy, Pillow.

## Repository Layout

The repo is organized around three main areas:

- Application code: `frontend/` and `backend/`
- Model assets: `MPR_STATIC_M/` and `new_dynamic/`
- Generated/runtime data: `.venv/`, `backend/.model_cache/`, `*.db`, logs, screenshots, and build output

```text
MPR_SEM_4/
├── README.md
├── frontend/                       # Next.js app
│   ├── app/                        # routes/pages
│   ├── components/                # UI sections and shared components
│   ├── contexts/                  # auth state
│   ├── hooks/                     # client hooks
│   ├── lib/                       # helpers
│   ├── public/                    # static public assets
│   └── package.json               # frontend scripts and dependencies
├── backend/                        # FastAPI API server
│   ├── app/
│   │   ├── core/                  # database and config
│   │   ├── models/                # ORM models
│   │   ├── routes/                # auth, quiz, predict, progress, dashboard, learning
│   │   ├── schemas/               # Pydantic request/response models
│   │   ├── services/              # static and dynamic inference services
│   │   └── utils/                 # auth helpers and utilities
│   ├── server.py                  # Uvicorn entry point
│   ├── requirements.txt           # backend dependencies
│   └── test_signup.py             # simple backend smoke test
├── MPR_STATIC_M/                   # static ASL model assets
├── new_dynamic/                    # dynamic ASL model assets
└── .gitignore
```

### Not committed to git

These files and folders are intentionally ignored or generated locally:

- Environment files: `backend/.env`, `frontend/.env.local`, and any other `.env*` secrets files.
- Python environments and caches: `.venv/`, `venv/`, `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.pyright/`.
- Database files: `*.db`, `*.sqlite`, `*.sqlite3`.
- Frontend build output: `frontend/.next/`, `frontend/out/`, `frontend/build/`, `node_modules/`, `frontend/node_modules/`.
- Large model weights and bundles: `*.pth`, `*.pth.zip`, `*.pt`, `*.h5`, `*.ckpt`, `*.safetensors`, `*.onnx`, `*.bin`, `*.pb`, `*.tflite`, `*.task`.
- Runtime caches and generated model artifacts: `backend/.model_cache/`.
- Logs and temporary files: `runtime_logs/`, `*.log`, `*.out`, `*.err`, `*.tmp`, `*.download`, `*.bak`.
- Captured media: `MPR_STATIC_M/screenshots/`, `screenshots/`, `captures/`.

The repo currently also ignores `pnpm-lock.yaml`, so package management is expected through npm or pnpm without committing that lockfile.

## Model Assets

The repository uses two inference pipelines:

- Static image model: assets in `MPR_STATIC_M/`.
- Dynamic sequence model: assets in `new_dynamic/`.

Required or expected files:

| File | Purpose | Notes |
|---|---|---|
| `MPR_STATIC_M/asl_classes.json` | Static class labels | Required |
| `MPR_STATIC_M/asl_resnet50.pth` or `MPR_STATIC_M/asl_resnet50.pth.zip` | Static model weights | Required for real trained predictions |
| `MPR_STATIC_M/hand_landmarker.task` | MediaPipe hand landmark model | Used by both pipelines; downloaded to cache if missing |
| `new_dynamic/models/best_model.pth` | Dynamic sequence model | Required |
| `new_dynamic/labels.json` | Dynamic class labels | Required |
| `new_dynamic/models/training_config.json` | Dynamic model config | Required |

Runtime cache files are stored under `backend/.model_cache/`.

## Prerequisites

- Python 3.10 or 3.11.
- Node.js 18+.
- A webcam and browser camera permission for live detection.
- Internet access on first backend start if the MediaPipe hand landmarker needs to be downloaded.

## Setup

### Backend

```bash
cd backend

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python server.py
```

Backend defaults:

- API URL: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`

Check model status:

```bash
curl http://localhost:8000/model/status?mode=static
curl http://localhost:8000/model/status?mode=dynamic
```

### Frontend

```bash
cd frontend

npm install
npm run dev
```

Open `http://localhost:3000`.

If you use a different backend URL, set `NEXT_PUBLIC_API_URL` in `frontend/.env.local`.

## Environment Variables

### Backend

Create `backend/.env` with values like these:

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite+aiosqlite:///./signlang.db` | Async SQLite database URL |
| `SECRET_KEY` | `CHANGE-ME-IN-PRODUCTION-use-a-long-random-secret` | JWT signing secret |
| `ALGORITHM` | `HS256` | JWT algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `10080` | JWT lifetime in minutes |

### Frontend

| Variable | Default | Description |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend API base URL |

## Backend API

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Basic health message |
| `GET` | `/health` | Health check |
| `GET` | `/model/status?mode=static` | Static model status |
| `GET` | `/model/status?mode=dynamic` | Dynamic model status |
| `POST` | `/predict` | Static prediction from uploaded image |
| `POST` | `/predict/base64` | Static prediction from base64 image |
| `POST` | `/predict/dynamic` | Dynamic prediction from uploaded image |
| `POST` | `/predict/dynamic/reset` | Reset a dynamic prediction session |
| `POST` | `/predict/sentence/enter` | Append a token to a sentence buffer |
| `POST` | `/predict/sentence/delete` | Remove the latest sentence token |
| `POST` | `/predict/sentence/clear` | Clear the sentence buffer |
| `POST` | `/auth/signup` | Register a new user |
| `POST` | `/auth/login` | Log in and receive a JWT |
| `GET` | `/quiz/questions` | Fetch quiz questions |
| `POST` | `/quiz/submit` | Submit quiz answers |
| `GET` | `/progress/{user_id}` | Fetch user progress |
| `GET` | `/dashboard/{user_id}` | Fetch dashboard summary |
| `GET` | `/learning/videos` | Fetch learning videos |

Note: `/auth/me` exists in the codebase but currently returns a 501 stub response.

## How It Works

### Static prediction flow

```text
Camera frame
  → Browser capture as JPEG
  → POST /predict
  → MediaPipe hand landmarks
  → Landmark normalization
  → ResNet-based classifier
  → Prediction label + confidence
```

### Dynamic prediction flow

```text
Camera frame stream
  → POST /predict/dynamic
  → MediaPipe hand + pose landmarks
  → 30-frame sequence buffer
  → LSTM classifier
  → Smoothed gesture prediction
```

## Frontend Pages

- `/` landing page with hero, process overview, developer section, and feedback section.
- `/detection` live detection UI with static and dynamic modes.
- `/learning` learning guide for alphabets, numbers, and words.
- `/quiz` quiz dashboard with level selection.
- `/dashboard` authenticated progress overview.

## Notes

- The backend will auto-download the MediaPipe hand landmarker if the local asset is missing.
- Static model status may fall back to a placeholder checkpoint if trained weights are not available.
- Static sentence mode treats the token `space` as a literal space character.


