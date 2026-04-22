from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.database import init_db
from app.routes import auth, quiz, predict, progress, dashboard
from app.services.model_service import initialize_model_service, shutdown_model_service
from app.services.dynamic_model_service import (
    initialize_dynamic_model_service,
    shutdown_dynamic_model_service,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    try:
        initialize_model_service()
    except Exception as e:
        print(f"WARNING: Model failed to initialize. Running without ML. Error: {e}")
    try:
        initialize_dynamic_model_service()
    except Exception as e:
        print(f"WARNING: Dynamic model failed to initialize. Running without dynamic ML. Error: {e}")
    yield
    try:
        shutdown_model_service()
    except Exception:
        pass
    try:
        shutdown_dynamic_model_service()
    except Exception:
        pass


app = FastAPI(
    title="SignLang AI API",
    description="Backend for AI Sign Language Learning & Detection Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.routes import auth, quiz, predict, progress, dashboard, learning
app.include_router(auth.router,      prefix="/auth",      tags=["Auth"])
app.include_router(quiz.router,      prefix="/quiz",      tags=["Quiz"])
app.include_router(predict.router,   prefix="",           tags=["Prediction"])
app.include_router(progress.router,  prefix="/progress",  tags=["Progress"])
app.include_router(dashboard.router, prefix="/dashboard", tags=["Dashboard"])
app.include_router(learning.router,  prefix="/learning",  tags=["Learning"])


@app.get("/")
async def root():
    return {"message": "SignLang AI API is running 🤟"}


@app.get("/health")
async def health():
    return {"status": "ok"}
