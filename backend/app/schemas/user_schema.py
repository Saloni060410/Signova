from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime


# ── Auth ──────────────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    name: str = Field(..., min_length=2)
    email: EmailStr
    password: str = Field(..., min_length=6)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    name: str
    email: str
    created_at: datetime

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# ── Quiz ──────────────────────────────────────────────────────────────────────

class QuizQuestion(BaseModel):
    id: int
    type: str               # "static" | "video" | "camera"
    sign: str
    options: Optional[List[str]] = None
    correct: Optional[int] = None
    image_url: Optional[str] = None
    video_url: Optional[str] = None


class QuizQuestionsResponse(BaseModel):
    level: str
    sub_quiz: int
    questions: List[QuizQuestion]


class QuizSubmitRequest(BaseModel):
    user_id: int
    level: str
    sub_quiz: int
    answers: List[int]          # index of chosen option per question
    total_questions: int


class QuizSubmitResponse(BaseModel):
    score: float
    correct_answers: int
    total_questions: int
    message: str


# ── Prediction ────────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    prediction: str
    confidence: Optional[float] = None


class DynamicPredictionResponse(BaseModel):
    ready: bool
    frames_collected: int
    frames_required: int
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    hand_detected: bool


class ModelStatusResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_loaded: bool
    is_trained_weights: bool      # True when real .pth/.pth.zip weights were used
    checkpoint_source: Optional[str] = None   # filename of the loaded checkpoint
    num_classes: int
    classes: List[str]
    device: str


# ── Progress ──────────────────────────────────────────────────────────────────

class ProgressRecord(BaseModel):
    level: str
    sub_quiz: int
    score: float
    correct_answers: int
    total_questions: int
    completed_at: datetime

    model_config = {"from_attributes": True}


class UserProgressResponse(BaseModel):
    overall_progress: float          # 0-100
    quizzes_completed: int
    accuracy_score: float
    beginner_progress: float
    medium_progress: float
    hard_progress: float
    records: List[ProgressRecord]


# ── Dashboard ─────────────────────────────────────────────────────────────────

class SubQuizStatus(BaseModel):
    sub_quiz: int
    completed: bool
    best_score: Optional[float]


class LevelStatus(BaseModel):
    level: str
    completed_count: int
    total_count: int = 5
    sub_quizzes: List[SubQuizStatus]


class DashboardResponse(BaseModel):
    user: UserOut
    overall_progress: float
    streak_days: int
    total_signs_detected: int
    level_statuses: List[LevelStatus]
    recent_activity: List[ProgressRecord]
