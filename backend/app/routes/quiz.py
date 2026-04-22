from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import random

from app.core.database import get_db
from app.schemas.user_schema import (
    QuizQuestionsResponse, QuizQuestion,
    QuizSubmitRequest, QuizSubmitResponse,
)
from app.services.quiz_data import QUIZ_BANK
from app.models.progress_model import Progress
from app.models.learning_model import LearningVideo

router = APIRouter()

LEVELS = ["beginner", "medium", "hard"]

HARD_MODEL_SIGN_POOL = [
    "Hello",
    "Thank You",
    "Please",
    "Sorry",
    "I Love You",
    "Bye",
    "Yes",
    "No",
]

ASL_ALPHABET_IMAGE_URLS = {
    "A": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Sign_language_A.svg/250px-Sign_language_A.svg.png",
    "B": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Sign_language_B.svg/250px-Sign_language_B.svg.png",
    "C": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Sign_language_C.svg/250px-Sign_language_C.svg.png",
    "D": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/06/Sign_language_D.svg/250px-Sign_language_D.svg.png",
    "E": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Sign_language_E.svg/250px-Sign_language_E.svg.png",
    "F": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Sign_language_F.svg/250px-Sign_language_F.svg.png",
    "G": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Sign_language_G.svg/250px-Sign_language_G.svg.png",
    "H": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Sign_language_H.svg/250px-Sign_language_H.svg.png",
    "I": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/Sign_language_I.svg/250px-Sign_language_I.svg.png",
    "J": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/Sign_language_J.svg/250px-Sign_language_J.svg.png",
    "K": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Sign_language_K.svg/250px-Sign_language_K.svg.png",
    "L": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/Sign_language_L.svg/250px-Sign_language_L.svg.png",
    "M": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/Sign_language_M.svg/250px-Sign_language_M.svg.png",
    "N": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Sign_language_N.svg/250px-Sign_language_N.svg.png",
    "O": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Sign_language_O.svg/250px-Sign_language_O.svg.png",
    "P": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/Sign_language_P.svg/250px-Sign_language_P.svg.png",
    "Q": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Sign_language_Q.svg/250px-Sign_language_Q.svg.png",
    "R": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Sign_language_R.svg/250px-Sign_language_R.svg.png",
    "S": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Sign_language_S.svg/250px-Sign_language_S.svg.png",
    "T": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/Sign_language_T.svg/250px-Sign_language_T.svg.png",
    "U": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7c/Sign_language_U.svg/250px-Sign_language_U.svg.png",
    "V": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Sign_language_V.svg/250px-Sign_language_V.svg.png",
    "W": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/Sign_language_W.svg/250px-Sign_language_W.svg.png",
    "X": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Sign_language_X.svg/250px-Sign_language_X.svg.png",
    "Y": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Sign_language_Y.svg/250px-Sign_language_Y.svg.png",
    "Z": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/Sign_language_Z.svg/250px-Sign_language_Z.svg.png",
}


def _normalize_word(text: str) -> str:
    return text.strip().lower().replace("_", " ")


@router.get("/questions", response_model=QuizQuestionsResponse)
async def get_questions(level: str, sub_quiz: int, db: AsyncSession = Depends(get_db)):
    level = level.lower()
    if level not in LEVELS:
        raise HTTPException(400, f"level must be one of {LEVELS}")
    if sub_quiz not in range(1, 6):
        raise HTTPException(400, "sub_quiz must be between 1 and 5")

    raw = QUIZ_BANK[level][sub_quiz]

    if level == "hard" and sub_quiz != 1:
        picked = random.sample(HARD_MODEL_SIGN_POOL, k=5)
        raw = [
            {
                "id": 300 + i,
                "type": "camera",
                "sign": sign,
                "options": None,
                "correct": None,
            }
            for i, sign in enumerate(picked, start=1)
        ]
    enriched = [dict(q) for q in raw]

    if level == "beginner":
        for q in enriched:
            sign = str(q.get("sign", "")).strip().upper()
            if len(sign) == 1 and sign in ASL_ALPHABET_IMAGE_URLS:
                q["image_url"] = ASL_ALPHABET_IMAGE_URLS[sign]

    if level == "medium":
        result = await db.execute(select(LearningVideo).where(LearningVideo.category == "word"))
        videos = result.scalars().all()
        video_map = {_normalize_word(v.title): v.video_id for v in videos}
        for q in enriched:
            if q.get("video_url"):
                continue
            sign = _normalize_word(str(q.get("sign", "")))
            video_url = video_map.get(sign)
            if video_url:
                q["video_url"] = video_url

    questions = [QuizQuestion(**q) for q in enriched]
    return QuizQuestionsResponse(level=level, sub_quiz=sub_quiz, questions=questions)


@router.post("/submit", response_model=QuizSubmitResponse)
async def submit_quiz(body: QuizSubmitRequest, db: AsyncSession = Depends(get_db)):
    level = body.level.lower()
    if level not in LEVELS:
        raise HTTPException(400, f"level must be one of {LEVELS}")

    raw = QUIZ_BANK.get(level, {}).get(body.sub_quiz, [])
    if not raw:
        raise HTTPException(404, "Quiz not found")

    # Hard mode – camera-based grading from per-question model scores (0-100)
    if level == "hard":
        hard_scores = [max(0, min(100, int(v))) for v in body.answers[: body.total_questions]]
        if hard_scores:
            score = round(sum(hard_scores) / len(hard_scores), 1)
            # Count strong performances as "correct" for dashboard consistency.
            correct = sum(1 for s in hard_scores if s >= 70)
        else:
            score = 0.0
            correct = 0
    else:
        correct = sum(
            1 for i, ans in enumerate(body.answers)
            if i < len(raw) and ans == raw[i]["correct"]
        )
        score = round((correct / body.total_questions) * 100, 1)

    # Persist result
    record = Progress(
        user_id=body.user_id,
        level=level,
        sub_quiz=body.sub_quiz,
        score=score,
        total_questions=body.total_questions,
        correct_answers=correct,
    )
    db.add(record)
    await db.commit()

    msg = (
        "🏆 Perfect score!" if score == 100
        else "✅ Great job!" if score >= 70
        else "📚 Keep practicing!"
    )
    return QuizSubmitResponse(
        score=score,
        correct_answers=correct,
        total_questions=body.total_questions,
        message=msg,
    )
