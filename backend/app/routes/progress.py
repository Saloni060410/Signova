from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.database import get_db
from app.models.progress_model import Progress
from app.schemas.user_schema import UserProgressResponse, ProgressRecord

router = APIRouter()


@router.get("/{user_id}", response_model=UserProgressResponse)
async def get_progress(user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Progress)
        .where(Progress.user_id == user_id)
        .order_by(Progress.completed_at.desc())
    )
    records = result.scalars().all()

    if not records:
        return UserProgressResponse(
            overall_progress=0,
            quizzes_completed=0,
            accuracy_score=0,
            beginner_progress=0,
            medium_progress=0,
            hard_progress=0,
            records=[],
        )

    total_quizzes = len(records)
    avg_score = sum(r.score for r in records) / total_quizzes

    def level_progress(level: str) -> float:
        lvl_records = [r for r in records if r.level == level]
        if not lvl_records:
            return 0.0
        completed_sub = len({r.sub_quiz for r in lvl_records})
        return round((completed_sub / 5) * 100, 1)

    # Overall = average of 3 level progresses
    bp = level_progress("beginner")
    mp_ = level_progress("medium")
    hp = level_progress("hard")
    overall = round((bp + mp_ + hp) / 3, 1)

    accuracy = round(
        sum(r.correct_answers for r in records)
        / max(1, sum(r.total_questions for r in records))
        * 100,
        1,
    )

    return UserProgressResponse(
        overall_progress=overall,
        quizzes_completed=total_quizzes,
        accuracy_score=accuracy,
        beginner_progress=bp,
        medium_progress=mp_,
        hard_progress=hp,
        records=[ProgressRecord.model_validate(r) for r in records[:20]],
    )
