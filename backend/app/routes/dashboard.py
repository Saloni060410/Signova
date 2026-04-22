from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timezone, timedelta

from app.core.database import get_db
from app.models.user_model import User
from app.models.progress_model import Progress
from app.schemas.user_schema import (
    DashboardResponse, LevelStatus, SubQuizStatus,
    UserOut, ProgressRecord,
)

router = APIRouter()

LEVELS = ["beginner", "medium", "hard"]


@router.get("/{user_id}", response_model=DashboardResponse)
async def get_dashboard(user_id: int, db: AsyncSession = Depends(get_db)):
    # Fetch user
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(404, "User not found")

    # Fetch all progress
    res2 = await db.execute(
        select(Progress)
        .where(Progress.user_id == user_id)
        .order_by(Progress.completed_at.desc())
    )
    records = res2.scalars().all()

    # ── Level statuses ─────────────────────────────────────────────────────
    level_statuses = []
    for level in LEVELS:
        lvl_records = [r for r in records if r.level == level]
        # best score per sub_quiz
        best: dict[int, float] = {}
        for r in lvl_records:
            if r.sub_quiz not in best or r.score > best[r.sub_quiz]:
                best[r.sub_quiz] = r.score

        sub_quizzes = [
            SubQuizStatus(
                sub_quiz=i,
                completed=i in best,
                best_score=best.get(i),
            )
            for i in range(1, 6)
        ]
        level_statuses.append(
            LevelStatus(
                level=level,
                completed_count=len(best),
                sub_quizzes=sub_quizzes,
            )
        )

    # ── Overall progress ───────────────────────────────────────────────────
    total_sub_quizzes = sum(ls.completed_count for ls in level_statuses)
    overall = round((total_sub_quizzes / 15) * 100, 1)

    # ── Streak (consecutive days with at least 1 quiz) ─────────────────────
    activity_days = sorted(
        {r.completed_at.date() for r in records}, reverse=True
    )
    streak = 0
    today = datetime.now(timezone.utc).date()
    for i, day in enumerate(activity_days):
        expected = today - timedelta(days=i)
        if day == expected:
            streak += 1
        else:
            break

    # ── Mock total detections (replace with a real detections table later) ─
    total_detections = len(records) * 12

    return DashboardResponse(
        user=UserOut.model_validate(user),
        overall_progress=overall,
        streak_days=streak,
        total_signs_detected=total_detections,
        level_statuses=level_statuses,
        recent_activity=[ProgressRecord.model_validate(r) for r in records[:5]],
    )
