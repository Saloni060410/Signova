from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional

from app.core.database import get_db
from app.models.learning_model import LearningVideo
from app.schemas.learning_schema import LearningVideosListResponse

router = APIRouter()

@router.get("/videos", response_model=LearningVideosListResponse)
async def get_learning_videos(
    category: Optional[str] = Query(None, description="Filter videos by category (e.g. alphabet, word)"),
    db: AsyncSession = Depends(get_db)
):
    query = select(LearningVideo)
    if category:
        query = query.where(LearningVideo.category == category)
        
    result = await db.execute(query)
    videos = result.scalars().all()
    
    return LearningVideosListResponse(videos=list(videos))
