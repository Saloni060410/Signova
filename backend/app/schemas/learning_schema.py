from pydantic import BaseModel
from typing import Optional, List

class LearningVideoBase(BaseModel):
    category: str
    title: str
    video_id: str
    description: Optional[str] = None
    start_time: int = 0

class LearningVideoResponse(LearningVideoBase):
    id: int

    class Config:
        from_attributes = True

class LearningVideosListResponse(BaseModel):
    videos: List[LearningVideoResponse]
