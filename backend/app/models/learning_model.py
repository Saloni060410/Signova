from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime, timezone
from app.core.database import Base

class LearningVideo(Base):
    __tablename__ = "learning_videos"

    id = Column(Integer, primary_key=True, index=True)
    category = Column(String, index=True, nullable=False) # e.g., "alphabet", "word"
    title = Column(String, index=True, nullable=False)    # e.g., "J", "Hello"
    video_id = Column(String, nullable=False)             # YouTube Video ID
    description = Column(String, nullable=True)           # "How to sign" text
    start_time = Column(Integer, default=0)               # Starting seconds for the video
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
