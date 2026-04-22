from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from app.core.config import get_settings

settings = get_settings()

engine = create_async_engine(settings.DATABASE_URL, echo=settings.DEBUG)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def init_db():
    """Create all tables on startup."""
    async with engine.begin() as conn:
        from app.models import user_model, progress_model, learning_model  # noqa – registers models
        await conn.run_sync(Base.metadata.create_all)
        
    # Seed the database with default generic videos if empty
    from app.models.learning_model import LearningVideo
    from sqlalchemy import select
    
    async with AsyncSessionLocal() as session:
        # Check if we already have videos
        result = await session.execute(select(LearningVideo).limit(1))
        existing = result.scalars().first()
        
        if existing:
            # Quick patch to fix OGV to MP4 if already seeded
            patch_result = await session.execute(select(LearningVideo).where(LearningVideo.video_id.like("%.ogv%")))
            ogv_vid = patch_result.scalars().first()
            if ogv_vid:
                print("Patching OGV Hello video to MP4...")
                ogv_vid.video_id = "https://media.spreadthesign.com/video/mp4/13/109919.mp4"
                
            # Quick patch to fix Thank You video variant
            patch_result2 = await session.execute(select(LearningVideo).where(LearningVideo.title == "Thank You"))
            ty_vid = patch_result2.scalars().first()
            if ty_vid and ty_vid.video_id != "https://media.spreadthesign.com/video/mp4/13/58476.mp4":
                print("Patching Thank You video variant...")
                ty_vid.video_id = "https://media.spreadthesign.com/video/mp4/13/58476.mp4"
                
            await session.commit()
                
        if not existing:
            print("Seeding initial learning videos into database from Spread the Sign / Wikimedia...")
            default_videos = [
                LearningVideo(category="alphabet", title="J", video_id="https://media.spreadthesign.com/video/mp4/13/alphabet-letter-600-1.mp4", description="Start with the 'I' handshape (pinky up), then trace a J shape downward in the air.", start_time=0),
                LearningVideo(category="alphabet", title="Z", video_id="https://media.spreadthesign.com/video/mp4/13/alphabet-letter-616-1.mp4", description="Point your index finger and trace the letter Z in the air.", start_time=0),
                LearningVideo(category="word", title="Hello", video_id="https://media.spreadthesign.com/video/mp4/13/109919.mp4", description="Bring your dominant hand to your forehead/temple area and move it outward, like a salute.", start_time=0),
                LearningVideo(category="word", title="Thank You", video_id="https://media.spreadthesign.com/video/mp4/13/58476.mp4", description="With your dominant hand flat, touch your fingertips to your chin and move your hand outward toward the person.", start_time=0),
                LearningVideo(category="word", title="Sorry", video_id="https://media.spreadthesign.com/video/mp4/13/94029.mp4", description="Form an 'A' handshape (fist with thumb alongside) and rub it in a circular motion on the center of your chest.", start_time=0),
                LearningVideo(category="word", title="Please", video_id="https://media.spreadthesign.com/video/mp4/13/49200.mp4", description="With your dominant hand flat, place it on your chest and move it in a circular motion.", start_time=0),
            ]
            session.add_all(default_videos)
            await session.commit()
            print("Initial learning videos seeded successfully.")

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
