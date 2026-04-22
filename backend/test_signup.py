import asyncio
import sys
from app.core.database import init_db, AsyncSessionLocal
from app.models.user_model import User
from app.schemas.user_schema import SignupRequest
from app.utils.auth_utils import hash_password, create_access_token

async def test_signup():
    try:
        await init_db()
        async with AsyncSessionLocal() as session:
            user = User(
                name="Test User",
                email="test_direct@example.com",
                hashed_password=hash_password("password123"),
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            print(f"User created: {user.id}, {user.email}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

asyncio.run(test_signup())
