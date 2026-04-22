from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    APP_NAME: str = "SignLang AI"
    DEBUG: bool = True

    # JWT
    SECRET_KEY: str = "CHANGE-ME-IN-PRODUCTION-use-a-long-random-secret"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # Database (SQLite for dev; swap DATABASE_URL in .env for Postgres)
    DATABASE_URL: str = "sqlite+aiosqlite:///./signlang.db"

    # ML Model paths  (place your files here once trained)
    MODEL_PATH: str = "models/sign_model.h5"
    LABEL_MAP_PATH: str = "models/label_info.json"

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
