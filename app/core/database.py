from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from motor.motor_asyncio import AsyncIOMotorClient   #the async MongoDB driver
import redis.asyncio as redis
from typing import AsyncGenerator
from app.core.config import settings

Base = declarative_base()


# PostgreSQL
engine = create_async_engine(settings.postgres_url.replace('postgresql://', 'postgresql+asyncpg://'))
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# MongoDB
mongodb_client = AsyncIOMotorClient(settings.mongodb_url)
mongodb = mongodb_client.ragdb

# Redis
redis_client = redis.from_url(settings.redis_url, decode_responses=True)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)