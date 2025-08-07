from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Database
    postgres_url: str 
    redis_url: str
    mongodb_url: str
    
    # Vector Databases
    qdrant_url: str 
    qdrant_api_key: str 
   
    
    # AI/ML
    cohere_api_key: Optional[str] 
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Email
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    
    # Vector Store Selection
    vector_store_type: str = "qdrant"  
    
    class Config:
        env_file = ".env"

settings = Settings()