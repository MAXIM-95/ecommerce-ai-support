import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "E-commerce AI Support"
    OPENAI_API_KEY: str
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/support.db"
    
    # Vector DB
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333 #стандартный порт Qdrant
    QDRANT_COLLECTION: str = "manuals" #имя «папки» (коллекции) в базе данных, где будут лежать вектора инструкций.
    
    # Models
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    LLM_MODEL: str = "gpt-4o-mini" #хранится 

    class Config:
        env_file = ".env"

settings = Settings()

