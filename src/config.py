import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "E-commerce AI Support"
    
    #Database
    DATABASE_URL: str = "sqlite:///./data/orders.db"
    
    #Vector DB (Qdrant)
    QDRANT_PATH: str = "./qdrant_storage"
    QDRANT_COLLECTION: str = "manuals"
    
    #Models & Ollama
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    LLM_MODEL: str = "qwen2.5"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

