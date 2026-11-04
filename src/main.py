from fastapi import FastAPI
from src.config import settings
from src.api.chat import router as chat_router

# Экземпляр FastAPI приложения
app = FastAPI(
    title="E-commerce AI Support API",
    description="API для умной техподдержки и консультанта на базе LangGraph и Ollama",
    version="1.0.0"
)

# Подключаем роутеры из папки api
app.include_router(chat_router)

@app.get("/")
def read_root():
    return {"status": "active", "service": settings.PROJECT_NAME}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)
