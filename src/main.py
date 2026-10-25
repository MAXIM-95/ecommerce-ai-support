from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# Импортирую своего готового агента
from src.agent.support_agent import agent_executor  

#экземпляр FastAPI приложения
app = FastAPI(
    title="E-commerce AI Support API",
    description="API для умной техподдержки и консультанта на базе LangGraph и Gemini",
    version="1.0.0"
)

class ChatRequest(BaseModel): #модель входящего запроса
    message: str
    history: List[Dict[str, Any]] = []

class ChatResponse(BaseModel): #модель ответа сервера
    response: str
    history: List[Dict[str, Any]]

@app.get("/") #корневой get-endpoint
def read_root():
    return {"status": "active", "service": "E-commerce AI Support Agent"}

@app.post("/chat", response_model=ChatResponse) #основной POST-endpoint
def chat_endpoint(request: ChatRequest):
    try:
        #Формируем сообщения для графа агента
        #Превращаем историю в формат, который ожидает LangGraph
        messages = []
        for h in request.history:
            if h.get("role") == "user":
                messages.append(("user", h["content"]))
            elif h.get("role") == "assistant":
                messages.append(("assistant", h["content"]))
        
        # Добавляем текущее сообщение пользователя
        messages.append(("user", request.message))
        
        # Запускаем агента
        result = agent_executor.invoke({"messages": messages})
        
        # Достаем последнее сообщение от ассистента
        last_message = result["messages"][-1]
        agent_reply = getattr(last_message, "content", str(last_message))
        
        # Обновляем историю для клиента
        updated_history = request.history + [
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": agent_reply}
        ]
        
        return ChatResponse(
            response=agent_reply,
            history=updated_history
        )

    #обработка ошибки    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)