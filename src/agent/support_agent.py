import os
from dotenv import load_dotenv
#from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent


#Импортируем три ранее созданных инструмента для агента
from src.tools.sql_tool import search_order #поиск в базе заказов
from src.tools.rag_tool import search_manuals #поиск в pdf инструкциях
from src.tools.web_search_tool import web_search #поиск в интернете

#Загружаем переменные окружения из файла .env
#Делаем ключ доступным для питон
load_dotenv()

#Проверяем, задан ли ключ для Groq (если ключа нет, то сразу выдаем ошибку)
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Не найден GOOGLE_API_KEY в переменных окружения (.env)!")

#Инициализируем языковую модель Llama 3.3 через Groq
#параметр temperature=0 делает поведение агента более точным и логичным при выборе инструментов (запрещаем агенту излишне фантазировать)
#llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# Инициализируем модель Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash", 
    temperature=0,
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)

#Объединяем все инструменты в единый список для агента
tools = [search_order, search_manuals, web_search]

#Системный промпт (инструкция для агента, задающая его роль и правила поведения)
SYSTEM_PROMPT = """
Ты — профессиональный и дружелюбный AI-ассистент технической поддержки интернет-магазина.
Твоя задача — помогать клиентам с их заказами, техническими вопросами по устройствам и инструкциями.

Правила работы:
1. Если клиент спрашивает статус своего заказа, доставку, адрес или состав заказа — используй инструмент `search_order`. Для этого тебе понадобится номер заказа (например, ORD-1001) или email клиента.
2. Если клиент спрашивает про ошибки, настройку, характеристики или обслуживание техники — сначала используй инструмент локальных инструкций `search_manuals`.
3. Если в локальных инструкциях информации нет или она неполная — используй `web_search` для поиска в интернете.
4. Всегда отвечай вежливо, четко и структурировано на русском языке. Опирайся только на данные, полученные от инструментов.
"""

# Создаем готового ReAct-агента с помощью встроенной фабрики LangGraph
agent_executor = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT
)

if __name__ == "__main__":
    #Быстрый локальный тест агента
    print("Агент техподдержки успешно инициализирован!\n")
    
    #Тестовый запрос на проверку заказа (который мы добавляли в базу SQLite)
    test_query = "Здравствуйте! Подскажите, где мой заказ? Номер ORD-1001?"
    print(f"Пользователь: {test_query}\n")
    
    #Запускаем агента
    messages = [{"role": "user", "content": test_query}]
    print("Отправляем запрос агенту...")
    response = agent_executor.invoke({"messages": messages})
    print("Ответ от агента получен!")
    
    print("\nИстория и ответы агента:")
    for m in response["messages"]:
        # Выводим только сообщения ИИ, у которых есть текстовое содержимое
        if m.type == "ai" and m.content:
            print(f"\n{m.content}")