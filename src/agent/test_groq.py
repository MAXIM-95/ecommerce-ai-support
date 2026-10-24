import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Инициализируем чистый клиент Groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

try:
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": "Привет! Работаешь?"}
        ]
    )
    print("Успех! Ответ от Groq:")
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"Ошибка при прямом вызове Groq: {e}")