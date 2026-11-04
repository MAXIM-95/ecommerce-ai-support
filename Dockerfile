#Используем официальный легковесный образ Python
FROM python:3.11-slim

WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Увеличиваем количество попыток (retries) и таймаут для pip, чтобы SSL-ошибки не обрывали сборку
RUN pip install --no-cache-dir --upgrade pip --default-timeout=100 --retries 10 && \
    pip install --no-cache-dir --default-timeout=100 --retries 10 -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]