import sqlite3
from pathlib import Path


def init_order_db(db_path: str = "data/orders.db"):
    #Убедимся, что директория существует
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Создаем таблицу заказов
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            status TEXT NOT NULL,
            item_name TEXT NOT NULL,
            shipping_date TEXT,
            delivery_address TEXT
        )
    """)

    # Тестовые данные (фейковые заказы)
    sample_orders = [
        ("ORD-1001", "ivan@test.ru", "В пути", "Робот-пылесос Xiaomi", "2026-10-20", "г. Москва, ул. Ленина, д. 5"),
        ("ORD-1002", "elena@mail.com", "Доставлен", "Посудомоечная машина Bosh", "2026-10-15", "г. Санкт-Петербург, пр. Невский, д. 10"),
        ("ORD-1003", "alex@yandex.ru", "Обрабатывается на складе", "Ноутбук Acer Nitro 16", "2026-10-25", "г. Казань, ул. Баумана, д. 12"),
    ]

    # Заполняем таблицу (используем INSERT OR REPLACE, чтобы скрипт можно было перезапускать)
    cursor.executemany("""
        INSERT OR REPLACE INTO orders (order_id, email, status, item_name, shipping_date, delivery_address)
        VALUES (?, ?, ?, ?, ?, ?)
    """, sample_orders)

    conn.commit() #фиксация транзакций
    conn.close() #закрываем активное соединение
    print(f"База данных заказов успешно создана и заполнена в файле: {db_file.resolve()}")

if __name__ == "__main__":
    init_order_db()