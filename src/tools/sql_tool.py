import sqlite3
from pathlib import Path


#Ищет информацию о заказе в базе данных SQLite по номеру заказа (order_id) или email клиента.
#Возвращает строку с подробной информацией о статусе, товаре и дате доставки.
def search_order(order_id: str = None, email: str = None) -> str:

    db_path = Path("data/orders.db")
    if not db_path.exists():
        return "Ошибка: База данных заказов не найдена."

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        #Ищем по номеру заказа или по email
        if order_id:
            cursor.execute("""
                SELECT order_id, email, status, item_name, shipping_date, delivery_address 
                FROM orders WHERE order_id = ?
            """, (order_id,))
        elif email:
            cursor.execute("""
                SELECT order_id, email, status, item_name, shipping_date, delivery_address 
                FROM orders WHERE email = ?
            """, (email,))
        else:
            return "Укажите номер заказа или email для поиска."

        rows = cursor.fetchall() #забираем все найденные строки результата sql запроса (rows - список кортежей)
        if not rows:
            return "Заказ с такими данными не найден в базе."

        result_lines = []
        for row in rows:
            oid, mail, status, item, ship_date, addr = row
            result_lines.append(
                f"Заказ №: {oid}\n"
                f"Email: {mail}\n"
                f"Товар: {item}\n"
                f"Статус: {status}\n"
                f"Дата доставки: {ship_date}\n"
                f"Адрес: {addr}\n"
            )
        return "\n".join(result_lines)

    except Exception as e:
        return f"Ошибка при запросе к базе данных: {e}"
    finally:
        conn.close()

if __name__ == "__main__":
    #Быстрый локальный тест инструмента
    print("Тест поиска по номеру заказа ORD-1001:")
    print(search_order(order_id="ORD-1001"))