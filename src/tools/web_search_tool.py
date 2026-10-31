from duckduckgo_search import DDGS

#Выполняет поиск в интернете по внешним источникам и сайтам.
#Скрипт будет использоваться тогда, когда информации нет в локальных инструкциях/базе знаний,
#а также, если пользователь прямо просит найти информацию в интернете.
def web_search(query: str) -> str:
    """РЕЗЕРВНЫЙ инструмент поиска в интернете. 
    Использовать СТРОГО ТОЛЬКО после того, как инструмент search_manuals вернул пустой результат 
    ("В инструкции не удалось найти информацию..."). Не использовать для внутренних инструкций.
    
    Args:
        query: Текст поискового запроса для интернета.
    """
    #отладочный вывод (потом убрать)
    print(f"\n >>> АГЕНТ ВЫЗВАЛ web_search с запросом: '{query}' <<<\n")
    
    try:
        results = list(DDGS().text(query, max_results=3))
        if not results:
            return "В интернете не удалось найти информацию по данному запросу."

        formatted_results = []
        for i, res in enumerate(results, 1): #пробегаемся по каждому найденному сайту
            title = res.get("title", "") #извлекаем заголовок статьи
            snippet = res.get("body", "") #сниппет
            url = res.get("href", "") #прямая ссылка на сайт
            formatted_results.append(f"--- Результат из веб-поиска #{i} ---\nЗаголовок: {title}\nОписание: {snippet}\nСсылка: {url}\n")

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Ошибка при выполнении веб-поиска: {e}"

if __name__ == "__main__":
    #Тест
    test_q = "Как сбросить ошибку E04 на кофемашине DeLonghi"
    print(f"Тест веб-поиска: '{test_q}'\n")
    print(web_search(test_q))