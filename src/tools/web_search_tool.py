from duckduckgo_search import DDGS

#Выполняет поиск в интернете по внешним источникам и сайтам.
#Скрипт будет использоваться тогда, когда информации нет в локальных инструкциях/базе знаний,
#а также, если пользователь прямо просит найти информацию в интернете.
def web_search(query: str) -> str:
   
    try:
        results = list(DDGS().text(query, max_results=3))
        if not results:
            return "В интернете не удалось найти информацию по данному запросу."

        formatted_results = []
        for i, res in enumerate(results, 1):
            title = res.get("title", "")
            snippet = res.get("body", "")
            url = res.get("href", "")
            formatted_results.append(f"--- Результат из веб-поиска #{i} ---\nЗаголовок: {title}\nОписание: {snippet}\nСсылка: {url}\n")

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Ошибка при выполнении веб-поиска: {e}"

if __name__ == "__main__":
    #Тест
    test_q = "Как сбросить ошибку E04 на кофемашине DeLonghi"
    print(f"Тест веб-поиска: '{test_q}'\n")
    print(web_search(test_q))