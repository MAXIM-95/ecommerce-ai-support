import traceback
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
       # Используем обновленный интерфейс DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            
        if not results:
            return "В интернете не удалось найти информацию по данному запросу."
            
        # Форматируем полученные результаты в читаемый текст для LLM
        formatted_results = []
        for r in results: #пробегаемся по каждому найденному сайту
            title = r.get("title", "") #извлекаем заголовок статьи
            snippet = r.get("body", "") #сниппет
            link = r.get("href", "") #прямая ссылка на сайт
            formatted_results.append(f"Заголовок: {title}\nИнформация: {snippet}\nСсылка: {link}")
            
        return "\n\n".join(formatted_results)

    except Exception as e:
        print(f"Ошибка при выполнении Web Search: {e}")
        print(traceback.format_exc())
        return "Не удалось получить свежие данные из интернета из-за сетевого сбоя или блокировки поисковика!"

if __name__ == "__main__":
    #Тест
    test_q = "Как сбросить ошибку E04 на кофемашине DeLonghi"
    print(f"Тест веб-поиска: '{test_q}'\n")
    print(web_search(test_q))