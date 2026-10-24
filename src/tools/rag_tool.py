from pathlib import Path
from src.rag.parser import PDFParser
from src.rag.vectorstore import VectorStoreManager

# Инициализируем менеджер векторного хранилища один раз при старте модуля
_vector_store = None

def get_vector_store() -> VectorStoreManager:

    global _vector_store
    if _vector_store is None:
        print("Инициализация и загрузка VectorStoreManager для агента...")
        _vector_store = VectorStoreManager()
        
        # Автоматически загружаем PDF-инструкции из папки pdfs при первом обращении
        pdf_dir = Path("data/pdfs")
        if pdf_dir.exists():
            pdf_files = list(pdf_dir.glob("*.pdf"))
            if pdf_files:
                parser = PDFParser(chunk_size=500, chunk_overlap=50)
                all_chunks = []
                for pdf_file in pdf_files:
                    print(f"Чтение мануала: {pdf_file.name}")
                    chunks = parser.process_pdf(pdf_file)
                    all_chunks.extend(chunks)
                
                if all_chunks:
                    _vector_store.add_chunks(all_chunks)
                    print(f"База знаний успешно проиндексирована! Всего чанков: {len(all_chunks)}")
            else:
                print("В папке 'pdfs' не найдены PDF-файлы.")
        else:
            print("Папка 'pdfs' не найдена.")
    
    return _vector_store

#Ищет ответы на вопросы по техническим инструкциям и мануалам с помощью гибридного поиска и реранкинга.
#Данный инструмент используется, когда пользователь спрашивает об ошибках, настройке, характеристиках или обслуживании техники
def search_manuals(query: str) -> str:
    """Ищет ответы на вопросы по техническим инструкциям и мануалам с помощью гибридного поиска и реранкинга.
    
    Данный инструмент используется, когда пользователь спрашивает об ошибках, настройке, характеристиках или обслуживании техники.
    
    Args:
        query: Текст поискового запроса или вопрос пользователя по технике.
    """

    store = get_vector_store()
    results = store.hybrid_search(query, top_k=5, rerank_top_k=3)

    if not results:
        print(f"DEBUG: По запросу '{query}' ничего не найдено в базе знаний.")
        return "В инструкции не удалось найти информацию по вашему вопросу."

    #Форматируем найденные чанки в красивый единый текст для ответа агенту
    formatted_output = []
    for i, res in enumerate(results, 1):
        content = res.get("content", "").strip()
        score = res.get("score", 0.0)
        formatted_output.append(f"--- Фрагмент из инструкции #{i} (релевантность: {score:.2f}) ---\n{content}\n")

    return "\n".join(formatted_output)

if __name__ == "__main__":
    #Локальный тест инструмента
    test_query = "ошибка E04"
    print(f"\nТест поиска в мануалах по запросу: '{test_query}'\n")
    print(search_manuals(test_query))