from pathlib import Path
from src.rag.parser import PDFParser
from src.rag.vectorstore import VectorStoreManager

def test_hybrid_search():
    #Загружаем чанки из папки pdfs и инициализируем менеджер
    pdf_dir = Path(__file__).resolve().parent.parent.parent / "data" / "pdfs"
    pdf_file = next(pdf_dir.glob("*.pdf"), None)

    if not pdf_file:
        print(f"Не найдены PDF файлы в директории {pdf_dir}")
        return

    print(f"Читаем файл для теста поиска: {pdf_file.name}")
    parser = PDFParser(chunk_size=500, chunk_overlap=50)
    chunks = parser.process_pdf(pdf_file)

    vector_store = VectorStoreManager()
    
    #Важно: вызываем add_chunks, чтобы проиндексировать BM25 в оперативной памяти
    print("Индексация чанков для BM25 и загрузка базы...")
    vector_store.add_chunks(chunks)

    #Задаем тестовый запрос
    query = "гарантия"  #как пример
    print(f"\nИщем по запросу: '{query}'\n")

    results = vector_store.hybrid_search(query, top_k=5, rerank_top_k=3)

    #Выводим результаты
    print(f"Найдено результатов после реранкинга: {len(results)}\n")
    for i, res in enumerate(results, 1):
        print(f"--- Результат #{i} ---")
        print(f"Источник: {res.get('source', 'unknown')}")
        print(f"Оценка реранкера (Score): {res.get('score', 0.0):.4f}")
        print(f"Текст:\n{res['content'][:300]}...\n")

if __name__ == "__main__":
    test_hybrid_search()