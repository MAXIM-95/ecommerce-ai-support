from pathlib import Path
from src.rag.parser import PDFParser
from src.rag.vectorstore import VectorStoreManager

def test_storage():
    #Ищем тестовый PDF в папке data/pdfs
    pdf_dir = Path("data/pdfs")
    pdf_file = next(pdf_dir.glob("*.pdf"), None)

    if not pdf_file:
        print(f"Не найдены PDF файлы в директории {pdf_dir}")
        return

    print(f"Читаем файл: {pdf_file.name}")
    
    #Нарезаем на чанки с помощью парсера
    parser = PDFParser(chunk_size=500, chunk_overlap=50)
    chunks = parser.process_pdf(pdf_file)
    print(f"Нарезано чанков: {len(chunks)}")

    #Инициализируем менеджер базы и сохраняем чанки
    vector_store = VectorStoreManager()
    vector_store.add_chunks(chunks)

    #Проверяем состояние базы через Qdrant API
    collection_info = vector_store.client.get_collection(vector_store.collection_name)
    print(f"\nСтатистика базы Qdrant:")
    print(f"   - Всего векторов в базе: {collection_info.points_count}")
    print(f"   - Статус коллекции: {collection_info.status}")
    print("Сохранение в Qdrant завершено успешно!")

if __name__ == "__main__":
    test_storage()