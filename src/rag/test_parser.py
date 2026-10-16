from pathlib import Path
from src.rag.parser import PDFParser #подключаю свой парсер

def test_parser():
    #путь к папке с pdf
    pdfs_dir = Path(__file__).resolve().parent.parent.parent / "data" / "pdfs"
    
    #ищем первый попавшийся PDF-файл в папке
    pdf_files = list(pdfs_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"В папке {pdfs_dir} не найдено ни одного PDF-файла!")
        return

    #берем первый файл для теста
    test_file = pdf_files[0]
    print(f"📄 Тестируем файл: {test_file.name}")

    #инициализируем наш парсер
    parser = PDFParser(chunk_size=500, chunk_overlap=50)

    try:
        chunks = parser.process_pdf(test_file)
        
        print(f"Успешно! Всего получилось чанков: {len(chunks)}")
        print("\n--- Пример первого чанка ---")
        print(f"Метаданные: {chunks[0]['metadata']}")
        print(f"Текст:\n{chunks[0]['content']}")
        
        print("\n--- Пример второго чанка ---")
        print(f"Метаданные: {chunks[1]['metadata']}")
        print(f"Текст:\n{chunks[1]['content']}")
        
    except Exception as e:
        print(f"Ошибка при обработке файла: {e}")

if __name__ == "__main__":
    test_parser()