from pathlib import Path
import pymupdf as fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any


class PDFParser:
    #инициализируем парсер. Здесь chunk_size - максимальный размер чанка в символах, 
    #chunk_overlap - пересечение между чанками для сохранения контекста на стыках.
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def clean_pdf_text(self, text: str) -> str:
        #Универсальная очистка текста от артефактов PDF-парсера:
        #- склеивает слова, разорванные дефисами на стыках строк (например, "монта-жу" -> "монтажу")
        #- убирает лишние пробелы и дублирующиеся переносы строк
        
        #1) убираем переносы через дефис
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        #2) убираем множественные пробелы и табуляцию
        text = re.sub(r'[ \t]+', ' ', text)
        
        #3) нормализуем пустые строки
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()

    #Читает PDF, извлекает текст и разбивает его на чанки с сохранением метаданных
    def process_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        # Открываем PDF через PyMuPDF
        doc = fitz.open(file_path) 
        full_text = ""

        #Постраничное извлечение текста
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")  #извлекает текст в обычном текстовом виде
            if text:
                full_text += f"\n{text}"
        
        #Проверка на случай, если внутри документа пусто (например, сканы без текста)
        if not full_text.strip():
            print(f"Внимание: в файле {file_path.name} не удалось извлечь текст (возможно, нужна OCR).")
            return []

        #КРИТИЧЕСКИ ВАЖНО: очищаем весь сырой текст от артефактов перед нарезкой
        cleaned_full_text = self.clean_pdf_text(full_text)

        #Нарезка текста с помощью LangChain
        chunks = self.text_splitter.create_documents( #в chunks будет список объектов класса Documents из LangChain
            texts=[cleaned_full_text],
            metadatas=[{"source": file_path.name}]
        )

        #Преобразование объектов LangChain в удобный формат словарей с ID
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunks.append({
                "content": chunk,
                "metadata": {
                    "source": file_path.name,
                    "chunk_id": i
                }
            })
            
        return processed_chunks

    