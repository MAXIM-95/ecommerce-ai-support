from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDFParser:
    #инициализируем парсер. Здесь chunk_size - максимальный размер чанка в символах, 
    #chunk_overlap - пересечение между чанками для сохранения контекста на стыках.
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    #функция извлечения всего текста из PDF-файла
    def extract_text_from_pdf(self, pdf_path: str | Path) -> str:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += f"\n{text}"
        return full_text

    #Читает PDF, извлекает текст и разбивает его на чанки с сохранением метаданных
    def process_pdf(self, pdf_path: str | Path) -> list[dict]:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")

        raw_text = self.extract_text_from_pdf(path)
        chunks = self.text_splitter.split_text(raw_text)
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunks.append({
                "content": chunk,
                "metadata": {
                    "source": path.name,
                    "chunk_id": i
                }
            })
            
        return processed_chunks