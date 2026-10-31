from pathlib import Path
import re #регулярные выражения
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
    #
    def process_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        # Открываем PDF через PyMuPDF
        doc = fitz.open(file_path) 
        
        processed_chunks = []
        current_section = "Общая информация" # дефолтный заголовок, если еще не встретили другой

        #Постраничное извлечение текста
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Извлекаем текст в виде блоков с деталями шрифтов (dict)
            blocks = page.get_text("dict")["blocks"]
            
            page_text_blocks = []

            #перебираем каждый визуальный блок на странице. 
            for b in blocks:
                if "lines" in b:  #текстовый блок -> собираем всю информацию внутри него
                    block_text = ""
                    max_font_size = 0.0
                    is_bold = False

                    #проходим по каждой строке и каждому кусочку текста и склеиваем их в единую строку блока
                    for line in b["lines"]: 
                        for span in line["spans"]:
                            block_text += span["text"] + " "
                            if span["size"] > max_font_size:
                                max_font_size = span["size"]
                            # Флаг жирного шрифта в PyMuPDF (бит 1). Для того, чтобы понять где заголовок, а где обычный текст
                            if span["flags"] & 2: 
                                is_bold = True

                    block_text = block_text.strip()
                    if not block_text:
                        continue

                    # Эвристика определения заголовка: 
                    # 1) либо крупный/жирный шрифт
                    # 2) либо текст начинается с номера главы/пункта (например, "16.8" или "Глава 3")
                    is_heading = (max_font_size > 11.5 or is_bold) and len(block_text) < 100
                    match_section_num = re.match(r'^(\d+(\.\d+)*)\s+[А-ЯA-Z]', block_text)

                    if is_heading or match_section_num:
                        current_section = block_text
                        continue # заголовок вынесли в контекст, сам по себе он чанком не будет

                    # Очищаем текст блока от артефактов
                    cleaned_block = self.clean_pdf_text(block_text)

                    if cleaned_block:
                        #добавляем контекст заголовка к каждому блоку текста
                        enriched_block = f"[Раздел: {current_section}]\n{cleaned_block}"
                        page_text_blocks.append(enriched_block)

            #Собираем все обогащенные блоки страницы в единый текст для сплиттера
            page_full_text = "\n\n".join(page_text_blocks)
            
            if not page_full_text.strip():
                continue

            #Нарезка через LangChain с учетом контекста разделов
            chunks = self.text_splitter.create_documents(
                texts=[page_full_text],
                metadatas=[{"source": file_path.name, "page": page_num + 1}]
            )

            #Преобразование объектов LangChain в удобный формат словарей с ID
            for i, chunk in enumerate(chunks):
                processed_chunks.append({
                    "content": chunk.page_content, #достаем конкретно саму строку текста из объекта, а не сам ланг чейн объект
                    "metadata": {
                        "source": file_path.name, #имя файла
                        "page": page_num + 1, #номер страницы
                        "chunk_id": len(processed_chunks)
                    }
                })

        if not processed_chunks:
            print(f"Внимание: в файле {file_path.name} не удалось извлечь структурированный текст.")
            
        return processed_chunks
    