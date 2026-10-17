from pathlib import Path
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

class VectorStoreManager:
    def __init__(self, collection_name: str = "support_docs", model_name: str = "all-MiniLM-L6-v2"):
        self.collection_name = collection_name
        
        #Загружаем локальную модель для эмбеддингов (легкая и быстрая)
        print(f"Загружаем модель эмбеддингов: {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        self.vector_size = self.encoder.get_sentence_embedding_dimension()

        #Инициализируем локальный Qdrant (данные будут сохраняться в папку qdrant_storage)
        self.client = QdrantClient(path="./qdrant_storage")
        
        #Создаем коллекцию, если она еще не существует
        self._init_collection()

    def _init_collection(self):
        collections = self.client.get_collections().collections
        exists = any(col.name == self.collection_name for col in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Создана новая коллекция Qdrant: {self.collection_name}")
        else:
            print(f"Коллекция {self.collection_name} уже существует.")


    def add_chunks(self, chunks: List[Any]):
        if not chunks:
            print("Нет чанков для добавления в базу.")
            return

        texts = []
        metadatas = []

        for chunk in chunks:
            #Извлекаем текст и метаданные универсально, чтобы не было проблем со словарями и LangChain документами
            if isinstance(chunk, dict):
                content = chunk.get("content", "")
                metadata = chunk.get("metadata", {})
            else:
                #Если чанк является объектом LangChain Document
                content = getattr(chunk, "page_content", str(chunk))
                metadata = getattr(chunk, "metadata", {})

            #Если content сам оказался объектом Document
            if hasattr(content, "page_content"):
                content = content.page_content

            texts.append(str(content))
            metadatas.append(metadata if isinstance(metadata, dict) else {})

        #Сохраняем в память для будущего BM25
        self.all_chunks = [{"content": t, "metadata": m} for t, m in zip(texts, metadatas)]

        #Генерируем векторные представления для всех чанков
        print(f"Генерация эмбеддингов для {len(texts)} чанков...")
        embeddings = self.encoder.encode(texts, show_progress_bar=True)

        # Формируем точки для загрузки в Qdrant
        points = []
        for i, (text, metadata, vector) in enumerate(zip(texts, metadatas, embeddings)):
            points.append(
                models.PointStruct(
                    id=i,  #порядковый индекс
                    vector=vector.tolist(),  # эмбеддинг
                    payload={  #текст чанка с метаданными
                        "content": text,
                        **metadata
                    }
                )
            )

        #Загружаем пачкой в Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Успешно сохранено чанков в Qdrant: {len(points)}")