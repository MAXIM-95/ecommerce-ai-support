from pathlib import Path
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

#конфиг с настройками
from src.config import settings

class VectorStoreManager:
    def __init__(self, collection_name: str = None, model_name: str = None):
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        model_name = model_name or settings.EMBEDDING_MODEL
        
        #Загружаем локальную модель для эмбеддингов (легкая и быстрая), которая прописана в конфиге
        print(f"Загружаем модель эмбеддингов: {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        self.vector_size = self.encoder.get_sentence_embedding_dimension()

        #Инициализируем локальный Qdrant, берем путь из централизованного конфига
        self.client = QdrantClient(path=settings.QDRANT_PATH)
        
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
            #Если коллекция уже есть, сразу подтягиваем данные для BM25 из базы
            self._load_existing_chunks_for_bm25()


    def _load_existing_chunks_for_bm25(self):
        try:
            # Проверяем, есть ли точки в коллекции
            count_result = self.client.count(collection_name=self.collection_name)
            if count_result.count > 0:
                print(f"Загрузка {count_result.count} чанков из Qdrant для восстановления BM25-индекса...")
                # Вытягиваем все сохраненные точки из Qdrant
                records, _ = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=count_result.count,
                    with_payload=True,
                    with_vectors=False
                )
                
                texts = []
                metadatas = []
                for record in records:
                    payload = record.payload or {}
                    content = payload.get("content", "")
                    metadata = {k: v for k, v in payload.items() if k != "content"}
                    texts.append(content)
                    metadatas.append(metadata)
                
                if texts:
                    self.all_chunks = [{"content": t, "metadata": m} for t, m in zip(texts, metadatas)]
                    tokenized_corpus = [doc.lower().split(" ") for doc in texts]
                    self.bm25 = BM25Okapi(tokenized_corpus)
                    print("BM25-индекс успешно восстановлен из сохраненной базы!")
        except Exception as e:
            print(f"Не удалось автоматически восстановить BM25 из базы: {e}")


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

        #Сохраняем в память для работы BM25
        self.all_chunks = [{"content": t, "metadata": m} for t, m in zip(texts, metadatas)]

        #Строим BM25-индекс по токенам
        print("Индексация текста для BM25...")
        tokenized_corpus = [doc.lower().split(" ") for doc in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

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


    #Гибридный поиск: Dense (Qdrant) + BM25 + Cross-Encoder Reranking
    def hybrid_search(self, query: str, top_k: int = 5, rerank_top_k: int = 3) -> List[Dict[str, Any]]:

        if not hasattr(self, "bm25") or not self.all_chunks:
            print("Индекс BM25 не инициализирован. Сначала добавьте чанки через add_chunks.")
            return []

        # ШИРОКИЙ ПОИСК: увеличиваем коэффициент захвата кандидатов (например, в 5 раз)
        fetch_limit = top_k * 5

        # ШАГ 1: BM25 поиск (по ключевым словам)
        tokenized_query = query.lower().split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:fetch_limit]
        
        # ШАГ 2: Dense поиск (векторный через Qdrant)
        query_vector = self.encoder.encode(query).tolist()
        qdrant_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=fetch_limit
        ).points

        # Собираем кандидатов в единый словарь по тексту
        candidates = {}

        for idx in bm25_top_indices:
            score = bm25_scores[idx]
            if score > 0.0:  # берем любые ненулевые совпадения
                chunk = self.all_chunks[idx]
                candidates[chunk["content"]] = {
                    "content": chunk["content"],
                    "metadata": chunk["metadata"],
                    "source": "bm25",
                    "raw_score": float(score)
                }

        for point in qdrant_results:
            content = point.payload.get("content", "")
            q_score = point.score if hasattr(point, 'score') else 0.0
            metadata = {k: v for k, v in point.payload.items() if k != "content"}
            
            if content in candidates:
                candidates[content]["source"] = "hybrid"
            else:
                candidates[content] = {
                    "content": content,
                    "metadata": metadata,
                    "source": "vector",
                    "raw_score": float(q_score)
                }

        candidate_list = list(candidates.values())
        print(f" >>> Всего уникальных кандидатов собрано для реранкинга: {len(candidate_list)} <<<")

        if not candidate_list:
            return []

        # ЖЕСТКОЕ ОГРАНИЧЕНИЕ: берем для реранкера не больше 10-12 штук, чтобы не вешать CPU
        candidate_list = candidate_list[:12]

        # ШАГ 3: Cross-Encoder Reranking
        try:
            from sentence_transformers import CrossEncoder
            if not hasattr(self, "_reranker"):
                self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            
            pairs = [[query, item["content"]] for item in candidate_list]
            ranks = self._reranker.predict(pairs)

            for i, item in enumerate(candidate_list):
                item["score"] = float(ranks[i])

            candidate_list = sorted(candidate_list, key=lambda x: x["score"], reverse=True)
            
            # Отладочный вывод топ-3 после реранкинга
            print(" >>> Топ-3 чанка после реранкинга:")
            for i, item in enumerate(candidate_list[:3]):
                print(f"   [{i+1}] score={item['score']:.2f} | текст: {item['content'][:120]}...")

        except Exception as e:
            print(f"Ошибка реранкинга (возвращаем без него): {e}")

        return candidate_list[:rerank_top_k]