from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import uuid

# Vector Store Implementations
class VectorStore(ABC):
    @abstractmethod
    async def add_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> List[str]:
        pass
    
    @abstractmethod
    async def search(self, query_vector: List[float], top_k: int = 5, filter_dict: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        pass
    
    @abstractmethod
    async def delete_vectors(self, ids: List[str]) -> bool:
        pass


from typing import List, Dict, Any, Optional, Tuple
import uuid
from qdrant_client import models
from qdrant_client.async_qdrant_client import AsyncQdrantClient
  # async client
from app.core.config import settings


class QdrantVectorStore(VectorStore):
    def __init__(self, collection_name: str = "documents"):
        # Use AsyncQdrantClient when your methods are async
        self.client = AsyncQdrantClient(
            url=settings.qdrant_url,            # e.g. "https://xyz-example.eu-central.aws.cloud.qdrant.io:6333"
            api_key=settings.qdrant_api_key,    # your Qdrant Cloud DB API key
            prefer_grpc=False,                  # set True if you want gRPC and have network permissions
        )
        self.collection_name = collection_name
        self.VectorParams = models.VectorParams
        self.Distance = models.Distance
        self.PointStruct = models.PointStruct

    async def _ensure_collection(self, vector_size: int):
        """Ensure collection exists"""
        try:
            collections = await self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            if self.collection_name not in collection_names:
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=self.VectorParams(
                        size=vector_size,
                        distance=self.Distance.COSINE,
                    ),
                )
        except Exception as e:
            # better to log than print in production
            print(f"Error ensuring collection: {e}")

    async def add_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> List[str]:
        if not vectors:
            return []

        await self._ensure_collection(len(vectors[0]))

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]

        points = [
            self.PointStruct(id=pid, vector=vec, payload=meta)
            for pid, vec, meta in zip(ids, vectors, metadata)
        ]

        await self.client.upsert(collection_name=self.collection_name, points=points)
        return ids

    async def search(self, query_vector: List[float], top_k: int = 5, filter_dict: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        try:
            results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=filter_dict,
            )
            return [(str(hit.id), hit.score, hit.payload or {}) for hit in results]
        except Exception as e:
            print(f"Search error: {e}")
            return []

    async def delete_vectors(self, ids: List[str]) -> bool:
        try:
            # note: delete API expects a selector; passing list of ids works via points_selector
            await self.client.delete(collection_name=self.collection_name, points_selector=ids)
            return True
        except Exception as e:
            print(f"Delete error: {e}")
            return False


# Simple in-memory vector store for development
class InMemoryVectorStore(VectorStore):
    def __init__(self):
        self.vectors: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
    
    async def add_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> List[str]:
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]
        
        for vector_id, vector, meta in zip(ids, vectors, metadata):
            self.vectors[vector_id] = vector
            self.metadata[vector_id] = meta
        
        return ids
    
    async def search(self, query_vector: List[float], top_k: int = 5, filter_dict: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        if not self.vectors:
            return []
        
        # Calculate similarities
        similarities = []
        query_vector = np.array(query_vector).reshape(1, -1)
        
        for vector_id, vector in self.vectors.items():
            if filter_dict:
                # Simple filter check
                meta = self.metadata.get(vector_id, {})
                if not all(meta.get(k) == v for k, v in filter_dict.items()):
                    continue
            
            stored_vector = np.array(vector).reshape(1, -1)
            similarity = cosine_similarity(query_vector, stored_vector)[0][0]
            similarities.append((vector_id, similarity, self.metadata.get(vector_id, {})))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    async def delete_vectors(self, ids: List[str]) -> bool:
        try:
            for vector_id in ids:
                self.vectors.pop(vector_id, None)
                self.metadata.pop(vector_id, None)
            return True
        except:
            return False

def get_vector_store() -> VectorStore:
    """Factory function to get vector store based on configuration"""
    if settings.vector_store_type == "qdrant":
        return QdrantVectorStore()
    else:
        return InMemoryVectorStore()