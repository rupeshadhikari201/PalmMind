from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np
import cohere
from app.core.config import settings

class EmbeddingService:
    def __init__(self):
        self.sentence_transformer = SentenceTransformer(settings.embedding_model)
        if settings.cohere_api_key:
            self.cohere_client = cohere.Client(settings.cohere_api_key)
        else:
            self.cohere_client = None
    
    async def generate_embeddings(self, texts: List[str], model_type: str = "sentence_transformer") -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if model_type == "cohere" and self.cohere_client:
            return await self._generate_cohere_embeddings(texts)
        else:
            return await self._generate_sentence_transformer_embeddings(texts)
    
    async def _generate_sentence_transformer_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using SentenceTransformer"""
        embeddings = self.sentence_transformer.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    async def _generate_cohere_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Cohere API"""
        try:
            # Cohere has a limit on batch size, so we process in chunks
            batch_size = 96  # Cohere's maximum batch size
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                response = self.cohere_client.embed(
                    texts=batch_texts,
                    model="embed-english-v3.0",  # Cohere's latest embedding model
                    input_type="search_document"  # For document indexing
                )
                all_embeddings.extend(response.embeddings)
            
            return all_embeddings
        except Exception as e:
            print(f"Cohere embedding error: {e}")
            # Fallback to sentence transformer
            return await self._generate_sentence_transformer_embeddings(texts)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        if self.cohere_client and settings.cohere_api_key:
            return 1024  # Cohere embed-english-v3.0 dimension
        else:
            return self.sentence_transformer.get_sentence_embedding_dimension()
