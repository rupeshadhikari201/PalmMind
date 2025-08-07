from typing import List, Dict, Any, Optional, Tuple
from app.services.vector_store import get_vector_store, VectorStore
from app.services.embeddings import EmbeddingService
from app.services.chat_memory import ChatMemoryService
import cohere
from app.core.config import settings
import re
from datetime import datetime

class RAGEngine:
    def __init__(self):
        self.vector_store: VectorStore = get_vector_store()
        self.embedding_service = EmbeddingService()
        self.memory_service = ChatMemoryService()
        
        if settings.cohere_api_key:
            self.llm_client = cohere.Client(settings.cohere_api_key)
        else:
            self.llm_client = None
    
    async def retrieve_relevant_chunks(
        self, 
        query: str, 
        top_k: int = 5,
        similarity_algorithm: str = "cosine"
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Retrieve relevant document chunks for a query"""
        
        # Generate query embedding
        query_embeddings = await self.embedding_service.generate_embeddings([query])
        query_vector = query_embeddings[0]
        
        # Search vector store
        results = await self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k
        )
        
        return results
    
    async def generate_response(
        self, 
        query: str, 
        context_chunks: List[Tuple[str, float, Dict[str, Any]]],
        chat_history: Optional[str] = None
    ) -> str:
        """Generate response using retrieved context"""
        
        # Prepare context
        context_texts = []
        for chunk_id, similarity, metadata in context_chunks:
            # Get chunk text from metadata or vector store
            chunk_text = metadata.get('text', f"Chunk {chunk_id}")
            context_texts.append(f"[Similarity: {similarity:.3f}] {chunk_text}")
        
        context = "\n\n".join(context_texts)
        
        # Build prompt
        prompt = self._build_rag_prompt(query, context, chat_history)
        
        if self.llm_client:
            try:
                response = self.llm_client.chat(
                    model=settings.cohere_model,
                    message=prompt,
                    max_tokens=500,
                    temperature=0.7,
                    connectors=[{"id": "web-search"}] if not context_texts else None
                )
                return response.text.strip()
            except Exception as e:
                return f"Error generating response: {str(e)}"
        else:
            # Fallback to a simple extractive response
            return self._generate_simple_response(query, context_texts)
    
    async def generate_response_with_citations(
        self, 
        query: str, 
        context_chunks: List[Tuple[str, float, Dict[str, Any]]],
        chat_history: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response with citations using Cohere's RAG capabilities"""
        
        if not self.llm_client:
            # Fallback to simple response
            simple_response = self._generate_simple_response(query, [chunk[2].get('text', '') for chunk in context_chunks])
            return {
                "response": simple_response,
                "citations": [],
                "documents": []
            }
        
        try:
            # Prepare documents for Cohere RAG
            documents = []
            for chunk_id, similarity, metadata in context_chunks:
                chunk_text = metadata.get('text', f"Chunk {chunk_id}")
                documents.append({
                    "title": f"Document Chunk {chunk_id}",
                    "snippet": chunk_text[:500],  # Limit snippet length
                    "text": chunk_text,
                    "url": f"chunk://{chunk_id}",
                    "similarity": similarity
                })
            
            # Use Cohere's RAG feature with documents
            response = self.llm_client.chat(
                model=settings.cohere_model,
                message=query,
                documents=documents,
                max_tokens=500,
                temperature=0.7,
                citation_quality="accurate"
            )
            
            return {
                "response": response.text.strip(),
                "citations": response.citations if hasattr(response, 'citations') else [],
                "documents": documents[:len(context_chunks)]
            }
            
        except Exception as e:
            # Fallback to regular generation
            regular_response = await self.generate_response(query, context_chunks, chat_history)
            return {
                "response": regular_response,
                "citations": [],
                "documents": []
            }
    
    def _build_rag_prompt(self, query: str, context: str, chat_history: Optional[str] = None) -> str:
        """Build RAG prompt for LLM"""
        prompt = f"""Based on the following context information, please answer the user's question. If the information is not available in the context, say so clearly. Provide specific and accurate information from the context.

Context:
{context}

"""
        if chat_history:
            prompt += f"Previous conversation:\n{chat_history}\n\n"
        
        prompt += f"User Question: {query}\n\nPlease provide a comprehensive answer based on the context above:"
        
        return prompt
    
    def _generate_simple_response(self, query: str, context_texts: List[str]) -> str:
        """Generate a simple extractive response when no LLM is available"""
        if not context_texts:
            return "I don't have enough information to answer your question."
        
        # Simple keyword matching and extraction
        query_words = set(query.lower().split())
        best_chunk = ""
        max_matches = 0
        
        for chunk in context_texts:
            chunk_words = set(chunk.lower().split())
            matches = len(query_words.intersection(chunk_words))
            if matches > max_matches:
                max_matches = matches
                best_chunk = chunk
        
        if max_matches > 0:
            # Extract relevant sentences
            sentences = re.split(r'[.!?]+', best_chunk)
            relevant_sentences = []
            
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                if len(query_words.intersection(sentence_words)) > 0:
                    relevant_sentences.append(sentence.strip())
            
            if relevant_sentences:
                return " ".join(relevant_sentences[:2])  # Return first 2 relevant sentences
        
        return "Based on the available information: " + context_texts[0][:200] + "..."
    
    async def process_chat_query(
        self, 
        session_id: str, 
        query: str, 
        top_k: int = 5,
        similarity_algorithm: str = "cosine",
        use_citations: bool = True
    ) -> Dict[str, Any]:
        """Process a chat query with context retrieval and response generation"""
        
        # Check for interview booking intent
        booking_intent = self._detect_booking_intent(query)
        if booking_intent:
            return {
                "response": "I can help you schedule an interview. Please provide your name, email, preferred date, and time.",
                "intent": "booking",
                "requires_booking_info": True,
                "retrieved_chunks": [],
                "citations": [],
                "documents": []
            }
        
        # Get chat history for context
        chat_history = await self.memory_service.get_conversation_context(session_id)
        
        # Retrieve relevant chunks
        relevant_chunks = await self.retrieve_relevant_chunks(
            query=query,
            top_k=top_k,
            similarity_algorithm=similarity_algorithm
        )
        
        # Generate response (with or without citations)
        if use_citations and self.llm_client:
            response_data = await self.generate_response_with_citations(
                query=query,
                context_chunks=relevant_chunks,
                chat_history=chat_history
            )
            response = response_data["response"]
            citations = response_data["citations"]
            documents = response_data["documents"]
        else:
            response = await self.generate_response(
                query=query,
                context_chunks=relevant_chunks,
                chat_history=chat_history
            )
            citations = []
            documents = []
        
        # Save to memory
        await self.memory_service.save_message(session_id, {
            "role": "user",
            "content": query,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        await self.memory_service.save_message(session_id, {
            "role": "assistant",
            "content": response,
            "timestamp": datetime.utcnow().isoformat(),
            "retrieved_chunks": len(relevant_chunks)
        })
        
        return {
            "response": response,
            "intent": "general_query",
            "retrieved_chunks": [
                {
                    "chunk_id": chunk_id,
                    "similarity": similarity,
                    "metadata": metadata
                }
                for chunk_id, similarity, metadata in relevant_chunks
            ],
            "citations": citations,
            "documents": documents
        }
    
    def _detect_booking_intent(self, query: str) -> bool:
        """Detect if the query is about booking an interview"""
        booking_keywords = [
            "schedule", "book", "interview", "appointment", "meeting",
            "available", "time", "date", "calendar", "when can"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in booking_keywords)