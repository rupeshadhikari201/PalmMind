from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str
    timestamp: Optional[datetime] = None

class ChatQueryRequest(BaseModel):
    session_id: str = Field(..., description="Chat session ID")
    query: str = Field(..., description="User query")
    top_k: int = Field(default=5, description="Number of chunks to retrieve")
    similarity_algorithm: str = Field(default="cosine", description="Similarity algorithm")

class RetrievedChunk(BaseModel):
    chunk_id: str
    similarity: float
    metadata: Dict[str, Any]

class ChatQueryResponse(BaseModel):
    response: str
    intent: str
    retrieved_chunks: List[RetrievedChunk]
    requires_booking_info: bool = False

class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    total_messages: int
