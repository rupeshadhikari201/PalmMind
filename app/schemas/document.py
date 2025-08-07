from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid

class DocumentUploadRequest(BaseModel):
    chunking_strategy: str = Field(..., description="Chunking strategy: 'fixed_size' or 'semantic'")

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    file_size: int
    chunking_strategy: str
    total_chunks: int
    content_hash: str
    created_at: datetime

class DocumentListResponse(BaseModel):
    documents: List[DocumentUploadResponse]
    total: int

class ChunkInfo(BaseModel):
    chunk_id: str
    text: str
    # metadata: Dict[str, Any]

class DocumentChunksResponse(BaseModel):
    document_id: str
    chunks: List[ChunkInfo]
    total_chunks: int