from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from datetime import datetime
import uuid

from app.core.database import get_db, mongodb
from app.models.document import Document
from app.schemas.document import (
    DocumentUploadResponse, 
    DocumentListResponse,
    DocumentChunksResponse,
    ChunkInfo
)
from app.services.document_processor import DocumentProcessor
from app.services.chunking import get_chunking_strategy
from app.services.embeddings import EmbeddingService
from app.services.vector_store import get_vector_store
import uuid

router = APIRouter()

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunking_strategy: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """Upload and process a document"""
    
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension not in ['pdf', 'txt']:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
    
    # Validate chunking strategy
    if chunking_strategy not in ['fixed_size', 'semantic']:
        raise HTTPException(status_code=400, detail="Invalid chunking strategy")
    
    try:
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        # Process document
        processor = DocumentProcessor()
        text_content, content_hash = await processor.process_document(
            file_content, file.filename, file_extension
        )
        
        # Check if document already exists
        existing_doc = await db.execute(
            select(Document).where(Document.content_hash == content_hash)
        )
        existing_doc = existing_doc.scalar_one_or_none()
        
        if existing_doc:
            raise HTTPException(status_code=409, detail="Document already exists")
        
        # Chunk the document
        chunking_service = get_chunking_strategy(chunking_strategy)
        chunks = chunking_service.chunk_text(text_content, {
            'document_id': str(uuid.uuid4()),
            'filename': file.filename,
            'file_type': file_extension
        })
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks generated from document")
        
        # Generate embeddings
        embedding_service = EmbeddingService()
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = await embedding_service.generate_embeddings(chunk_texts)
        
        # Create document record
        document = Document(
            filename=file.filename,
            file_type=file_extension,
            file_size=file_size,
            content_hash=content_hash,
            chunking_strategy=chunking_strategy,
            total_chunks=len(chunks),
            metadata={'original_text_length': len(text_content)}
        )
        
        db.add(document)
        await db.flush()
        
        # Store in vector database
        vector_store = get_vector_store()
        chunk_metadata = []
        chunk_ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            chunk_ids.append(chunk_id)
            
            chunk_meta = {
                **chunk['metadata'],
                'document_id': str(document.id),
                'text': chunk['text']  # Store text in metadata for retrieval
            }
            chunk_metadata.append(chunk_meta)
        
        await vector_store.add_vectors(embeddings, chunk_metadata, chunk_ids)
        
        # Store chunks in MongoDB for detailed retrieval
        chunks_collection = mongodb.document_chunks
        chunk_documents = []
        for i, (chunk, chunk_id) in enumerate(zip(chunks, chunk_ids)):
            chunk_doc = {
                'chunk_id': chunk_id,
                'document_id': str(document.id),
                'text': chunk['text'],
                'metadata': chunk['metadata'],
                'chunk_index': i,
                'created_at': datetime.utcnow()
            }
            chunk_documents.append(chunk_doc)
        
        if chunk_documents:
            await chunks_collection.insert_many(chunk_documents)
        
        await db.commit()
        
        return DocumentUploadResponse(
            document_id=str(document.id),
            filename=document.filename,
            file_type=document.file_type,
            file_size=document.file_size,
            chunking_strategy=document.chunking_strategy,
            total_chunks=document.total_chunks,
            content_hash=document.content_hash,
            created_at=document.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    skip: int = 0,
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """List all documents"""
    try:
        # Get total count
        count_result = await db.execute(select(Document))
        total = len(count_result.scalars().all())
        
        # Get documents with pagination
        result = await db.execute(
            select(Document).offset(skip).limit(limit).order_by(Document.created_at.desc())
        )
        documents = result.scalars().all()
        
        document_responses = [
            DocumentUploadResponse(
                document_id=str(doc.id),
                filename=doc.filename,
                file_type=doc.file_type,
                file_size=doc.file_size,
                chunking_strategy=doc.chunking_strategy,
                total_chunks=doc.total_chunks,
                content_hash=doc.content_hash,
                created_at=doc.created_at
            )
            for doc in documents
        ]
        
        return DocumentListResponse(documents=document_responses, total=total)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@router.get("/{document_id}/chunks", response_model=DocumentChunksResponse)
async def get_document_chunks(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get chunks for a specific document"""
    try:
        # Verify document exists
        result = await db.execute(select(Document).where(Document.id == document_id))
        document = result.scalar_one_or_none()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get chunks from MongoDB
        chunks_collection = mongodb.document_chunks
        cursor = chunks_collection.find({'document_id': document_id}).sort('chunk_index', 1)
        chunks = await cursor.to_list(length=None)
        
        chunk_infos = [
            ChunkInfo(
                chunk_id=chunk['chunk_id'],
                text=chunk['text'],
                metadata=chunk['metadata']
            )
            for chunk in chunks
        ]
        
        return DocumentChunksResponse(
            document_id=document_id,
            chunks=chunk_infos,
            total_chunks=len(chunk_infos)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chunks: {str(e)}")

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a document and its chunks"""
    try:
        # Get document
        result = await db.execute(select(Document).where(Document.id == document_id))
        document = result.scalar_one_or_none()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get chunk IDs for vector store deletion
        chunks_collection = mongodb.document_chunks
        cursor = chunks_collection.find({'document_id': document_id}, {'chunk_id': 1})
        chunks = await cursor.to_list(length=None)
        chunk_ids = [chunk['chunk_id'] for chunk in chunks]
        
        # Delete from vector store
        vector_store = get_vector_store()
        await vector_store.delete_vectors(chunk_ids)
        
        # Delete from MongoDB
        await chunks_collection.delete_many({'document_id': document_id})
        
        # Delete from PostgreSQL
        await db.delete(document)
        await db.commit()
        
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")
