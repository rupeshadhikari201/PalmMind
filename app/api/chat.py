from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, date, time
from typing import Optional

from app.core.database import get_db
from app.models.chat import ChatSession
from app.models.booking import InterviewBooking
from app.schemas.chat import (
    ChatQueryRequest, 
    ChatQueryResponse, 
    ChatHistoryResponse,
    ChatMessage
)
from app.schemas.booking import InterviewBookingRequest, InterviewBookingResponse
from app.services.rag_engine import RAGEngine
from app.services.chat_memory import ChatMemoryService
from app.services.email_service import EmailService

router = APIRouter()

@router.post("/query", response_model=ChatQueryResponse)
async def process_chat_query(
    request: ChatQueryRequest,
    db: AsyncSession = Depends(get_db)
):
    """Process a chat query with RAG"""
    try:
        # Ensure chat session exists
        result = await db.execute(
            select(ChatSession).where(ChatSession.session_id == request.session_id)
        )
        session = result.scalar_one_or_none()
        
        if not session:
            # Create new session
            session = ChatSession(session_id=request.session_id)
            db.add(session)
            await db.commit()
        
        # Process query with RAG engine
        rag_engine = RAGEngine()
        result = await rag_engine.process_chat_query(
            session_id=request.session_id,
            query=request.query,
            top_k=request.top_k,
            similarity_algorithm=request.similarity_algorithm
        )
        
        return ChatQueryResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_id: str,
    limit: int = 10
):
    """Get chat history for a session"""
    try:
        memory_service = ChatMemoryService()
        messages_data = await memory_service.get_chat_history(session_id, limit)
        
        messages = [
            ChatMessage(
                role=msg['role'],
                content=msg['content'],
                timestamp=datetime.fromisoformat(msg.get('timestamp', datetime.utcnow().isoformat()))
            )
            for msg in messages_data
        ]
        
        return ChatHistoryResponse(
            session_id=session_id,
            messages=messages,
            total_messages=len(messages)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {str(e)}")

@router.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear chat history for a session"""
    try:
        memory_service = ChatMemoryService()
        success = await memory_service.clear_chat_history(session_id)
        
        if success:
            return {"message": "Chat history cleared successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found or already empty")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing chat history: {str(e)}")

@router.post("/book-interview", response_model=InterviewBookingResponse)
async def book_interview(
    request: InterviewBookingRequest,
    db: AsyncSession = Depends(get_db)
):
    """Book an interview appointment"""
    try:
        # Parse date and time
        interview_date = datetime.strptime(request.interview_date, "%Y-%m-%d").date()
        interview_time_obj = datetime.strptime(request.interview_time, "%H:%M").time()
        interview_datetime = datetime.combine(interview_date, interview_time_obj)
        
        # Check if the date/time is in the future
        if interview_datetime <= datetime.now():
            raise HTTPException(status_code=400, detail="Interview must be scheduled for a future date and time")
        
        # Create booking
        booking = InterviewBooking(
            name=request.name,
            email=request.email,
            interview_date=interview_datetime,
            interview_time=request.interview_time,
            notes=request.notes
        )
        
        db.add(booking)
        await db.flush()
        
        # Send confirmation email
        email_service = EmailService()
        booking_data = {
            "name": request.name,
            "email": request.email,
            "interview_date": interview_date.strftime("%Y-%m-%d"),
            "interview_time": request.interview_time,
            "status": "Confirmed"
        }
        
        confirmation_sent = await email_service.send_interview_confirmation(booking_data)
        booking.confirmation_sent = confirmation_sent
        
        await db.commit()
        
        # Update chat memory with booking confirmation
        memory_service = ChatMemoryService()
        await memory_service.save_message(request.session_id, {
            "role": "assistant",
            "content": f"Interview scheduled successfully for {request.name} on {request.interview_date} at {request.interview_time}. Confirmation email {'sent' if confirmation_sent else 'failed to send'}.",
            "timestamp": datetime.utcnow().isoformat(),
            "booking_id": str(booking.id)
        })
        
        return InterviewBookingResponse(
            booking_id=str(booking.id),
            name=booking.name,
            email=booking.email,
            interview_date=booking.interview_date,
            interview_time=booking.interview_time,
            status=booking.status,
            confirmation_sent=booking.confirmation_sent,
            created_at=booking.created_at
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date/time format: {str(e)}")
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Error booking interview: {str(e)}")

@router.get("/bookings")
async def list_bookings(
    skip: int = 0,
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """List all interview bookings"""
    try:
        result = await db.execute(
            select(InterviewBooking)
            .offset(skip)
            .limit(limit)
            .order_by(InterviewBooking.created_at.desc())
        )
        bookings = result.scalars().all()
        
        booking_responses = [
            InterviewBookingResponse(
                booking_id=str(booking.id),
                name=booking.name,
                email=booking.email,
                interview_date=booking.interview_date,
                interview_time=booking.interview_time,
                status=booking.status,
                confirmation_sent=booking.confirmation_sent,
                created_at=booking.created_at
            )
            for booking in bookings
        ]
        
        return {"bookings": booking_responses, "total": len(booking_responses)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing bookings: {str(e)}")
