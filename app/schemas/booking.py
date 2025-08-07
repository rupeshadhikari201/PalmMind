from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional

class InterviewBookingRequest(BaseModel):
    session_id: str = Field(..., description="Chat session ID")
    name: str = Field(..., min_length=1, max_length=255)
    email: EmailStr
    interview_date: str = Field(..., description="Date in YYYY-MM-DD format")
    interview_time: str = Field(..., description="Time in HH:MM format")
    notes: Optional[str] = None

class InterviewBookingResponse(BaseModel):
    booking_id: str
    name: str
    email: str
    interview_date: datetime
    interview_time: str
    status: str
    confirmation_sent: bool
    created_at: datetime