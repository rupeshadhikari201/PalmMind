from sqlalchemy import Column, String, DateTime, Text, Boolean
from sqlalchemy.dialects.postgresql import UUID
from app.core.database import Base
import uuid
from datetime import datetime

class InterviewBooking(Base):
    __tablename__ = "interview_bookings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False)
    interview_date = Column(DateTime, nullable=False)
    interview_time = Column(String(10), nullable=False)
    status = Column(String(20), default="scheduled")
    notes = Column(Text)
    confirmation_sent = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
