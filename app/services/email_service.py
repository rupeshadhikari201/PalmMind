import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self):
        self.smtp_host = settings.smtp_host
        self.smtp_port = settings.smtp_port
        self.smtp_username = settings.smtp_username
        self.smtp_password = settings.smtp_password
    
    async def send_interview_confirmation(self, booking_data: Dict[str, Any]) -> bool:
        """Send interview confirmation email"""
        try:
            # Create message
            message = MIMEMultipart()
            message["From"] = self.smtp_username
            message["To"] = booking_data["email"]
            message["Subject"] = "Interview Confirmation"
            
            # Email body
            body = f"""
Dear {booking_data["name"]},

Thank you for scheduling an interview with us!

Interview Details:
- Date: {booking_data["interview_date"]}
- Time: {booking_data["interview_time"]}
- Status: {booking_data.get("status", "Confirmed")}

We look forward to speaking with you.

Best regards,
The Interview Team
            """
            
            message.attach(MIMEText(body, "plain"))
            
            # Send email
            if self.smtp_username and self.smtp_password:
                await aiosmtplib.send(
                    message,
                    hostname=self.smtp_host,
                    port=self.smtp_port,
                    start_tls=True,
                    username=self.smtp_username,
                    password=self.smtp_password,
                )
                logger.info(f"Confirmation email sent to {booking_data['email']}")
                return True
            else:
                logger.warning("SMTP credentials not configured, email not sent")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return False