import PyPDF2
import aiofiles
from typing import BinaryIO, Tuple
import hashlib

class DocumentProcessor:
    @staticmethod
    async def extract_text_from_pdf(file_content: bytes) -> str:
        """Extract text from PDF content"""
        try:
            import io
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text
        except Exception as e:
            raise ValueError(f"Error extracting PDF text: {str(e)}")
    
    @staticmethod
    async def extract_text_from_txt(file_content: bytes) -> str:
        """Extract text from TXT content"""
        try:
            return file_content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return file_content.decode('latin-1')
            except Exception as e:
                raise ValueError(f"Error decoding text file: {str(e)}")
    
    @staticmethod
    def calculate_content_hash(content: str) -> str:
        """Calculate SHA256 hash of content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    async def process_document(self, file_content: bytes, filename: str, file_type: str) -> Tuple[str, str]:
        """Process document and return (text_content, content_hash)"""
        if file_type.lower() == 'pdf':
            text = await self.extract_text_from_pdf(file_content)
        elif file_type.lower() == 'txt':
            text = await self.extract_text_from_txt(file_content)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        content_hash = self.calculate_content_hash(text)
        return text, content_hash