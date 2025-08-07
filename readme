## API Documentation

Once running, visit:

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Key Endpoints

#### Document Management

- `POST /api/v1/documents/upload` - Upload documents
- `GET /api/v1/documents/` - List documents
- `GET /api/v1/documents/{id}/chunks` - Get document chunks
- `DELETE /api/v1/documents/{id}` - Delete document

#### Chat & RAG

- `POST /api/v1/chat/query` - Process chat queries
- `GET /api/v1/chat/history/{session_id}` - Get chat history
- `DELETE /api/v1/chat/history/{session_id}` - Clear history

#### Interview Booking

- `POST /api/v1/chat/book-interview` - Book interview
- `GET /api/v1/chat/bookings` - List bookings

## Installation

```bash
# Clone Repo

# Install dependencies
pip install -r requirements.txt

# Set up databases (PostgreSQL, MongoDB, Redis, Qdrant)
# Update .env with your database URLs

# Start the application
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
