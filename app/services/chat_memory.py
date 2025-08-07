from typing import List, Dict, Any, Optional
import json
from app.core.database import redis_client

class ChatMemoryService:
    def __init__(self, ttl: int = 3600):  # 1 hour TTL
        self.redis = redis_client
        self.ttl = ttl
    
    async def save_message(self, session_id: str, message: Dict[str, Any]):
        """Save a message to chat history"""
        key = f"chat:{session_id}"
        message_json = json.dumps(message)
        
        # Add to list and set expiration
        await self.redis.lpush(key, message_json)
        await self.redis.expire(key, self.ttl)
    
    async def get_chat_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get chat history for a session"""
        key = f"chat:{session_id}"
        messages = await self.redis.lrange(key, 0, limit - 1)
        
        return [json.loads(msg) for msg in reversed(messages)]
    
    async def clear_chat_history(self, session_id: str) -> bool:
        """Clear chat history for a session"""
        key = f"chat:{session_id}"
        result = await self.redis.delete(key)
        return result > 0
    
    async def get_conversation_context(self, session_id: str, max_turns: int = 5) -> str:
        """Get formatted conversation context"""
        history = await self.get_chat_history(session_id, max_turns * 2)
        
        context_parts = []
        for msg in history:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            context_parts.append(f"{role.title()}: {content}")
        
        return "\n".join(context_parts)