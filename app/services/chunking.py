from abc import ABC, abstractmethod
from typing import List, Dict, Any
import re

class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        pass

class FixedSizeChunking(ChunkingStrategy):
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunk_metadata = {
                'chunk_index': len(chunks),
                'chunk_size': len(chunk_words),
                'start_word': i,
                'end_word': i + len(chunk_words) - 1,
                **(metadata or {})
            }
            
            chunks.append({
                'text': chunk_text,
                'metadata': chunk_metadata
            })
        
        return chunks

class SemanticChunking(ChunkingStrategy):
    def __init__(self, max_chunk_size: int = 1000, min_chunk_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        # Split by paragraphs and sentences
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para_idx, paragraph in enumerate(paragraphs):
            sentences = re.split(r'[.!?]+', paragraph)
            
            for sent_idx, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Check if adding this sentence exceeds max chunk size
                if len(current_chunk) + len(sentence) > self.max_chunk_size:
                    if len(current_chunk) >= self.min_chunk_size:
                        chunk_metadata = {
                            'chunk_index': len(chunks),
                            'chunk_size': len(current_chunk.split()),
                            'paragraph_start': para_idx,
                            'semantic_boundary': True,
                            **(metadata or {})
                        }
                        
                        chunks.append({
                            'text': current_chunk.strip(),
                            'metadata': chunk_metadata
                        })
                        current_chunk = sentence
                    else:
                        current_chunk += " " + sentence
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
        
        # Add remaining chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunk_metadata = {
                'chunk_index': len(chunks),
                'chunk_size': len(current_chunk.split()),
                'semantic_boundary': True,
                **(metadata or {})
            }
            
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': chunk_metadata
            })
        
        return chunks

def get_chunking_strategy(strategy_name: str) -> ChunkingStrategy:
    strategies = {
        'fixed_size': FixedSizeChunking(),
        'semantic': SemanticChunking()
    }
    return strategies.get(strategy_name, FixedSizeChunking())