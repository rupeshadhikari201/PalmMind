import asyncio
from typing import List, Dict, Any, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import numpy as np

from app.services.chunking import get_chunking_strategy
from app.services.embeddings import EmbeddingService
from app.services.vector_store import get_vector_store
from app.services.rag_engine import RAGEngine
from app.evaluation.metrics import EvaluationMetrics

class RAGEvaluator:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = get_vector_store()
        self.rag_engine = RAGEngine()
        self.metrics = EvaluationMetrics()
    
    async def evaluate_chunking_strategies(
        self, 
        test_documents: List[str],
        ground_truth_chunks: List[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate different chunking strategies"""
        
        strategies = ['fixed_size', 'semantic']
        results = {}
        
        for strategy in strategies:
            print(f"Evaluating chunking strategy: {strategy}")
            
            chunking_service = get_chunking_strategy(strategy)
            strategy_results = {
                'total_documents': len(test_documents),
                'total_chunks': 0,
                'avg_chunk_size': 0,
                'processing_time': 0
            }
            
            start_time = time.time()
            all_chunks = []
            
            for doc in test_documents:
                chunks = chunking_service.chunk_text(doc)
                all_chunks.extend(chunks)
            
            strategy_results['processing_time'] = time.time() - start_time
            strategy_results['total_chunks'] = len(all_chunks)
            strategy_results['avg_chunk_size'] = np.mean([len(chunk['text'].split()) for chunk in all_chunks])
            
            # Calculate coherence score (simple heuristic)
            coherence_scores = []
            for chunk in all_chunks:
                text = chunk['text']
                words = text.split()
                # Simple coherence: ratio of unique words to total words
                coherence = len(set(words)) / len(words) if words else 0
                coherence_scores.append(coherence)
            
            strategy_results['avg_coherence'] = np.mean(coherence_scores)
            results[strategy] = strategy_results
        
        return results
    
    async def evaluate_similarity_search(
        self,
        test_queries: List[str],
        relevant_docs: List[List[str]],  # Ground truth relevant documents for each query
        algorithms: List[str] = ['cosine']
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate similarity search algorithms"""
        
        results = {}
        
        for algorithm in algorithms:
            print(f"Evaluating similarity algorithm: {algorithm}")
            
            precision_scores = []
            recall_scores = []
            f1_scores = []
            latencies = []
            
            for i, query in enumerate(test_queries):
                start_time = time.time()
                
                # Perform search
                search_results = await self.rag_engine.retrieve_relevant_chunks(
                    query=query,
                    top_k=10,
                    similarity_algorithm=algorithm
                )
                
                latency = time.time() - start_time
                latencies.append(latency)
                
                # Extract retrieved document IDs
                retrieved_ids = [result[0] for result in search_results]  # chunk_id
                relevant_ids = relevant_docs[i] if i < len(relevant_docs) else []
                
                # Calculate metrics
                if relevant_ids:
                    precision, recall, f1 = self.metrics.calculate_precision_recall_f1(
                        retrieved_ids, relevant_ids
                    )
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    f1_scores.append(f1)
            
            algorithm_results = {
                'avg_precision': np.mean(precision_scores) if precision_scores else 0,
                'avg_recall': np.mean(recall_scores) if recall_scores else 0,
                'avg_f1': np.mean(f1_scores) if f1_scores else 0,
                'avg_latency': np.mean(latencies),
                'total_queries': len(test_queries)
            }
            
            results[algorithm] = algorithm_results
        
        return results
    
    async def comprehensive_evaluation(
        self,
        test_documents: List[str],
        test_queries: List[str],
        relevant_docs: List[List[str]] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive evaluation of the RAG system"""
        
        print("Starting comprehensive RAG evaluation...")
        
        # Evaluate chunking strategies
        chunking_results = await self.evaluate_chunking_strategies(test_documents)
        
        # Evaluate similarity search
        similarity_results = await self.evaluate_similarity_search(
            test_queries, relevant_docs or [[] for _ in test_queries]
        )
        
        # Overall system performance
        overall_start = time.time()
        
        # Process test documents with both chunking strategies
        for strategy in ['fixed_size', 'semantic']:
            chunking_service = get_chunking_strategy(strategy)
            all_chunks = []
            chunk_metadata = []
            
            for doc_idx, doc in enumerate(test_documents):
                chunks = chunking_service.chunk_text(doc, {'doc_id': f'test_{doc_idx}'})
                all_chunks.extend([chunk['text'] for chunk in chunks])
                chunk_metadata.extend([chunk['metadata'] for chunk in chunks])
            
            # Generate embeddings
            embeddings = await self.embedding_service.generate_embeddings(all_chunks)
            
            # Store in vector database (temporary collection)
            chunk_ids = [f"eval_{strategy}_{i}" for i in range(len(all_chunks))]
            metadata_with_text = [
                {**meta, 'text': text} 
                for meta, text in zip(chunk_metadata, all_chunks)
            ]
            
            await self.vector_store.add_vectors(embeddings, metadata_with_text, chunk_ids)
        
        overall_time = time.time() - overall_start
        
        return {
            'chunking_evaluation': chunking_results,
            'similarity_evaluation': similarity_results,
            'overall_processing_time': overall_time,
            'evaluation_timestamp': time.time()
        }   