from typing import List, Tuple, Set
import numpy as np

class EvaluationMetrics:
    @staticmethod
    def calculate_precision_recall_f1(
        retrieved: List[str], 
        relevant: List[str]
    ) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score"""
        
        retrieved_set = set(retrieved)
        relevant_set = set(relevant)
        
        if not retrieved_set:
            return 0.0, 0.0, 0.0
        
        if not relevant_set:
            return 0.0, 0.0, 0.0
        
        # True positives: items that are both retrieved and relevant
        true_positives = len(retrieved_set.intersection(relevant_set))
        
        # Precision: TP / (TP + FP) = TP / total_retrieved
        precision = true_positives / len(retrieved_set)
        
        # Recall: TP / (TP + FN) = TP / total_relevant
        recall = true_positives / len(relevant_set)
        
        # F1 Score: 2 * (precision * recall) / (precision + recall)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return precision, recall, f1
    
    @staticmethod
    def calculate_accuracy(predicted: List[str], actual: List[str]) -> float:
        """Calculate accuracy score"""
        if len(predicted) != len(actual):
            return 0.0
        
        correct = sum(1 for p, a in zip(predicted, actual) if p == a)
        return correct / len(predicted)
    
    @staticmethod
    def calculate_mean_reciprocal_rank(rankings: List[List[str]], relevant: List[str]) -> float:
        """Calculate Mean Reciprocal Rank (MRR)"""
        reciprocal_ranks = []
        
        for ranking in rankings:
            reciprocal_rank = 0.0
            for i, item in enumerate(ranking):
                if item in relevant:
                    reciprocal_rank = 1.0 / (i + 1)
                    break
            reciprocal_ranks.append(reciprocal_rank)
        
        return np.mean(reciprocal_ranks)
    
    @staticmethod
    def calculate_ndcg(rankings: List[List[str]], relevant: List[str], k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain (NDCG)"""
        def dcg(relevances: List[float]) -> float:
            return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
        
        ndcg_scores = []
        
        for ranking in rankings:
            # Create relevance scores (1 if relevant, 0 if not)
            relevances = [1.0 if item in relevant else 0.0 for item in ranking[:k]]
            
            # Calculate DCG
            dcg_score = dcg(relevances)
            
            # Calculate ideal DCG (sort relevances in descending order)
            ideal_relevances = sorted(relevances, reverse=True)
            ideal_dcg = dcg(ideal_relevances)
            
            # Calculate NDCG
            if ideal_dcg > 0:
                ndcg = dcg_score / ideal_dcg
            else:
                ndcg = 0.0
            
            ndcg_scores.append(ndcg)
        
        return np.mean(ndcg_scores)