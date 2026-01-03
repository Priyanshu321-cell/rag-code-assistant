"""Implements Recall@K, Precision@K, MRR ,and NDCG"""

import numpy as np
from typing import List, Dict , Set
from loguru import logger

class RetrievalMetrics:
    """Calculate satandard IR metrics for search evaluation"""
    
    def __init__(self):
        """Initialize metrics calculator"""
        logger.info("RetrievalMetrics initialized")
        
    def recall_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids : Set[str],
        k: int
    ) -> float:
        """Calculate Recall@K"""
        if not relevant_ids:
            return 0.0
        
        # Get top k results
        top_k = set(retrieved_ids[:k])
        
        # how many relevant items did we retrieve ?
        retrieved_relevant = top_k & relevant_ids
        
        # recall = retrieved relevant / total relevant
        recall = len(retrieved_relevant) / len(relevant_ids)
        
        return recall
    
    def precision_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int
    ) -> float:
        """Calculate Precision@K"""
        if k == 0:
            return 0.0
        
        top_k = retrieved_ids[:k]
        num_relevant = sum(1 for chunk_id in top_k if chunk_id in relevant_ids)
        
        # precision = relevant in top K / k
        precision = num_relevant / k
        
        return precision
    
    def reciprocal_rank(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str]
    ) -> float:
        """Calculate Reciprocal Rank"""
        # position of first relevant result
        for i, chunk_id in enumerate(retrieved_ids, 1):
            if chunk_id in relevant_ids:
                return 1.0 / i
            
        return 0.0
    
    def average_precision(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str]
    )->float:
        """Calculate Average Precision (AP)"""
        if not relevant_ids:
            return 0.0
        
        precision_sum = 0.0
        num_relevant_found = 0
        
        for i , chunk_id in enumerate(retrieved_ids, 1):
            if chunk_id in relevant_ids:
                num_relevant_found += 1
                # Precision at this position
                precision_at_i = num_relevant_found /i
                precision_sum += precision_at_i
                
        if num_relevant_found == 0:
            return 0.0
        
        return precision_sum / len(relevant_ids)
    
    def ndcg_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain at K"""
        if not relevant_ids:
            return 0.0
        
        top_k = retrieved_ids[:k]
        dcg = 0.0
        for i, chunk_id in enumerate(top_k, 1):
            if chunk_id in relevant_ids:
                # binary relevance: 1 || 0
                relevance = 1
                # discounted by position (log base 2)
                dcg += relevance / np.log2(i + 1)
        # calculate ideal dcg (if all relevant were at top)
        num_relevant_int_top_k = min(len(relevant_ids), k)
        idcg = sum(1.0 / np.log2(i+2) for i in range(num_relevant_int_top_k))
        
        if idcg == 0:
            return 0.0
        
        # Normalize
        ndcg = dcg / idcg
        return ndcg
    
    def calculate_all_metrics(
        self,
        retrieved_ids: List[str],
        relevant_ids : Set[str],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """Calculate all mterics for single query"""
        metrics = {}
        
        # calculate for each k value
        for k in k_values:
            metrics[f'recall@{k}'] = self.recall_at_k(retrieved_ids, relevant_ids, k)
            metrics[f'precision@{k}'] = self.precision_at_k(retrieved_ids,relevant_ids,k)
            metrics[f'ndcg@{k}'] = self.ndcg_at_k(retrieved_ids,relevant_ids,k)
            
        # single value metrics
        metrics['mrr'] = self.reciprocal_rank(retrieved_ids, relevant_ids)
        metrics['map'] = self.average_precision(relevant_ids,relevant_ids)
        
        return metrics
    
if __name__ == "__main__":
    # Test metrics
    metrics = RetrievalMetrics()
    
    # Example: 5 relevant chunks, retrieved 10 chunks
    relevant = {"chunk_1", "chunk_3", "chunk_5", "chunk_8", "chunk_12"}
    retrieved = ["chunk_3", "chunk_2", "chunk_5", "chunk_7", "chunk_1",
                 "chunk_9", "chunk_10", "chunk_8", "chunk_11", "chunk_12"]
    
    print("\n" + "="*60)
    print("METRICS TEST")
    print("="*60)
    print(f"\nRelevant chunks: {len(relevant)}")
    print(f"Retrieved chunks: {len(retrieved)}")
    print(f"\nPositions of relevant chunks: {[i+1 for i, c in enumerate(retrieved) if c in relevant]}")
    
    results = metrics.calculate_all_metrics(retrieved, relevant)
    
    print("\nMetrics:")
    for metric, score in sorted(results.items()):
        print(f"  {metric:15s}: {score:.4f}")    
                
            