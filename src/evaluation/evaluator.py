"""
Evaluation pipeline for running metrics on golden dataset.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from loguru import logger

from src.evaluation.metrics import RetrievalMetrics
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_search import BM25Search
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import Reranker
from src.retrieval.query_classifier import QueryClassifier

class SearchEvaluator:
    """Evaluate search methods on golden dataset."""
    def __init__(self ,golden_dataset_path: str = "data/evaluation/golden_dataset.json"):
        """Initialize evaluator"""
        # load golden dataset
        with open(golden_dataset_path) as f:
            self.golden_dataset = json.load(f)
            
        logger.info(f"Loaded {len(self.golden_dataset)} test queries")
        
        # initialize metrics calculator
        self.metrics_calc = RetrievalMetrics()
        
        # initialize search components
        logger.info("Initialize search components ... ")
        self.embedder = Embedder()
        self.vector_store = VectorStore(self.embedder)
        
        with open("data/processed/bm25_index.pkl", 'rb') as f:
            self.bm25 = BM25Search()
            self.bm25 = pickle.load(f)
            
        self.reranker = Reranker()
        self.classifier = QueryClassifier()
        
        # initialize search methods to evaluate
        self.search_methods = self._initialize_search_methods()
        
    def _initialize_search_methods(self) -> Dict:
        """Initialize different search configurations"""
        return {
            'vector_only': {
                'searcher': self.vector_store,
                'search_func': lambda q: self.vector_store.search(q, n_results=10)
            },
            'bm25_only': {
                'searcher': self.bm25,
                'search_func': lambda q: self.bm25.search(q, top_k=10)
            },
            'hybrid_basic': {
                'searcher': HybridSearch(self.bm25, self.vector_store),
                'search_func': lambda q: HybridSearch(self.bm25, self.vector_store).search(
                    q, n_results=10, use_classifier=False
                )
            },
            'hybrid_rerank': {
                'searcher': HybridSearch(self.bm25, self.vector_store, reranker=self.reranker),
                'search_func': lambda q: HybridSearch(
                    self.bm25, self.vector_store, reranker=self.reranker
                ).search(q, n_results=10, use_classifier=False)
            },
            'hybrid_classified': {
                'searcher': HybridSearch(
                    self.bm25, self.vector_store, 
                    reranker=self.reranker, query_classifier=self.classifier
                ),
                'search_func': lambda q: HybridSearch(
                    self.bm25, self.vector_store,
                    reranker=self.reranker, query_classifier=self.classifier
                ).search(q, n_results=10, use_classifier=True)
            }
        }
        
    def compare_all_methods_by_category(self) -> Dict:
        """Compare all methods broken down by query category"""
        logger.info("Running category-wise comparison for all methods")
        
        all_results = {}
        
        for method_name in self.search_methods.keys():
            logger.info(f"Evaluating {method_name} by category...")
            all_results[method_name] = self.evaluate_by_category(method_name)
            
        return all_results
    
    def print_category_comparison(self, results:Dict):
        """Print category comparison in formatted table"""
        print(f"\n" + "="*100)
        print("PERFORMANCE BY QUERY CATEGORY")
        print("="*100)
    
        # Get all categories
        categories = list(next(iter(results.values())).keys())
        
        for category in categories:
            print(f"\n{'='*100}")
            print(f"Category: {category.upper()}")
            print('='*100)
            
            print(f"{'Method':<25} {'Recall@5':>12} {'Precision@5':>12} {'MRR':>12} {'NDCG@5':>12}")
            print("-" * 100)
            
            for method_name in results.keys():
                metrics = results[method_name][category]
                
                print(f"{method_name:<25} "
                    f"{metrics['recall@5']:>12.4f} "
                    f"{metrics['precision@5']:>12.4f} "
                    f"{metrics['mrr']:>12.4f} "
                    f"{metrics['ndcg@5']:>12.4f}")
        
    def evaluate_method(self, method_name: str, k_values: List[int] = [1, 3, 5, 10]) -> Dict:
        """
        Evaluate a single search method on all queries.
        
        Args:
            method_name: Name of method to evaluate
            k_values: K values for metrics
            
        Returns:
            Dictionary of averaged metrics
        """
        logger.info(f"Evaluating method: {method_name}")
        
        method = self.search_methods[method_name]
        search_func = method['search_func']
        
        # Store metrics for each query
        all_metrics = defaultdict(list)
        
        for query_data in self.golden_dataset:
            query = query_data['query']
            relevant_ids = set(query_data['expected_chunk_ids'])
            
            # Search
            results = search_func(query)
            retrieved_ids = [r['id'] for r in results]
            
            # DEBUG: Check result count
            if len(results) < 10:
                print(f"⚠️  {method_name} on '{query}': only {len(results)} results (expected 10)")
                
        # Evaluate each query
        for query_data in self.golden_dataset:
            query = query_data['query']
            relevant_ids = set(query_data['expected_chunk_ids'])
            
            # Search
            results = search_func(query)
            retrieved_ids = [r['id'] for r in results]
            
            # Calculate metrics
            metrics = self.metrics_calc.calculate_all_metrics(
                retrieved_ids, relevant_ids, k_values
            )
            
            # Store
            for metric_name, score in metrics.items():
                all_metrics[metric_name].append(score)
        
        # Average across all queries
        averaged_metrics = {}
        for metric_name, scores in all_metrics.items():
            averaged_metrics[metric_name] = sum(scores) / len(scores)
        
        logger.info(f"✓ Completed evaluation for {method_name}")
        return averaged_metrics
    
    def evaluate_all_methods(self) -> Dict:
        """
        Evaluate all search methods.
        
        Returns:
            Dictionary mapping method names to their metrics
        """
        results = {}
        
        for method_name in self.search_methods.keys():
            results[method_name] = self.evaluate_method(method_name)
        
        return results
    
    
    def evaluate_by_category(self, method_name: str) -> Dict:
        """
        Evaluate method broken down by query category.
        
        Args:
            method_name: Name of method to evaluate
            
        Returns:
            Dictionary mapping categories to metrics
        """
        logger.info(f"Evaluating {method_name} by category")
        
        method = self.search_methods[method_name]
        search_func = method['search_func']
        
        # Group queries by category
        by_category = defaultdict(list)
        for query_data in self.golden_dataset:
            by_category[query_data['category']].append(query_data)
        
        # Evaluate each category
        category_results = {}
        
        for category, queries in by_category.items():
            all_metrics = defaultdict(list)
            
            for query_data in queries:
                query = query_data['query']
                relevant_ids = set(query_data['expected_chunk_ids'])
                
                results = search_func(query)
                retrieved_ids = [r['id'] for r in results]
                
                metrics = self.metrics_calc.calculate_all_metrics(
                    retrieved_ids, relevant_ids
                )
                
                for metric_name, score in metrics.items():
                    all_metrics[metric_name].append(score)
            
            # Average for this category
            category_results[category] = {
                metric: sum(scores) / len(scores)
                for metric, scores in all_metrics.items()
            }
        
        return category_results
    
    
    def print_results(self, results: Dict):
        """Print evaluation results in formatted table"""
        
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        # Print header
        metrics_to_show = ['recall@1', 'recall@3', 'recall@5', 'precision@5', 'mrr', 'ndcg@5']
        print(f"\n{'Method':<20}", end='')
        for metric in metrics_to_show:
            print(f"{metric:>12}", end='')
        print()
        print("-" * 80)
        
        # Print each method
        for method, scores in results.items():
            print(f"{method:<20}", end='')
            for metric in metrics_to_show:
                score = scores.get(metric, 0.0)
                print(f"{score:>12.4f}", end='')
            print()
        
        print("="*80)
    
    
    def save_results(self, results: Dict, output_path: str = "data/evaluation/results.json"):
        """Save evaluation results to JSON"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    # Run full evaluation
    evaluator = SearchEvaluator()
    
    # Evaluate all methods
    results = evaluator.evaluate_all_methods()
    
    # Print results
    evaluator.print_results(results)
    
    # Save results
    evaluator.save_results(results)
    
    # Evaluate by category (example for one method)
    print("\n\nBY CATEGORY (hybrid_classified):")
    category_results = evaluator.evaluate_by_category('hybrid_classified')
    
    for category, metrics in category_results.items():
        print(f"\n{category}:")
        print(f"  Recall@5: {metrics['recall@5']:.4f}")
        print(f"  Precision@5: {metrics['precision@5']:.4f}")
        print(f"  MRR: {metrics['mrr']:.4f}")