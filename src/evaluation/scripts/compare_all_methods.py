# compare_all_methods.py

"""
Comprehensive comparison of all retrieval methods.
Tests each method on diverse queries and documents results.
"""

import time
import pickle
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_search import BM25Search
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import Reranker
from src.retrieval.query_classifier import QueryClassifier


def compare_all_methods():
    """Compare all retrieval methods systematically"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE RETRIEVAL METHOD COMPARISON")
    print("="*80)
    
    # Initialize components
    print("\nInitializing components...")
    embedder = Embedder()
    vector_store = VectorStore(embedder)
    
    with open("data/processed/bm25_index.pkl", 'rb') as f:
        bm25 = pickle.load(f)
    
    reranker = Reranker()
    classifier = QueryClassifier()
    
    # Initialize different search configurations
    methods = {
        'vector_only': vector_store,
        'bm25_only': bm25,
        'hybrid_basic': HybridSearch(bm25, vector_store),
        'hybrid_rerank': HybridSearch(bm25, vector_store, reranker=reranker),
        'hybrid_classified': HybridSearch(bm25, vector_store, reranker=reranker, 
                                          query_classifier=classifier)
    }
    
    # Test queries with categories
    test_queries = {
        'specific_terms': [
            "APIRouter",
            "HTTPException",
            "Request"
        ],
        'semantic': [
            "how to authenticate users",
            "validate request data",
            "handle errors in API"
        ],
        'concepts': [
            "routing",
            "validation",
            "middleware"
        ],
        'code_patterns': [
            "async endpoint function",
            "dependency injection",
            "exception handler"
        ]
    }
    
    # Results storage
    results = defaultdict(lambda: defaultdict(dict))
    latencies = defaultdict(lambda: defaultdict(list))
    
    # Run experiments
    print("\nRunning experiments (3 runs per query)...\n")
    
    for category, queries in test_queries.items():
        print(f"\nCategory: {category.upper()}")
        print("-" * 80)
        
        for query in queries:
            print(f"  Query: '{query}'")
            
            for method_name, method in methods.items():
                
                # Run 3 times for latency stats
                for run in range(3):
                    start = time.time()
                    
                    # Search with appropriate method
                    if method_name == 'vector_only':
                        search_results = method.search(query, n_results=5)
                    elif method_name == 'bm25_only':
                        search_results = method.search(query, top_k=5)
                    elif method_name == 'hybrid_basic':
                        search_results = method.search(query, n_results=5, 
                                                       use_classifier=False)
                    elif method_name == 'hybrid_rerank':
                        search_results = method.search(query, n_results=5, 
                                                       use_classifier=False)
                    elif method_name == 'hybrid_classified':
                        search_results = method.search(query, n_results=5, 
                                                       use_classifier=True)
                    
                    latency = (time.time() - start) * 1000
                    latencies[category][method_name].append(latency)
                
                # Store results from last run
                results[category][query][method_name] = [
                    r['metadata']['function'] for r in search_results[:3]
                ]
            
            print(f"    ✓ Tested with all methods")
    
    # Print results
    print_results(results, latencies, test_queries)
    
    # Save results
    save_results(results, latencies)


def print_results(results, latencies, test_queries):
    """Print formatted comparison results"""
    
    print("\n" + "="*80)
    print("RESULTS BY QUERY CATEGORY")
    print("="*80)
    
    for category, queries in test_queries.items():
        print(f"\n{'='*80}")
        print(f"Category: {category.upper()}")
        print('='*80)
        
        for query in queries:
            print(f"\n📝 Query: '{query}'")
            print("-" * 80)
            
            for method_name in ['vector_only', 'bm25_only', 'hybrid_basic', 
                               'hybrid_rerank', 'hybrid_classified']:
                top_results = results[category][query][method_name]
                print(f"\n  {method_name:20s}:")
                for i, func in enumerate(top_results, 1):
                    print(f"    {i}. {func}")
    
    # Print latency summary
    print("\n" + "="*80)
    print("LATENCY SUMMARY (milliseconds)")
    print("="*80)
    
    import numpy as np
    
    print(f"\n{'Method':<25} {'Overall Mean':>12} {'Min Category':>15} {'Max Category':>15}")
    print("-" * 80)
    
    for method_name in ['vector_only', 'bm25_only', 'hybrid_basic', 
                       'hybrid_rerank', 'hybrid_classified']:
        
        # Collect all latencies for this method
        all_latencies = []
        category_means = {}
        
        for category in latencies.keys():
            cat_latencies = latencies[category][method_name]
            all_latencies.extend(cat_latencies)
            category_means[category] = np.mean(cat_latencies)
        
        overall_mean = np.mean(all_latencies)
        min_cat = min(category_means.items(), key=lambda x: x[1])
        max_cat = max(category_means.items(), key=lambda x: x[1])
        
        print(f"{method_name:<25} {overall_mean:>12.1f} "
              f"{min_cat[0]:>8}:{min_cat[1]:>5.1f} "
              f"{max_cat[0]:>8}:{max_cat[1]:>5.1f}")


def save_results(results, latencies):
    """Save results to file for documentation"""
    
    import json
    
    output_dir = Path("docs/week2_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    with open(output_dir / "comparison_results.json", 'w') as f:
        json.dump(dict(results), f, indent=2)
    
    # Save latency data
    latency_data = {}
    for category, methods in latencies.items():
        latency_data[category] = {}
        for method, lats in methods.items():
            import numpy as np
            latency_data[category][method] = {
                'mean': float(np.mean(lats)),
                'median': float(np.median(lats)),
                'min': float(np.min(lats)),
                'max': float(np.max(lats))
            }
    
    with open(output_dir / "latency_results.json", 'w') as f:
        json.dump(latency_data, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir}/")


if __name__ == "__main__":
    compare_all_methods()