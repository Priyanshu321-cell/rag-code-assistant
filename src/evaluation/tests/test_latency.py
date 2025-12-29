# test_latency.py

import time
import numpy as np
from pathlib import Path
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_search import BM25Search
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import Reranker
import pickle


def measure_end_to_end_latency():
    """Comprehensive latency measurement"""
    
    print("\n" + "="*60)
    print("LATENCY MEASUREMENT")
    print("="*60)
    
    # Initialize
    print("\nInitializing components...")
    embedder = Embedder()
    vector_store = VectorStore(embedder)
    
    with open("data/processed/bm25_index.pkl", 'rb') as f:
        bm25 = pickle.load(f)
    
    reranker = Reranker()
    hybrid = HybridSearch(bm25, vector_store, reranker=reranker)
    
    # Test queries
    queries = [
        "APIRouter",
        "how to authenticate users",
        "create API endpoint",
        "validate request data",
        "middleware configuration",
        "handle HTTP exceptions",
        "dependency injection",
        "async database queries",
        "WebSocket connection",
        "parse JSON response"
    ]
    
    print(f"\nTesting with {len(queries)} queries, 3 runs each...\n")
    
    # Collect measurements
    all_timings = {
        'vector_only': [],
        'bm25_only': [],
        'hybrid': [],
        'hybrid_rerank': []
    }
    
    for query in queries:
        for _ in range(3):  # 3 runs per query
            
            # Vector only
            start = time.time()
            vector_store.search(query, n_results=10)
            all_timings['vector_only'].append((time.time() - start) * 1000)
            
            # BM25 only
            start = time.time()
            bm25.search(query, top_k=10)
            all_timings['bm25_only'].append((time.time() - start) * 1000)
            
            # Hybrid (no rerank)
            start = time.time()
            hybrid.search(query, n_results=10, use_reranker=False)
            all_timings['hybrid'].append((time.time() - start) * 1000)
            
            # Hybrid + Rerank
            start = time.time()
            hybrid.search(query, n_results=10, use_reranker=True)
            all_timings['hybrid_rerank'].append((time.time() - start) * 1000)
    
    # Print statistics
    print("="*60)
    print("RESULTS (milliseconds)")
    print("="*60)
    print(f"{'Method':<20} {'Mean':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'Max':>8}")
    print("-"*60)
    
    for method, latencies in all_timings.items():
        latencies = np.array(latencies)
        print(f"{method:<20} "
              f"{np.mean(latencies):>8.1f} "
              f"{np.percentile(latencies, 50):>8.1f} "
              f"{np.percentile(latencies, 95):>8.1f} "
              f"{np.percentile(latencies, 99):>8.1f} "
              f"{np.max(latencies):>8.1f}")
    
    print("="*60)
    
    # Overhead analysis
    print("\nOVERHEAD ANALYSIS:")
    print("-"*60)
    
    hybrid_overhead = np.mean(all_timings['hybrid']) - max(
        np.mean(all_timings['vector_only']),
        np.mean(all_timings['bm25_only'])
    )
    print(f"Hybrid merge overhead:  {hybrid_overhead:>6.1f}ms")
    
    rerank_overhead = (np.mean(all_timings['hybrid_rerank']) - 
                       np.mean(all_timings['hybrid']))
    print(f"Reranking overhead:     {rerank_overhead:>6.1f}ms")
    
    total_overhead = (np.mean(all_timings['hybrid_rerank']) - 
                      np.mean(all_timings['vector_only']))
    print(f"Total overhead:         {total_overhead:>6.1f}ms")
    
    print("="*60)


if __name__ == "__main__":
    measure_end_to_end_latency()