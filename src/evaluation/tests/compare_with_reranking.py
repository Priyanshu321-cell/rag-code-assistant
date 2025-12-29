# compare_with_reranking.py

"""Compare hybrid search with and without reranking"""

from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_search import BM25Search
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import Reranker


def compare():
    # Initialize (use your saved indices)
    embedder = Embedder()
    vector_store = VectorStore(embedder=embedder)
    
    import pickle
    with open("data/processed/bm25_index.pkl", 'rb') as f:
        bm25 = pickle.load(f)
    
    # Hybrid without reranker
    hybrid_basic = HybridSearch(bm25, vector_store)
    
    # Hybrid with reranker
    reranker = Reranker()
    hybrid_rerank = HybridSearch(bm25, vector_store, reranker=reranker)
    
    # Test queries
    queries = [
        "how to authenticate users",
        "create API endpoint",
        "validate request data"
    ]
    
    for query in queries:
        print("\n" + "="*80)
        print(f"Query: '{query}'")
        print("="*80)
        
        # Without reranking
        print("\n❌ WITHOUT Reranking:")
        results_basic = hybrid_basic.search(query, n_results=5, use_reranker=False)
        for i, r in enumerate(results_basic, 1):
            print(f"  {i}. {r['metadata']['function']} (RRF: {r['rrf_score']:.4f})")
        
        # With reranking
        print("\n✅ WITH Reranking:")
        results_rerank = hybrid_rerank.search(query, n_results=5, use_reranker=True)
        for i, r in enumerate(results_rerank, 1):
            print(f"  {i}. {r['metadata']['function']} (Score: {r['rerank_score']:.4f})")


if __name__ == "__main__":
    compare()