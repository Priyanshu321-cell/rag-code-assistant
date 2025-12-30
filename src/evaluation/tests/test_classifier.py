# test_classifier.py

from src.retrieval.query_classifier import QueryClassifier, QueryType


def test_classification():
    """Test query classification"""
    
    classifier = QueryClassifier()
    
    test_cases = [
        # (query, expected_type)
        ("APIRouter", QueryType.SPECIFIC_TERM),
        ("HTTPException", QueryType.SPECIFIC_TERM),
        ("authenticate_user", QueryType.SPECIFIC_TERM),
        ("BaseModel", QueryType.SPECIFIC_TERM),
        
        ("how to authenticate users", QueryType.SEMANTIC),
        ("what is dependency injection", QueryType.SEMANTIC),
        ("best way to validate input", QueryType.SEMANTIC),
        ("explain middleware usage", QueryType.SEMANTIC),
        
        ("authentication", QueryType.CONCEPT),
        ("routing", QueryType.CONCEPT),
        ("validation", QueryType.CONCEPT),
        ("middleware", QueryType.CONCEPT),
        
        ("async database query", QueryType.CODE_PATTERN),
        ("decorator for validation", QueryType.CODE_PATTERN),
        ("context manager example", QueryType.CODE_PATTERN),
        ("async def function", QueryType.CODE_PATTERN),
    ]
    
    print("\n" + "="*80)
    print("QUERY CLASSIFICATION TESTS")
    print("="*80)
    
    correct = 0
    total = len(test_cases)
    
    for query, expected in test_cases:
        result = classifier.classify(query)
        config = classifier.get_search_config(result)
        
        match = "✓" if result == expected else "✗"
        
        print(f"\n{match} Query: '{query}'")
        print(f"  Expected: {expected.value}")
        print(f"  Got:      {result.value}")
        print(f"  Config:   BM25={config['bm25_weight']}, "
              f"Vector={config['vector_weight']}, "
              f"Expand={config['use_expansion']}, "
              f"Rerank={config['use_reranking']}")
        
        if result == expected:
            correct += 1
    
    print("\n" + "="*80)
    print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    print("="*80)


def test_with_search():
    """Test classification impact on search"""
    
    from src.retrieval.embedder import Embedder
    from src.retrieval.vector_store import VectorStore
    from src.retrieval.bm25_search import BM25Search
    from src.retrieval.hybrid_search import HybridSearch
    import pickle
    import time
    
    # Initialize
    embedder = Embedder()
    vector_store = VectorStore(embedder)
    
    with open("data/processed/bm25_index.pkl", 'rb') as f:
        bm25 = pickle.load(f)
    
    classifier = QueryClassifier()
    
    # Without classifier
    hybrid_basic = HybridSearch(bm25, vector_store)
    
    # With classifier
    hybrid_smart = HybridSearch(bm25, vector_store, query_classifier=classifier)
    
    test_queries = [
        ("APIRouter", "SPECIFIC_TERM query"),
        ("how to validate input data", "SEMANTIC query"),
        ("authentication", "CONCEPT query")
    ]
    
    print("\n" + "="*80)
    print("CLASSIFICATION IMPACT ON SEARCH")
    print("="*80)
    
    for query, description in test_queries:
        print(f"\n📝 {description}: '{query}'")
        print("-"*80)
        
        # Without classification
        start = time.time()
        results_basic = hybrid_basic.search(query, n_results=3, use_classifier=False)
        time_basic = (time.time() - start) * 1000
        
        print(f"\n❌ WITHOUT Classification ({time_basic:.0f}ms):")
        for i, r in enumerate(results_basic, 1):
            print(f"  {i}. {r['metadata']['function']}")
        
        # With classification
        start = time.time()
        results_smart = hybrid_smart.search(query, n_results=3, use_classifier=True)
        time_smart = (time.time() - start) * 1000
        
        print(f"\n✅ WITH Classification ({time_smart:.0f}ms):")
        for i, r in enumerate(results_smart, 1):
            print(f"  {i}. {r['metadata']['function']}")
        
        speedup = time_basic / time_smart
        print(f"\nSpeedup: {speedup:.1f}x faster")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Test classification accuracy
    test_classification()
    
    # Test impact on search
    test_with_search()