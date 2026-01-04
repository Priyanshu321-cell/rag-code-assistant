# test_query_expansion.py

import os
from src.retrieval.query_expander import QueryExpander


def test_expansion():
    """Test query expansion"""
    
    # Set API key
    # export ANTHROPIC_API_KEY=your_key_here
    
    expander = QueryExpander()
    
    test_queries = [
        "authentication",
        "routing",
        "validate",
        "errors",
        "middleware"
    ]
    
    print("\n" + "="*80)
    print("QUERY EXPANSION TESTS")
    print("="*80)
    
    for query in test_queries:
        print(f"\n📝 Original: '{query}'")
        print("-" * 40)
        
        expanded = expander.expand(query, n_variations=3)
        
        print("Expanded queries:")
        for i, q in enumerate(expanded, 1):
            marker = "  [ORIGINAL]" if i == 1 else ""
            print(f"  {i}. {q}{marker}")
    
    print("\n" + "="*80)


def test_with_search():
    """Test expansion impact on search results"""
    
    from src.retrieval.embedder import Embedder
    from src.retrieval.vector_store import VectorStore
    from src.retrieval.bm25_search import BM25Search
    from src.retrieval.hybrid_search import HybridSearch
    import pickle
    
    # Initialize
    embedder = Embedder()
    vector_store = VectorStore(embedder)
    
    with open("data/processed/bm25_index.pkl", 'rb') as f:
        bm25 = pickle.load(f)
    
    expander = QueryExpander()
    
    # Hybrid without expansion
    hybrid_basic = HybridSearch(bm25, vector_store)
    
    # Hybrid with expansion
    hybrid_expanded = HybridSearch(bm25, vector_store, query_expander=expander)
    
    query = "validate"
    
    print("\n" + "="*80)
    print(f"Query: '{query}'")
    print("="*80)
    
    # Without expansion
    print("\n❌ WITHOUT Query Expansion:")
    results = hybrid_basic.search(query, n_results=5, expand_query=False)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['metadata']['function']}")
    
    # With expansion
    print("\n✅ WITH Query Expansion:")
    results = hybrid_expanded.search(query, n_results=5, expand_query=True)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['metadata']['function']}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    test_expansion()
    
    test_with_search()  # Uncomment after setting up API key
