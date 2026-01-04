# compare_search.py

from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_search import BM25Search

def compare_searches():
    """Compare BM25 and Vector search on same queries"""
    
    # Test chunks
    test_chunks = [
        {
            'id': 'chunk_1',
            'text': 'class APIRouter:\n    """Main router class for FastAPI"""\n    pass',
            'metadata': {'file': 'routing.py', 'function': 'APIRouter'}
        },
        {
            'id': 'chunk_2',
            'text': 'async def authenticate_user(username, password):\n    """Verify user credentials against database"""\n    return check_db(username, password)',
            'metadata': {'file': 'auth.py', 'function': 'authenticate_user'}
        },
        {
            'id': 'chunk_3',
            'text': 'def create_route(path, handler):\n    """Add new route to router"""\n    router.add(path, handler)',
            'metadata': {'file': 'routing.py', 'function': 'create_route'}
        }
    ]
    
    # Initialize both
    embedder = Embedder()
    vector_store = VectorStore(embedder=embedder)
    vector_store.clear()
    vector_store.add_chunks(test_chunks)
    
    bm25 = BM25Search()
    bm25.index_documents(test_chunks)
    
    # Test queries
    queries = [
        ("APIRouter", "Exact class name"),
        ("how to verify user login", "Semantic query"),
        ("routing", "General concept"),
    ]
    
    print("\n" + "="*80)
    print("BM25 vs VECTOR SEARCH COMPARISON")
    print("="*80)
    
    for query, description in queries:
        print(f"\n📝 Query: '{query}' ({description})")
        print("-"*80)
        
        # BM25 results
        print("\n🔤 BM25 (Keyword):")
        bm25_results = bm25.search(query, top_k=3)
        for i, r in enumerate(bm25_results[:2], 1):
            print(f"  {i}. {r['metadata']['function']} (score: {r['score']:.2f})")
        
        # Vector results
        print("\n🧠 Vector (Semantic):")
        vec_results = vector_store.search(query, n_results=2)
        for i, r in enumerate(vec_results, 1):
            sim = 1 - r['distance']
            print(f"  {i}. {r['metadata']['function']} (similarity: {sim:.2%})")
        
        print()


if __name__ == "__main__":
    compare_searches()