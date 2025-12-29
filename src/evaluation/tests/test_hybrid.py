# test_hybrid.py

from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_search import BM25Search
from src.retrieval.hybrid_search import HybridSearch


def test_hybrid():
    """Test hybrid search vs individual methods"""
    
    # Test data
    test_chunks = [
        {
            'id': 'chunk_1',
            'text': 'class APIRouter:\n    """Main routing class for FastAPI applications"""\n    def add_route(self): pass',
            'metadata': {'file': 'routing.py', 'function': 'APIRouter'}
        },
        {
            'id': 'chunk_2',
            'text': 'async def authenticate_user(username: str, password: str):\n    """Verify user credentials against database"""\n    return await db.check(username, password)',
            'metadata': {'file': 'auth.py', 'function': 'authenticate_user'}
        },
        {
            'id': 'chunk_3',
            'text': 'def create_api_route(path: str, handler):\n    """Helper function to add new routes to the router"""\n    router.add(path, handler)',
            'metadata': {'file': 'routing.py', 'function': 'create_api_route'}
        },
        {
            'id': 'chunk_4',
            'text': 'async def verify_token(token: str):\n    """Check if authentication token is valid"""\n    return jwt.decode(token)',
            'metadata': {'file': 'auth.py', 'function': 'verify_token'}
        },
        {
            'id': 'chunk_5',
            'text': 'def handle_http_exception(exc: HTTPException):\n    """Process HTTP exceptions and return error response"""\n    return JSONResponse(status_code=exc.status_code)',
            'metadata': {'file': 'errors.py', 'function': 'handle_http_exception'}
        }
    ]
    
    # Initialize all search methods
    print("Initializing search systems...")
    
    embedder = Embedder()
    vector_store = VectorStore(embedder=embedder)
    vector_store.clear()
    vector_store.add_chunks(test_chunks)
    
    bm25 = BM25Search()
    bm25.index_documents(test_chunks)
    
    hybrid = HybridSearch(bm25, vector_store)
    
    print("✓ All search systems ready\n")
    
    # Test cases
    test_queries = [
        ("APIRouter", "Exact class name - BM25 should excel"),
        ("how to authenticate users", "Semantic query - Vector should excel"),
        ("routing", "General concept - Both should contribute"),
        ("verify user credentials", "Mixed: 'verify' (keyword) + semantic meaning"),
    ]
    
    for query, description in test_queries:
        print("="*80)
        print(f"Query: '{query}'")
        print(f"Type: {description}")
        print("="*80)
        
        # BM25 only
        print("\n🔤 BM25 Results:")
        bm25_results = bm25.search(query, top_k=3)
        for i, r in enumerate(bm25_results, 1):
            print(f"  {i}. {r['metadata']['function']} (score: {r['score']:.2f})")
        
        # Vector only
        print("\n🧠 Vector Results:")
        vec_results = vector_store.search(query, n_results=3)
        for i, r in enumerate(vec_results, 1):
            dist = r['distance']
            print(f"  {i}. {r['metadata']['function']} (distance: {dist:.2f})")
        
        # Hybrid
        print("\n🔀 Hybrid Results (RRF):")
        hybrid_results = hybrid.search(query, n_results=3)
        for i, r in enumerate(hybrid_results, 1):
            bm25_r = r.get('bm25_rank', '-')
            vec_r = r.get('vector_rank', '-')
            print(f"  {i}. {r['metadata']['function']} (RRF: {r['rrf_score']:.4f}, BM25 rank: {bm25_r}, Vector rank: {vec_r})")
        
        print()


if __name__ == "__main__":
    test_hybrid()