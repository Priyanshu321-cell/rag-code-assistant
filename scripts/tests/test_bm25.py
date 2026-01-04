from pathlib import Path
from src.ingestion.parser import parse_directory
from src.ingestion.chunker import chunk_by_function
from src.retrieval.bm25_search import BM25Search

def test_bm25():
    """Test BM25 search"""
    
    # Create small test set
    test_chunks = [
        {
            'id': 'chunk_1',
            'text': 'async def authenticate_user(username, password):\n    """Verify user credentials"""\n    return True',
            'metadata': {'file': 'auth.py', 'function': 'authenticate_user'}
        },
        {
            'id': 'chunk_2',
            'text': 'class APIRouter:\n    """Handles API routing"""\n    def add_route(self): pass',
            'metadata': {'file': 'routing.py', 'function': 'APIRouter'}
        },
        {
            'id': 'chunk_3',
            'text': 'def validate_data(data):\n    """Validate input data"""\n    return BaseModel.validate(data)',
            'metadata': {'file': 'validation.py', 'function': 'validate_data'}
        },
        {
            'id': 'chunk_4',
            'text': 'def handle_exception(exc: HTTPException):\n    """Handle HTTP exceptions"""\n    return Response(status=500)',
            'metadata': {'file': 'errors.py', 'function': 'handle_exception'}
        }
    ]
    
    # Build index
    bm25 = BM25Search()
    bm25.index_documents(test_chunks)
    
    print("="*60)
    print("BM25 SEARCH TESTS")
    print("="*60)
    
    # Test 1: Exact term match
    print("\nTest 1: Search for 'APIRouter'")
    results = bm25.search("APIRouter", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['metadata']['function']} (score: {r['score']:.2f})")
    
    # Test 2: Common term
    print("\nTest 2: Search for 'authenticate'")
    results = bm25.search("authenticate", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['metadata']['function']} (score: {r['score']:.2f})")
    
    # Test 3: Multiple terms
    print("\nTest 3: Search for 'HTTP exception'")
    results = bm25.search("HTTP exception", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['metadata']['function']} (score: {r['score']:.2f})")
    
    # Test 4: Semantic query (should work poorly)
    print("\nTest 4: Search for 'how to verify user login'")
    results = bm25.search("how to verify user login", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['metadata']['function']} (score: {r['score']:.2f})")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    test_bm25()
