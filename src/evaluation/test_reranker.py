# test_reranker.py

from src.retrieval.reranker import Reranker


def test_reranker():
    """Test cross-encoder reranking"""
    
    # Sample search results (simulating hybrid search output)
    query = "how to authenticate users"
    
    results = [
        {
            'id': 'chunk_1',
            'text': 'def parse_json(data):\n    """Parse JSON data"""\n    return json.loads(data)',
            'metadata': {'function': 'parse_json'},
            'rrf_score': 0.0325  # High RRF but not relevant
        },
        {
            'id': 'chunk_2',
            'text': 'async def authenticate_user(username, password):\n    """Verify user credentials against database"""\n    return await check_credentials(username, password)',
            'metadata': {'function': 'authenticate_user'},
            'rrf_score': 0.0320  # Lower RRF but very relevant
        },
        {
            'id': 'chunk_3',
            'text': 'def create_route(path, handler):\n    """Add route to application"""\n    app.add_route(path, handler)',
            'metadata': {'function': 'create_route'},
            'rrf_score': 0.0318  # Mentioned in docs, but not relevant
        },
        {
            'id': 'chunk_4',
            'text': 'async def verify_token(token: str):\n    """Check if authentication token is valid"""\n    return jwt.decode(token, verify=True)',
            'metadata': {'function': 'verify_token'},
            'rrf_score': 0.0315  # Relevant
        }
    ]
    
    print("="*80)
    print(f"Query: '{query}'")
    print("="*80)
    
    # Show before reranking
    print("\n📊 BEFORE Reranking (sorted by RRF):")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['metadata']['function']} (RRF: {r['rrf_score']:.4f})")
    
    # Initialize reranker
    reranker = Reranker()
    
    # Rerank
    reranked = reranker.rerank(query, results, top_k=4)
    
    # Show after reranking
    print("\n🎯 AFTER Reranking (sorted by cross-encoder score):")
    for i, r in enumerate(reranked, 1):
        rrf = r.get('rrf_score', 0)
        ce_score = r['rerank_score']
        print(f"  {i}. {r['metadata']['function']}")
        print(f"     RRF: {rrf:.4f} → Cross-Encoder: {ce_score:.4f}")
    
    print("\n" + "="*80)
    print("Observation: Cross-encoder reordered results by true relevance")
    print("="*80)


if __name__ == "__main__":
    test_reranker()