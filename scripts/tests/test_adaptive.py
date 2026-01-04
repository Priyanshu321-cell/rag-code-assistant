from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_search import BM25Search
from src.retrieval.adaptive_search import AdaptiveSearch
from src.retrieval.reranker import Reranker
import pickle

# Initialize
embedder = Embedder()
vector_store = VectorStore(embedder)

with open("data/processed/bm25_index.pkl", 'rb') as f:
    bm25 = pickle.load(f)

reranker = Reranker()

# Create adaptive search
adaptive = AdaptiveSearch(bm25, vector_store, reranker)

# Test different query types
test_queries = [
    "how to authenticate users",      # Should route to: how_to
    "APIRouter",                       # Should route to: specific_term
    "async endpoint function",         # Should route to: complex
    "routing",                         # Should route to: default
]

print("\n" + "="*80)
print("ADAPTIVE ROUTING TEST")
print("="*80)

for query in test_queries:
    print(f"\nQuery: '{query}'")
    
    # This will show routing decision in logs
    results = adaptive.search(query, n_results=5)
    
    print(f"Top 3 results:")
    for i, r in enumerate(results[:3], 1):
        print(f"  {i}. {r['metadata']['function']}")