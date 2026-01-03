# test_reranker_behavior.py

from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_search import BM25Search
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import Reranker
import pickle
import json

# Load
embedder = Embedder()
store = VectorStore(embedder)
with open("data/processed/bm25_index.pkl", 'rb') as f:
    bm25 = pickle.load(f)

reranker = Reranker()

# Load one test query
with open("data/evaluation/golden_dataset.json") as f:
    dataset = json.load(f)

query_data = dataset[5]  # Pick one
query = query_data['query']
relevant_ids = set(query_data['expected_chunk_ids'])

print(f"Query: {query}")
print(f"Relevant IDs: {relevant_ids}")

# Get hybrid results BEFORE reranking
hybrid = HybridSearch(bm25, store)
results_before = hybrid.search(query, n_results=10, use_classifier=False)

print(f"\nBEFORE Reranking:")
for i, r in enumerate(results_before[:5], 1):
    match = "✓" if r['id'] in relevant_ids else "✗"
    print(f"  {i}. {match} {r['metadata']['function']} (RRF: {r['rrf_score']:.4f})")

# Now WITH reranking
hybrid_rerank = HybridSearch(bm25, store, reranker=reranker)
results_after = hybrid_rerank.search(query, n_results=10, use_classifier=False)

print(f"\nAFTER Reranking:")
for i, r in enumerate(results_after[:5], 1):
    match = "✓" if r['id'] in relevant_ids else "✗"
    score = r.get('rerank_score', 0)
    print(f"  {i}. {match} {r['metadata']['function']} (Rerank: {score:.4f})")

# Check if relevant moved up or down
before_positions = [i for i, r in enumerate(results_before) if r['id'] in relevant_ids]
after_positions = [i for i, r in enumerate(results_after) if r['id'] in relevant_ids]

print(f"\nRelevant result positions:")
print(f"  Before: {[p+1 for p in before_positions]}")
print(f"  After:  {[p+1 for p in after_positions]}")