# check_duplicates.py

def check_all_duplicates():
    from pathlib import Path
    from src.ingestion.parser import parse_directory
    from src.ingestion.chunker import chunk_by_function
    from collections import Counter
    
    # Parse
    functions = parse_directory(Path("data/raw/fastapi/fastapi"))
    func_sigs = [(f['file'], f['name'], f['line_start']) for f in functions]
    
    print("="*60)
    print("DUPLICATE CHECK REPORT")
    print("="*60)
    
    # Check 1: Functions
    print(f"\n1. Functions: {len(func_sigs)} total, {len(set(func_sigs))} unique")
    if len(func_sigs) != len(set(func_sigs)):
        dups = [sig for sig, cnt in Counter(func_sigs).items() if cnt > 1]
        print(f"   ⚠️  {len(dups)} duplicates found: {dups[:3]}")
    else:
        print("   ✅ No duplicates")
    
    # Check 2: Chunks
    chunks = chunk_by_function(functions)
    chunk_ids = [c['id'] for c in chunks]
    
    print(f"\n2. Chunks: {len(chunk_ids)} total, {len(set(chunk_ids))} unique")
    if len(chunk_ids) != len(set(chunk_ids)):
        dups = [id for id, cnt in Counter(chunk_ids).items() if cnt > 1]
        print(f"   ⚠️  {len(dups)} duplicate IDs: {dups[:3]}")
    else:
        print("   ✅ No duplicates")
    
    # Check 3: Test search
    from src.retrieval.embedder import Embedder
    from src.retrieval.vector_store import VectorStore
    import pickle
    
    embedder = Embedder()
    store = VectorStore(embedder)
    
    with open("data/processed/bm25_index.pkl", 'rb') as f:
        bm25 = pickle.load(f)
    
    from src.retrieval.hybrid_search import HybridSearch
    hybrid = HybridSearch(bm25, store)
    
    results = hybrid.search("authentication", n_results=10)
    result_ids = [r['id'] for r in results]
    
    print(f"\n3. Search Results: {len(result_ids)} total, {len(set(result_ids))} unique")
    if len(result_ids) != len(set(result_ids)):
        dups = [id for id, cnt in Counter(result_ids).items() if cnt > 1]
        print(f"   ⚠️  {len(dups)} duplicate results: {dups}")
    else:
        print("   ✅ No duplicates")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    check_all_duplicates()