# create_golden_dataset.py

"""
Interactive tool for creating golden dataset.
Searches your index and helps you label correct results.
"""

import json
import pickle
from pathlib import Path
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_search import BM25Search
from src.retrieval.hybrid_search import HybridSearch


def create_dataset_interactively():
    """
    Interactive labeling of queries.
    Shows search results, you mark which are correct.
    """
    
    # Initialize search
    print("Loading search system...")
    embedder = Embedder()
    vector_store = VectorStore(embedder)
    
    with open("data/processed/bm25_index.pkl", 'rb') as f:
        bm25 = pickle.load(f)
    
    hybrid = HybridSearch(bm25, vector_store)
    
    # Queries to label
    queries = [
        # Category: Specific Terms
        ("APIRouter", "specific_term", "easy"),
        ("HTTPException", "specific_term", "easy"),
        ("Request", "specific_term", "easy"),
        ("Response", "specific_term", "easy"),
        ("Depends", "specific_term", "easy"),
        ("BackgroundTasks", "specific_term", "easy"),
        ("WebSocket", "specific_term", "easy"),
        ("UploadFile", "specific_term", "easy"),
        
        # Category: How-To
        ("how to create an API endpoint", "how_to", "medium"),
        ("how to validate request data", "how_to", "medium"),
        ("how to handle authentication", "how_to", "hard"),
        ("how to add middleware", "how_to", "medium"),
        ("how to use dependency injection", "how_to", "hard"),
        ("how to handle file uploads", "how_to", "medium"),
        ("how to add CORS headers", "how_to", "medium"),
        ("how to create background tasks", "how_to", "medium"),
        ("how to handle WebSocket connections", "how_to", "hard"),
        ("how to return custom responses", "how_to", "medium"),
        
        # Category: Concepts
        ("routing", "concept", "medium"),
        ("validation", "concept", "medium"),
        ("authentication", "concept", "hard"),
        ("middleware", "concept", "medium"),
        ("dependencies", "concept", "hard"),
        ("parameters", "concept", "medium"),
        ("responses", "concept", "medium"),
        ("exceptions", "concept", "medium"),
        
        # Category: Code Patterns
        ("async endpoint function", "code_pattern", "medium"),
        ("path operation decorator", "code_pattern", "medium"),
        ("dependency injection pattern", "code_pattern", "hard"),
        ("exception handler", "code_pattern", "medium"),
        ("startup event handler", "code_pattern", "medium"),
        ("request validation", "code_pattern", "medium"),
        ("response model", "code_pattern", "medium"),
        ("background task function", "code_pattern", "medium"),
    ]
    
    golden_dataset = []
    
    print(f"\n{'='*80}")
    print(f"INTERACTIVE DATASET CREATION")
    print(f"{'='*80}")
    print(f"\nTotal queries to label: {len(queries)}")
    print("\nFor each query, you'll see top 15 results.")
    print("Mark which ones are RELEVANT (actually answer the query).\n")
    
    for idx, (query, category, difficulty) in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {idx}/{len(queries)}: '{query}'")
        print(f"Category: {category}, Difficulty: {difficulty}")
        print(f"{'='*80}")
        
        # Search with hybrid
        results = hybrid.search(query, n_results=15)
        
        # Show results
        print("\nTop 15 results:")
        for i, r in enumerate(results, 1):
            func_name = r['metadata']['function']
            file_name = r['metadata'].get('file', 'unknown')
            print(f"  [{i:2d}] {func_name:<40s} ({file_name})")
        
        # Get user input
        print("\nWhich results are RELEVANT? (enter numbers separated by spaces)")
        print("Example: 1 2 5 7  (or press Enter to skip)")
        user_input = input("Relevant results: ").strip()
        
        if user_input:
            try:
                relevant_indices = [int(x)-1 for x in user_input.split()]
                relevant_chunks = [results[i] for i in relevant_indices]
                
                entry = {
                    "query_id": f"Q{idx:03d}",
                    "query": query,
                    "category": category,
                    "difficulty": difficulty,
                    "expected_chunk_ids": [r['id'] for r in relevant_chunks],
                    "expected_functions": [r['metadata']['function'] for r in relevant_chunks],
                    "num_relevant": len(relevant_chunks)
                }
                
                golden_dataset.append(entry)
                print(f"✓ Marked {len(relevant_chunks)} results as relevant")
                
            except (ValueError, IndexError) as e:
                print(f"✗ Invalid input, skipping query: {e}")
        else:
            print("⊘ Skipped (no relevant results marked)")
    
    # Save dataset
    output_path = Path("data/evaluation/golden_dataset.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(golden_dataset, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"DATASET CREATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total queries labeled: {len(golden_dataset)}")
    print(f"Saved to: {output_path}")
    print(f"\nCategory breakdown:")
    
    from collections import Counter
    categories = Counter(q['category'] for q in golden_dataset)
    for cat, count in categories.items():
        print(f"  {cat}: {count} queries")
    
    difficulties = Counter(q['difficulty'] for q in golden_dataset)
    print(f"\nDifficulty breakdown:")
    for diff, count in difficulties.items():
        print(f"  {diff}: {count} queries")


if __name__ == "__main__":
    create_dataset_interactively()