import json
from collections import Counter
from pathlib import Path

def analyze_dataset():
    """Analyze the golden dataset"""
    
    with open("data/evaluation/golden_dataset.json") as f:
        dataset = json.load(f)
    
    print("\n" + "="*80)
    print("GOLDEN DATASET ANALYSIS")
    print("="*80)
    
    print(f"\nTotal queries: {len(dataset)}")
    
    # By category
    categories = Counter(q['category'] for q in dataset)
    print(f"\nBy category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat:20s}: {count:2d} queries")
    
    # By difficulty
    difficulties = Counter(q['difficulty'] for q in dataset)
    print(f"\nBy difficulty:")
    for diff, count in sorted(difficulties.items()):
        print(f"  {diff:20s}: {count:2d} queries")
    
    # Relevant results per query
    relevance_counts = [q['num_relevant'] for q in dataset]
    print(f"\nRelevant results per query:")
    print(f"  Mean: {sum(relevance_counts)/len(relevance_counts):.1f}")
    print(f"  Min:  {min(relevance_counts)}")
    print(f"  Max:  {max(relevance_counts)}")
    
    # Show sample queries
    print(f"\nSample queries:")
    for q in dataset[:5]:
        print(f"\n  Query: {q['query']}")
        print(f"  Relevant functions: {', '.join(q['expected_functions'][:3])}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    analyze_dataset()