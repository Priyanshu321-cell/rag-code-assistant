# analyze_ground_truth.py

import json

with open("data/evaluation/golden_dataset.json") as f:
    dataset = json.load(f)

print("Ground Truth Analysis:")
print("="*80)

# Relevant results per query
relevant_counts = [len(q['expected_chunk_ids']) for q in dataset]

print(f"\nRelevant results per query:")
print(f"  Mean: {sum(relevant_counts)/len(relevant_counts):.2f}")
print(f"  Min: {min(relevant_counts)}")
print(f"  Max: {max(relevant_counts)}")
print(f"  Total relevant chunks: {sum(relevant_counts)}")

# Queries with only 1 relevant result
single_relevant = [q for q in dataset if len(q['expected_chunk_ids']) == 1]
print(f"\nQueries with only 1 relevant result: {len(single_relevant)}/{len(dataset)}")

if len(single_relevant) > 10:
    print("⚠️  Too many queries with single relevant result!")
    print("   This might mean you were too conservative in labeling.")

# Show some examples
print(f"\nSample queries:")
for q in dataset[:5]:
    print(f"\n  Query: {q['query']}")
    print(f"  Relevant: {len(q['expected_chunk_ids'])} functions")
    print(f"  Functions: {', '.join(q['expected_functions'][:3])}")