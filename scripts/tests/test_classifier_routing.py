# test_classifier_routing.py

from src.retrieval.query_classifier import QueryClassifier
import json

classifier = QueryClassifier()

with open("data/evaluation/golden_dataset.json") as f:
    dataset = json.load(f)

print("Classification Analysis:")
print("="*80)

from collections import Counter
classifications = Counter()

for entry in dataset:
    query = entry['query']
    expected_category = entry['category']
    
    query_type = classifier.classify(query)
    config = classifier.get_search_config(query_type)
    
    classifications[query_type.value] += 1
    
    # Check if classification seems wrong
    if expected_category == 'specific_term' and query_type.value != 'specific_term':
        print(f"⚠️  Misclassified: '{query}'")
        print(f"    Expected: {expected_category}, Got: {query_type.value}")
        print(f"    Config: BM25={config['bm25_weight']}, Vector={config['vector_weight']}")

print(f"\nClassification distribution:")
for qtype, count in classifications.items():
    print(f"  {qtype}: {count} queries")