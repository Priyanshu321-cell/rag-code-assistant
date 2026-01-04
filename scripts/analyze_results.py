# analyze_results.py

"""
Deep analysis of evaluation results.
Generates insights, comparisons, and identifies patterns.
"""

import json
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

from src.evaluation.evaluator import SearchEvaluator
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_search import BM25Search


def analyze_by_category():
    """Detailed analysis by query category"""
    
    print("\n" + "="*100)
    print("COMPREHENSIVE CATEGORY ANALYSIS")
    print("="*100)
    
    evaluator = SearchEvaluator()
    
    # Get category-wise results for all methods
    results = evaluator.compare_all_methods_by_category()
    
    # Print formatted comparison
    evaluator.print_category_comparison(results)
    
    # Identify best method per category
    print("\n" + "="*100)
    print("BEST METHOD PER CATEGORY")
    print("="*100)
    
    categories = list(next(iter(results.values())).keys())
    
    for category in categories:
        print(f"\n{category.upper()}:")
        
        # Compare methods on Recall@5
        category_scores = {
            method: results[method][category]['recall@5']
            for method in results.keys()
        }
        
        # Sort by performance
        ranked = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        
        print(f"  Ranked by Recall@5:")
        for rank, (method, score) in enumerate(ranked, 1):
            marker = "👑" if rank == 1 else f"{rank}."
            print(f"    {marker} {method:<25} {score:.4f}")
    
    return results


def analyze_difficult_queries():
    """Find queries where all methods struggle"""
    
    print("\n" + "="*100)
    print("DIFFICULT QUERIES ANALYSIS")
    print("="*100)
    
    evaluator = SearchEvaluator()
    
    # Load golden dataset
    with open("data/evaluation/golden_dataset.json") as f:
        dataset = json.load(f)
    
    # Evaluate each query with best method (hybrid_basic)
    method = evaluator.search_methods['hybrid_basic']
    search_func = method['search_func']
    
    query_scores = []
    
    for query_data in dataset:
        query = query_data['query']
        relevant_ids = set(query_data['expected_chunk_ids'])
        category = query_data['category']
        
        # Search
        results = search_func(query)
        retrieved_ids = [r['id'] for r in results]
        
        # Calculate recall
        recall = evaluator.metrics_calc.recall_at_k(retrieved_ids, relevant_ids, 5)
        
        query_scores.append({
            'query': query,
            'category': category,
            'recall@5': recall,
            'num_relevant': len(relevant_ids)
        })
    
    # Sort by recall (worst first)
    query_scores.sort(key=lambda x: x['recall@5'])
    
    # Show hardest queries
    print(f"\n{'='*100}")
    print("HARDEST QUERIES (Lowest Recall@5):")
    print('='*100)
    print(f"{'Query':<50} {'Category':<20} {'Recall@5':>12} {'Relevant':>10}")
    print("-" * 100)
    
    for q in query_scores[:10]:  # Bottom 10
        print(f"{q['query']:<50} {q['category']:<20} {q['recall@5']:>12.2%} {q['num_relevant']:>10}")
    
    # Show easiest queries
    print(f"\n{'='*100}")
    print("EASIEST QUERIES (Highest Recall@5):")
    print('='*100)
    print(f"{'Query':<50} {'Category':<20} {'Recall@5':>12} {'Relevant':>10}")
    print("-" * 100)
    
    for q in query_scores[-10:]:  # Top 10
        print(f"{q['query']:<50} {q['category']:<20} {q['recall@5']:>12.2%} {q['num_relevant']:>10}")
    
    return query_scores


def analyze_method_agreement():
    """Check how often different methods agree on top results"""
    
    print("\n" + "="*100)
    print("METHOD AGREEMENT ANALYSIS")
    print("="*100)
    
    evaluator = SearchEvaluator()
    
    # Load dataset
    with open("data/evaluation/golden_dataset.json") as f:
        dataset = json.load(f)
    
    # For each query, get top 3 from each method
    agreements = []
    
    for query_data in dataset[:5]:  # Sample 5 queries
        query = query_data['query']
        
        print(f"\n{'='*100}")
        print(f"Query: '{query}'")
        print('='*100)
        
        method_results = {}
        
        for method_name, method in evaluator.search_methods.items():
            results = method['search_func'](query)
            top_3_funcs = [r['metadata']['function'] for r in results[:3]]
            method_results[method_name] = top_3_funcs
            
            print(f"\n{method_name}:")
            for i, func in enumerate(top_3_funcs, 1):
                print(f"  {i}. {func}")
        
        # Calculate overlap
        print(f"\nOverlap analysis:")
        methods = list(method_results.keys())
        
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                set1 = set(method_results[method1])
                set2 = set(method_results[method2])
                overlap = len(set1 & set2)
                print(f"  {method1} ∩ {method2}: {overlap}/3 functions")


def analyze_recall_vs_num_relevant():
    """Check if recall correlates with number of relevant results"""
    
    print("\n" + "="*100)
    print("RECALL vs NUMBER OF RELEVANT RESULTS")
    print("="*100)
    
    evaluator = SearchEvaluator()
    
    with open("data/evaluation/golden_dataset.json") as f:
        dataset = json.load(f)
    
    # Group by number of relevant results
    by_num_relevant = defaultdict(list)
    
    method = evaluator.search_methods['hybrid_basic']
    search_func = method['search_func']
    
    for query_data in dataset:
        query = query_data['query']
        relevant_ids = set(query_data['expected_chunk_ids'])
        num_relevant = len(relevant_ids)
        
        results = search_func(query)
        retrieved_ids = [r['id'] for r in results]
        
        recall = evaluator.metrics_calc.recall_at_k(retrieved_ids, relevant_ids, 5)
        
        by_num_relevant[num_relevant].append(recall)
    
    # Calculate average recall for each group
    print(f"\n{'Num Relevant':<15} {'Queries':<10} {'Avg Recall@5':<15}")
    print("-" * 45)
    
    for num in sorted(by_num_relevant.keys()):
        recalls = by_num_relevant[num]
        avg_recall = sum(recalls) / len(recalls)
        print(f"{num:<15} {len(recalls):<10} {avg_recall:<15.2%}")
    
    print("\nInsight: Queries with more relevant results are typically harder")
    print("         (lower recall) because we return only top 5.")


def generate_insights_summary():
    """Generate key insights from all analyses"""
    
    print("\n" + "="*100)
    print("KEY INSIGHTS SUMMARY")
    print("="*100)
    
    insights = []
    
    # Run analyses and collect insights
    category_results = analyze_by_category()
    
    # Insight 1: Best method per category
    categories = list(next(iter(category_results.values())).keys())
    
    for category in categories:
        category_scores = {
            method: category_results[method][category]['recall@5']
            for method in category_results.keys()
        }
        best_method = max(category_scores.items(), key=lambda x: x[1])
        
        insights.append(
            f"For {category} queries, {best_method[0]} performs best "
            f"({best_method[1]:.1%} Recall@5)"
        )
    
    # Insight 2: Overall best
    overall_best = max(
        category_results.items(),
        key=lambda x: sum(cat['recall@5'] for cat in x[1].values())
    )
    insights.append(f"Overall best method: {overall_best[0]}")
    
    # Print insights
    print("\n📊 Key Findings:\n")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    return insights


if __name__ == "__main__":
    # Run all analyses
    
    print("\n" + "="*100)
    print("STARTING COMPREHENSIVE ANALYSIS")
    print("="*100)
    
    # 1. Category-wise comparison
    category_results = analyze_by_category()
    
    # 2. Difficult queries
    query_scores = analyze_difficult_queries()
    
    # 3. Method agreement
    analyze_method_agreement()
    
    # 4. Recall vs num relevant
    analyze_recall_vs_num_relevant()
    
    # 5. Generate insights
    insights = generate_insights_summary()
    
    # Save insights
    output_dir = Path("docs/week3_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "insights.txt", 'w') as f:
        f.write("KEY INSIGHTS FROM EVALUATION\n")
        f.write("="*80 + "\n\n")
        for i, insight in enumerate(insights, 1):
            f.write(f"{i}. {insight}\n")
    
    print(f"\n✓ Analysis complete. Insights saved to {output_dir}/insights.txt")