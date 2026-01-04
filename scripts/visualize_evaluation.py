# visualize_evaluation.py

"""
Create visualizations of evaluation results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_method_comparison():
    """Bar chart comparing all methods"""
    
    # Load results
    with open("data/evaluation/results.json") as f:
        results = json.load(f)
    
    methods = list(results.keys())
    metrics = ['recall@5', 'precision@5', 'mrr', 'ndcg@5']
    
    # Create subplot for each metric
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Retrieval Method Comparison', fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        scores = [results[method][metric] for method in methods]
        
        bars = ax.bar(range(len(methods)), scores, color='steelblue', alpha=0.8)
        
        # Highlight best method
        best_idx = scores.index(max(scores))
        bars[best_idx].set_color('darkgreen')
        bars[best_idx].set_alpha(1.0)
        
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace('_', '\n') for m in methods], 
                           rotation=0, fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_path = Path("docs/week3_analysis/method_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    
    plt.close()


def plot_category_heatmap():
    """Heatmap showing method performance by category"""
    
    # You'll need category-wise results
    # For now, create sample data structure
    
    print("ℹ️  Category heatmap requires category-wise evaluation data")
    print("   Run evaluator.compare_all_methods_by_category() first")


def plot_recall_at_k():
    """Line plot showing Recall@K for different K values"""
    
    with open("data/evaluation/results.json") as f:
        results = json.load(f)
    
    methods = list(results.keys())
    k_values = [1, 3, 5, 10]
    
    plt.figure(figsize=(10, 6))
    
    for method in methods:
        recalls = [results[method][f'recall@{k}'] for k in k_values]
        plt.plot(k_values, recalls, marker='o', label=method, linewidth=2)
    
    plt.xlabel('K (Number of Results)', fontsize=12)
    plt.ylabel('Recall@K', fontsize=12)
    plt.title('Recall@K Comparison Across Methods', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    plt.ylim(0, 1.0)
    
    output_path = Path("docs/week3_analysis/recall_at_k.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    
    plt.close()

def plot_category_breakdown():
    """Show best method per category"""
    
    categories = ['specific_term', 'how_to', 'concept', 'code_pattern']
    best_methods = ['vector_only', 'hybrid_basic', 'hybrid_rerank', 'hybrid_rerank']
    scores = [0.528, 0.578, 0.542, 0.648]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(categories, scores, color=colors, alpha=0.8)
    
    ax.set_ylabel('Recall@5', fontsize=12)
    ax.set_title('Best Method Performance by Query Category', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 0.8)
    ax.grid(axis='y', alpha=0.3)
    
    # Add labels
    for bar, method, score in zip(bars, best_methods, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{method}\n{score:.1%}',
               ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('docs/week3_analysis/category_breakdown.png', dpi=300)
    print("✓ Saved category_breakdown.png")


def create_summary_table():
    """Create a nice summary table image"""
    
    with open("data/evaluation/results.json") as f:
        results = json.load(f)
    
    methods = list(results.keys())
    metrics = ['recall@1', 'recall@5', 'precision@5', 'mrr', 'ndcg@5']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for method in methods:
        row = [method] + [f"{results[method][m]:.4f}" for m in metrics]
        table_data.append(row)
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=['Method'] + [m.replace('@', '\n@') for m in metrics],
        cellLoc='center',
        loc='center',
        colWidths=[0.25] + [0.15]*len(metrics)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(metrics) + 1):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best scores in each column
    for col_idx, metric in enumerate(metrics, 1):
        scores = [float(results[method][metric]) for method in methods]
        best_idx = scores.index(max(scores))
        table[(best_idx + 1, col_idx)].set_facecolor('#90EE90')
    
    plt.title('Evaluation Results Summary', fontsize=14, fontweight='bold', pad=20)
    
    output_path = Path("docs/week3_analysis/results_table.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    
    plt.close()


if __name__ == "__main__":
    print("Generating visualizations...")
    
    plot_method_comparison()
    plot_recall_at_k()
    create_summary_table()
    
    print("\n✓ All visualizations created in docs/week3_analysis/")