# visualize_results.py

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_charts():
    """Create comparison charts"""
    
    # Load data
    with open("docs/week2_evaluation/latency_results.json") as f:
        latency_data = json.load(f)
    
    # Chart 1: Latency comparison
    methods = ['vector_only', 'bm25_only', 'hybrid_basic', 
               'hybrid_rerank', 'hybrid_classified']
    
    categories = list(latency_data.keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(categories))
    width = 0.15
    
    for i, method in enumerate(methods):
        means = [latency_data[cat][method]['mean'] for cat in categories]
        ax.bar(x + i*width, means, width, label=method)
    
    ax.set_xlabel('Query Category')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Retrieval Method Latency by Query Category')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('docs/week2_evaluation/latency_comparison.png', dpi=300)
    print("✓ Saved latency_comparison.png")
    
    # Chart 2: Method comparison radar (if you have quality scores)
    # Add more charts as needed

if __name__ == "__main__":
    create_charts()