# Week 3 Evaluation: Key Findings

## Surprising Insights

### 1. Vector Search Excels at Specific Terms
Contrary to expectations, vector-only achieved **52.8% Recall@5** on specific term queries like "APIRouter", outperforming BM25 (30.8%). This suggests the embedding model has learned technical terminology well.

### 2. Reranking Hurts Simple Queries
On clear how-to queries, hybrid basic (57.8%) outperformed hybrid+rerank (47.2%) by 10 percentage points. Reranking appears to reshuffle already-good rankings for explicit queries.

### 3. Reranking Shines on Complexity
For code pattern queries, reranking provided a **15% improvement** (64.8% vs 49.6%), demonstrating its value for ambiguous, complex queries.

## Performance by Query Type

| Query Type | Best Method | Recall@5 | Notes |
|------------|-------------|----------|-------|
| Specific Term | Vector only | 52.8% | Embeddings capture technical terms well |
| How-To | Hybrid basic | 57.8% | Clear queries don't need reranking |
| Concept | Hybrid + rerank | 54.2% | Ambiguity benefits from reranking |
| Code Pattern | Hybrid + rerank | 64.8% | Complex patterns need deep understanding |

## Production Strategy: Adaptive Routing

Based on these findings, implemented query-based routing:
```python
if is_how_to_query(query):
    return hybrid_basic(query)  # Fast, effective
elif is_complex_query(query):
    return hybrid_rerank(query)  # Quality-focused
else:
    return hybrid_basic(query)  # Safe default
```

**Result:** 56% average recall at 140ms average latency (vs 53.9% at 280ms for always-rerank)

## Key Learnings

1. **No one-size-fits-all:** Different query types need different strategies
2. **Complexity matters:** Simple queries need speed, complex need quality
3. **Embeddings are powerful:** Modern models handle technical terms surprisingly well
4. **Reranking is expensive:** Use selectively where it provides clear value
5. **Measure everything:** Evaluation revealed counterintuitive patterns