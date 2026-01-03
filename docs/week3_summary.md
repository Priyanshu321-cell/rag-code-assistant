# Week 3 Evaluation Results

## Methods Tested

Evaluated 5 retrieval strategies on 34 test queries with 139 relevant results.

## Key Findings

### Performance Results

| Method | Recall@5 | MRR | Latency | Notes |
|--------|----------|-----|---------|-------|
| Vector only | 47.69% | 0.756 | 45ms | Baseline |
| BM25 only | 33.94% | 0.592 | 12ms | Weak on semantic queries |
| **Hybrid basic** | **51.59%** | **0.814** | **98ms** | **Production choice** |
| Hybrid + rerank | 53.93% | 0.741 | 278ms | +2.3% recall, +180ms |
| Hybrid + classify | 49.11% | 0.670 | Variable | Classification needs tuning |

## Decisions

### ✅ Included: Hybrid Search (RRF)
- **Impact:** +8% Recall@5 over baseline
- **Cost:** 2x latency (acceptable)
- **Verdict:** Clear win, production-ready

### ❌ Excluded: Cross-Encoder Reranking
- **Impact:** +2.3% Recall@5 over hybrid
- **Cost:** +180ms latency (3x slower)
- **Verdict:** Diminishing returns, not worth latency cost

### ❌ Excluded: Query Classification
- **Impact:** -2.5% Recall@5 
- **Cost:** Minimal
- **Verdict:** Current rules don't match test set, needs refinement

## Insights

1. **Hybrid search is the sweet spot:** Combining BM25 and vector provides clear benefits
2. **Advanced features have diminishing returns:** Reranking helps but not proportionally to cost
3. **Simple is often better:** Best system is hybrid-basic, not the most complex configuration

## Production Configuration
```python
# Recommended production setup
hybrid = HybridSearch(
    bm25_searcher=bm25,
    vector_store=vector_store,
    # No reranker - cost/benefit not favorable
    # No classifier - needs tuning first
)
results = hybrid.search(query, n_results=10)
```

Performance: **51.59% Recall@5** at **98ms p50 latency**