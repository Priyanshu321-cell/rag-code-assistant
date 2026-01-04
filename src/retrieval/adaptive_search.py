"""
Adaptive search system that routes queries to optimal retrieval strategy.
Based on Week 3 evaluation findings.
"""

from typing import List, Dict
from loguru import logger

from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_search import BM25Search
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import Reranker


class AdaptiveSearch:
    """
    Intelligent search router that selects optimal strategy per query type.
    
    Strategy selection based on evaluation:
    - How-to queries (57.8% recall): hybrid_basic
    - Specific terms (52.8% recall): vector_only
    - Complex queries (64.8% recall): hybrid_rerank
    - Default: hybrid_basic
    """
    
    def __init__(
        self,
        bm25_searcher: BM25Search,
        vector_store: VectorStore,
        reranker: Reranker = None
    ):
        """
        Initialize adaptive search.
        
        Args:
            bm25_searcher: BM25 search instance
            vector_store: Vector store instance
            reranker: Optional reranker for complex queries
        """
        self.bm25 = bm25_searcher
        self.vector = vector_store
        self.reranker = reranker
        
        # Initialize hybrid search for routes that need it
        self.hybrid = HybridSearch(
            bm25_searcher=bm25_searcher,
            vector_store=vector_store,
            reranker=reranker
        )
        
        logger.info("AdaptiveSearch initialized")
    
    
    def search(self, query: str, n_results: int = 10) -> List[Dict]:
        """
        Search with adaptive routing.
        
        Args:
            query: Search query
            n_results: Number of results
            
        Returns:
            Search results from optimal strategy
        """
        
        query_lower = query.lower().strip()
        
        # Determine route
        route = self._determine_route(query_lower)
        
        # Execute search based on route
        if route == 'how_to':
            logger.info(f"Route: how_to → hybrid_basic")
            return self.hybrid.search(query, n_results, use_reranker=False)
        
        elif route == 'specific_term':
            logger.info(f"Route: specific_term → vector_only")
            return self.vector.search(query, n_results=n_results)
        
        elif route == 'complex':
            logger.info(f"Route: complex → hybrid_rerank")
            if self.reranker:
                return self.hybrid.search(query, n_results, use_reranker=True)
            else:
                # Fallback if no reranker
                return self.hybrid.search(query, n_results, use_reranker=False)
        
        else:  # default
            logger.info(f"Route: default → hybrid_basic")
            return self.hybrid.search(query, n_results, use_reranker=False)
    
    
    def _determine_route(self, query: str) -> str:
        """
        Determine optimal route for query.
        
        Returns:
            Route name: 'how_to', 'specific_term', 'complex', or 'default'
        """
        
        # Check how-to (highest priority - clear indicator)
        if self._is_how_to_query(query):
            return 'how_to'
        
        # Check specific term
        if self._is_single_term(query):
            return 'specific_term'
        
        # Check complex
        if self._is_complex_query(query):
            return 'complex'
        
        return 'default'
    
    
    def _is_how_to_query(self, query: str) -> bool:
        """Check if query is a how-to question"""
        how_to_indicators = [
            'how to', 'how do', 'how can', 'how should',
            'how would', 'how does', 'how will'
        ]
        return any(indicator in query for indicator in how_to_indicators)
    
    
    def _is_single_term(self, query: str) -> bool:
        """Check if query is a single specific term"""
        words = query.split()
        
        if len(words) == 1:
            return True
        
        # Two words with common prefix
        if len(words) == 2:
            common_prefixes = {'api', 'http', 'web', 'background', 'file'}
            if any(w.lower() in common_prefixes for w in words):
                return True
        
        return False
    
    
    def _is_complex_query(self, query: str) -> bool:
        """Check if query is complex (needs reranking)"""
        
        # Code pattern keywords
        code_patterns = [
            'async', 'await', 'decorator', 'pattern',
            'context manager', 'generator', 'function',
            'handler', 'middleware', 'lifecycle',
            'injection', 'validation'
        ]
        
        if any(pattern in query for pattern in code_patterns):
            return True
        
        # Multi-word queries (concepts)
        words = query.split()
        if len(words) >= 2 and len(words) <= 4:
            return True
        
        return False


if __name__ == "__main__":
    # Test adaptive routing
    
    test_queries = [
        ("how to authenticate users", "how_to"),
        ("APIRouter", "specific_term"),
        ("async endpoint function", "complex"),
        ("routing", "default"),
        ("dependency injection pattern", "complex"),
        ("Request", "specific_term"),
        ("how to add middleware", "how_to"),
    ]
    
    # Mock initialization (for testing routing logic only)
    class MockSearcher:
        def search(self, q, **kwargs):
            return []
    
    adaptive = AdaptiveSearch(MockSearcher(), MockSearcher())
    
    print("ADAPTIVE ROUTING TESTS")
    print("="*60)
    
    for query, expected_route in test_queries:
        actual_route = adaptive._determine_route(query.lower())
        match = "✓" if actual_route == expected_route else "✗"
        print(f"{match} '{query}'")
        print(f"   Expected: {expected_route}, Got: {actual_route}")