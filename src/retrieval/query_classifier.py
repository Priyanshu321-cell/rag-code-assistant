"""Query classification for intelligent routing"""

import re 
from typing import Dict , List
from enum import Enum
from loguru import logger

class QueryType(Enum):
    """Query type classifications"""
    SPECIFIC_TERM = "specific_term"
    SEMANTIC = "semantic"
    CONCEPT = "concept"
    CODE_PATTERN = "code_pattern"

class QueryClassifier:
    """Classifies search queries to route to optimal retrieval strategy."""
    
    def __init__(self):
        "Initialize classifier with pattern rules"
        
        self.semantic_indicators=[
            'how to', 'how do' ,'how can' ,'how should',
            'what is','what are','what does',
            'why does', 'why is', 'why should',
            'when to', 'when should' ,'where to',
            'where can', 'best way', 'explain', 'guide','tutorial'
        ]
        
        self.code_patterns = [
            'async', 'await', 'def', 'class',
            'decorator', 'context manager', 'generator',
            'lambda', 'comprehension', 'iterator'
        ]
        
        self.python_specifics = [
            'Exception', 'Error', 'Model', 'Router',
            'Request', 'Response', 'Handler', 'Middleware'
        ]
        
        logger.info("QueryClassifier initialized")
        
        
    def get_search_config(self, query_type: QueryType) -> Dict:
        """Get search config for query type"""
        
        configs = {
            QueryType.SPECIFIC_TERM:{
                'use_bm25' : True,
                'use_vector' : False,
                'bm25_weight' : 1.0,
                'vector_weight': 0.0,
                'use_expansion': False,
                'use_reranking' : False
            }, 
            
            QueryType.SEMANTIC: {
                'use_bm25' : True,
                'use_vector' : True,
                'bm25_weight' : 1.0,
                'vector_weight': 1.0,
                'use_expansion': False,
                'use_reranking' : True
            },
            
            QueryType.CONCEPT: {
                'use_bm25' : True,
                'use_vector' : True,
                'bm25_weight' : 1.0,
                'vector_weight': 1.0,
                'use_expansion': True,
                'use_reranking' : True
            }, 
            QueryType.CODE_PATTERN: {
                'use_bm25': True,
                'use_vector': True,
                'bm25_weight': 0.7,   
                'vector_weight': 1.0,
                'use_expansion': False,
                'use_reranking': True
            }
        }
        
        return configs[query_type]
        
    def classify(self ,query: str)-> QueryType:
        """Classify a query into one of the QueryType categories"""
        query = query.strip()
    
        logger.debug(f"Classifying query: '{query}'")
        
        # Priority order matters!
        
        # 1. Specific terms (highest priority - very distinctive)
        if self._is_specific_term(query):
            logger.debug(f"  → SPECIFIC_TERM")
            return QueryType.SPECIFIC_TERM
        
        # 2. Semantic queries (clear indicators)
        if self._is_semantic_query(query):
            logger.debug(f"  → SEMANTIC")
            return QueryType.SEMANTIC
        
        # 3. Code patterns (specific keywords)
        if self._is_code_pattern(query):
            logger.debug(f"  → CODE_PATTERN")
            return QueryType.CODE_PATTERN
        
        # 4. Default to concept (general terms)
        logger.debug(f"  → CONCEPT")
        return QueryType.CONCEPT
        
    def _is_specific_term(self, query: str)->bool:
        """Check if query is a specific term (class/function)"""
        words = query.split()
        if len(words) <= 2:
            if re.search(r'[a-z][A-Z]', query): # CamelCase
                return True
            
            if '_' in query and not ' ' in query: # snake_case
                return True
            
            for suffix in self.python_specifics: # technical
                if query.endswith(suffix):
                    return True
                
        return False
    
    def _is_semantic_query(self, query: str) -> bool:
        """Check if query is semantic / how to style"""
        
        query_lower = query.lower()
        
        for indicator in self.semantic_indicators:
            if indicator in query_lower:
                return True
            
        if len(query.split())>6:
            return True
        
        if '?' in query:
            return True
        
        return False
    
    def _is_code_pattern(self, query: str) -> bool:
        """check if query is about code patterns"""
        
        query_lower = query.lower()
        
        for pattern in self.code_patterns:
            if pattern in query_lower:
                return True
        
        return False
    
        