from sentence_transformers import CrossEncoder
from typing import List,Dict
from loguru import logger

class Reranker:
    """Cross encoder based reranker"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize reranker"""
        self.model_name = model_name
        logger.info(f"Loading cross encoder model: {self.model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("Reranker ready")
    
    def rerank(
            self,
            query:str,
            results:List[Dict],
            top_k: int | None = None
        )-> List[Dict]:
            """Rerank results using cross encoder"""
            if results == [] :
                logger.warning("No results found to reranked")
                return []
            logger.debug(f"Reranking {len(results)} results for query: '{query}'")
            pairs = self._create_pairs(query,results)
            scores = self.model.predict(pairs)
            for i , result in enumerate(results):
                result['rerank_score'] = float(scores[i])
                
            reranked= sorted(results, key=lambda x : x['rerank_score'], reverse=True)
            
            if top_k:
                reranked = reranked[:top_k]
                
            logger.debug(f"Reranking complete, returning {len(reranked)} results")
            return reranked
        
    def _create_pairs(self, query: str, results: List[Dict]) -> List[List[str]]:
            """Create pairs with enhanced context for better accuracy"""
            
            pairs = []
            for result in results:
                metadata = result['metadata']
                
                # Build rich context string
                context_parts = []
                
                # Function name (most important)
                if 'function' in metadata:
                    context_parts.append(metadata['function'])
                
                # File path for context
                if 'file' in metadata:
                    context_parts.append(f"file: {metadata['file']}")
                
                # Function signature if available
                if 'signature' in metadata:
                    context_parts.append(f"sig: {metadata['signature']}")
                
                # Docstring excerpt for semantic context
                if 'docstring' in metadata and metadata['docstring']:
                    docstring = metadata['docstring'][:100]  # Limit length
                    context_parts.append(f"doc: {docstring}")
                
                # Class context if available
                if 'class' in metadata:
                    context_parts.append(f"class: {metadata['class']}")
                
                # Join all context parts
                context = " | ".join(context_parts)
                pairs.append([query, context])
            
            return pairs
                    
                
if __name__ == "__main__":
    
    pass   
        
        