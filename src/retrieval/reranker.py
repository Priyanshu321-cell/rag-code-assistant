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
        top_k: int = None
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
        """Create pairs - SIMPLE VERSION"""
        
        pairs = []
        for result in results:
            function_name = result['metadata']['function']
            file = result['metadata']['file']
            pairs.append([query, f"{function_name} - {file}"])
        
        return pairs
                
                
if __name__ == "__main__":
    
    pass   
        
        