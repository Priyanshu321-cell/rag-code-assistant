from loguru import logger
from typing import List,Dict

class HybridSearch:
    def __init__(self, bm25_searcher, vector_store, k: int = 60):
        """Initialize hybrid search"""
        self.bm25 = bm25_searcher
        self.vector = vector_store
        self.k = k
        logger.info(f"Initialized HybridSearch with k = {k}")
        
    def search(self, query:str ,n_results: int = 10,bm25_weight: float=1.0,vector_weight:float =1.0):
        """Hybrid search using RRF"""
        logger.debug(f"Hybrid search for : '{query}'")
        retrieve_k = n_results *2
        
        bm25_results = self.bm25.search(query, top_k=retrieve_k)
        vector_results = self.vector.search(query, n_results=retrieve_k)
        
        logger.debug(f"BM25 returned {len(bm25_results)} results")
        logger.debug(f"Vector returned {len(vector_results)} results")
        
        merged = self._merge_results(
            bm25_results,
            vector_results,
            bm25_weight,
            vector_weight,
        )
        
        final = merged[:n_results]
        
        logger.info(f"Hybrid search returned {len(final)} results")
        return final
        
    def _rrf_score(self, rank:int)->float:
        """Calculate RRF score for a given rank"""
        return 1.0/(rank+self.k)
    
    def _merge_results(
        self,bm25_results:List[Dict],
        vector_results:List[Dict],
        bm25_weight: float,
        vector_weight:float
    )->List[Dict]:
        """Merge results using RRf"""
        scores = {}
        
        for rank, result in enumerate(bm25_results):
            chunk_id = result['id']
            rrf = self._rrf_score(rank) * bm25_weight
            
            if chunk_id not in scores:
                scores[chunk_id] = {
                    'rrf_score':0,
                    'bm25_rank':rank,
                    'vector_rank':None,
                    'chunk':result
                }
            scores[chunk_id]['rrf_score'] += rrf
            scores[chunk_id]['bm25_rank'] = rank
            
        for rank , result in enumerate(vector_results):
            chunk_id = result['id']
            rrf = self._rrf_score(rank) * vector_weight
            
            if chunk_id not in scores:
                scores[chunk_id] = {
                    'rrf_score':0,
                    'bm25_rank':None,
                    'vector_rank':rank,
                    'chunk':result
                }
            scores[chunk_id]['rrf_score'] += rrf
            scores[chunk_id]['vector_rank'] = rank
            
        merged = []
        for chunk_id, data in scores.items():
            result = data['chunk'].copy()
            result['rrf_score'] = data['rrf_score']
            result['bm25_rank'] = data['bm25_rank']
            result['vector_rank'] = data['vector_rank']
            merged.append(result)
            
        merged.sort(key=lambda x : x['rrf_score'], reverse=True)
        
        return merged
            

    
        