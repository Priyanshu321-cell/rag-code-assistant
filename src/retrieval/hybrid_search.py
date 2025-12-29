from loguru import logger
from typing import List,Dict

class HybridSearch:
    def __init__(self, bm25_searcher, vector_store, reranker = None,query_expander=None, k: int = 60):
        """Initialize hybrid search"""
        self.bm25 = bm25_searcher
        self.vector = vector_store
        self.k = k
        self.reranker = reranker
        self.query_expander = query_expander
        logger.info(f"Initialized HybridSearch with k={k}, reranker={reranker is not None}")
        
    def search(self, query:str ,n_results: int = 10,bm25_weight: float=1.0,vector_weight:float =1.0, use_reranker:bool = True, expand_query: bool=True):
        """Hybrid search using RRF"""
        import time
        start = time.time()
        
        # Expand query if enabled 
        if expand_query and self.query_expander:
            queries = self.query_expander.expand(query)
            logger.debug(f"Expanded into {len(queries)} queries: {queries}")
        else :
            queries = [query]
            
        all_results = {}
        for query in queries:
            if use_reranker and self.reranker:
                retrieve_k = n_results * 3  
            else:
                retrieve_k = n_results
            
            bm25_results = self.bm25.search(query, top_k=retrieve_k)
            vector_results = self.vector.search(query, n_results=retrieve_k)  
            
            merged = self._merge_results(
                bm25_results,
                vector_results,
                bm25_weight,
                vector_weight,
            )
            
            for result in merged:
                chunk_id = result['id']
                if chunk_id not in all_results:
                    all_results[chunk_id] = result
                else:
                    if result['rrf_score'] > all_results[chunk_id]['rrf_score']:
                        all_results[chunk_id] = result
        
        merged = list(all_results.values())
        merged.sort(key=lambda x: x['rrf_score'], reverse=True)
        if use_reranker and self.reranker and len(merged) > 0:
            logger.debug("Applying cross-encoder reranking")
            merged = self.reranker.rerank(query, merged, top_k=n_results)
        else:
            merged = merged[:n_results]
        
        
        logger.info(f"Hybrid search returned {len(merged)} results")
        logger.debug(f"Search completed in {(time.time()-start)*1000:.0f}ms")
        return merged
        
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
            else:
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
            

    
        