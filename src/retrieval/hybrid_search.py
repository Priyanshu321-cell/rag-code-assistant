from loguru import logger
from typing import List,Dict

class HybridSearch:
    def __init__(self, bm25_searcher, vector_store, reranker = None,query_expander=None, query_classifier = None, k: int = 30):
        """Initialize hybrid search"""
        self.bm25 = bm25_searcher
        self.vector = vector_store
        self.k = k
        self.reranker = reranker
        self.query_expander = query_expander
        self.query_classifier = query_classifier
        logger.info(f"Initialized HybridSearch with k={k}, reranker={reranker is not None}")
        
    def search(self, query:str ,n_results: int = 10,use_classifier: bool= True, use_reranker = False):
        """search"""
        
        
        if use_classifier and self.query_classifier:
            query_type = self.query_classifier.classify(query)
            config = self.query_classifier.get_search_config(query_type)
    
            logger.info(f"Query type: {query_type.value}, Config: {config}")
            
        else :
            config = {
                'use_bm25': True,
                'use_vector': True,
                'bm25_weight': 1.0,
                'vector_weight': 1.0,
                'use_expansion': True,
                'use_reranking': True
            }
            
        if config['use_expansion'] and self.query_expander:
            queries = self.query_expander.expand(query)
        else:
            queries = [query]
        
        all_results = {}
        for q in queries:
            retrieve_k = n_results * 2 if config['use_reranking'] else n_results
            
            # Get BM25 results if enabled
            if config['use_bm25']:
                bm25_results = self.bm25.search(q, top_k=retrieve_k)
            else:
                bm25_results = []
            
            # Get vector results if enabled
            if config['use_vector']:
                vector_results = self.vector.search(q, n_results=retrieve_k)
            else:
                vector_results = []
                
            merged = self._merge_results(
                bm25_results, 
                vector_results,
                config['bm25_weight'],
                config['vector_weight']
            )
            
            for result in merged:
                chunk_id = result['id']
                if chunk_id not in all_results:
                    all_results[chunk_id] = result
                elif result['rrf_score'] > all_results[chunk_id]['rrf_score']:
                    all_results[chunk_id] = result
                    
        final = list(all_results.values())
        final.sort(key=lambda x: x['rrf_score'], reverse=True)
            
        if config['use_reranking'] and self.reranker and len(final) > 0:
            final = self.reranker.rerank(query, final, top_k=n_results)
        else:
            final = final[:n_results]
            
        return final
        
    def _rrf_score(self, rank:int)->float:
        """Calculate RRF score for a given rank - optimized k=30 for better merging"""
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
            

    
        