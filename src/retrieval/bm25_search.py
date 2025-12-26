from rank_bm25 import BM25Okapi
from typing import List, Dict
from loguru import logger

class BM25Search:
    """BM25 keyword search for code chunks"""
    
    def __init__(self):
        """Initialize BM25 search"""
        self.bm25 = None  
        self.chunks = []   
        self.chunk_ids = []
    
    def index_documents(self,chunks: List[Dict])->None:
        """Build BM25 index from chunks"""
        if not chunks:
            logger.warning("No chunks to index")
            return
        
        logger.info(f"Building BM25 index for {len(chunks)} chunks...")
        self.chunks = chunks
        self.chunk_ids = [chunk['id'] for chunk in chunks]
        
        texts = [chunk['text'] for chunk in chunks]
        tokenized_docs = [self._tokenize(text) for text in texts]
        
        self.bm25 = BM25Okapi(tokenized_docs)
        
        logger.info(f"BM25 index built with {len(self.chunks)} documents")
        
    def _tokenize(self, text:str) -> List[str]:
        """simple tokenization"""
        text = text.lower()
        
        import re
        tokens = re.findall(r'\b\w+\b', text)
        
        tokens = [ t for t in tokens if len(t) >= 2]
        return tokens
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search with BM25"""
        
        if not hasattr(self, 'bm25'):
            logger.error("Index not built! Call index_documents first")
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    'id': self.chunk_ids[idx],
                    'text': self.chunks[idx]['text'],
                    'metadata':self.chunks[idx]['metadata'],
                    'score':float(scores[idx])
                })
        logger.debug(f"BM25 search for '{query}' returned {len(results)} results")
        return results
    
if __name__ == '__main__':
    pass