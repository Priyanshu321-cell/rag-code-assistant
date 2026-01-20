from sentence_transformers import SentenceTransformer
from loguru import logger
import numpy as np
from typing import List, Union
import hashlib

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2", cache_size: int = 1000):
        """Initialize model with caching"""
        self.model_name = model_name
        logger.info(f"Loading embedding model: {self.model_name}")
        
        self.model = SentenceTransformer(self.model_name)
        dim = self.model.get_sentence_embedding_dimension()
        self.embedding_dim = dim if dim is not None else 384  # Default for MiniLM
        
        # Initialize cache for query embeddings
        self.cache = {}
        self.cache_size = cache_size
        
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}, Cache size: {cache_size}")
        
    def embed(self, texts:Union[str, List[str]])->np.ndarray:
        """Embed single text or list of texts with caching"""
        
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else: 
            single_input = False
        
        # Check cache for each text
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            text_hash = self._get_text_hash(text)
            if text_hash in self.cache:
                cached_embeddings.append((i, self.cache[text_hash]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Embed uncached texts
        if uncached_texts:
            new_embeddings = self.model.encode(uncached_texts, convert_to_numpy=True)
            
            # Update cache
            for text, embedding in zip(uncached_texts, new_embeddings):
                text_hash = self._get_text_hash(text)
                self._update_cache(text_hash, embedding)
        else:
            new_embeddings = np.array([])
        
        # Combine cached and new embeddings
        if single_input:
            if cached_embeddings:
                return cached_embeddings[0][1]
            else:
                return new_embeddings[0]
        
        # Reconstruct full embeddings list
        if len(cached_embeddings) > 0 or len(new_embeddings) > 0:
            all_embeddings = np.zeros((len(texts), int(self.embedding_dim)))
            for i, embedding in cached_embeddings:
                all_embeddings[i] = embedding
            for i, embedding in zip(uncached_indices, new_embeddings):
                all_embeddings[i] = embedding
            return all_embeddings
        else:
            return np.array([]).reshape(0, int(self.embedding_dim))
        
    def embed_batch(self, text_list:List[str] ,batch_size:int = 32)->np.ndarray:
        """To embed large batches"""
        embeddings = self.model.encode(
            text_list,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        logger.info(f"Embedded {len(text_list)} texts")
        return embeddings
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text caching"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _update_cache(self, text_hash: str, embedding: np.ndarray):
        """Update cache with LRU eviction if needed"""
        if len(self.cache) >= self.cache_size:
            # Simple LRU: remove first item (could be improved)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[text_hash] = embedding
    
if __name__ == "__main__":
    from pathlib import Path
    from src.ingestion.parser import parse_directory, format_function_for_embedding
    
    functions = parse_directory(Path("data/raw/fastapi/fastapi"))
    texts = [format_function_for_embedding(f) for f in functions]
    
    embedder = Embedder()
    embeddings = embedder.embed_batch(texts)
    print(f"Embeddings shape: {embeddings.shape}")
    