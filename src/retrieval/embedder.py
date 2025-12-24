from sentence_transformers import SentenceTransformer
from loguru import logger
import numpy as np
from typing import List, Union

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize model"""
        self.model_name = model_name
        logger.info(f"Loading embedding model: {self.model_name}")
        
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Model loaded. Embedding dimenaion: {self.embedding_dim}")
        
    def embed(self, texts:Union[str, List[str]])->np.ndarray:
        """Embed single text or list of texts"""
        
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else: 
            single_input = False
            
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        if single_input:
            return embeddings[0]
        
        return embeddings
        
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
    
if __name__ == "__main__":
    from pathlib import Path
    from src.ingestion.parser import parse_directory, format_function_for_embedding
    
    functions = parse_directory(Path("data/raw/fastapi/fastapi"))
    texts = [format_function_for_embedding(f) for f in functions]
    
    embedder = Embedder()
    embeddings = embedder.embed_batch(texts)
    print(f"Embeddings shape: {embeddings.shape}")
    