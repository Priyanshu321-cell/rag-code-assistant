import chromadb
from loguru import logger
from typing import List,Dict,Optional
from src.retrieval.embedder import Embedder
from pathlib import Path

class VectorStore:
    def __init__(self, embedder: Embedder, persist_directory: str = "data/vector_db", collection_name: str = "code_chunks"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(self.persist_directory)
        
        self.collection_name = collection_name
        self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)
        self.embedder = embedder
        
        logger.info("Initialized chroma db client")
        
    def add_chunks(self, chunks: List[Dict])->None:
        if not chunks:
            logger.warning("No chunks to add")
            return
        ids = []
        texts = []
        metadata = []
        for chunk in chunks:
            ids.append(chunk['id'])
            texts.append(chunk['text'])
            metadata.append(chunk['metadata'])
        
        embeddings = self.embedder.embed_batch(texts)
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadata,
            embeddings=embeddings.tolist()
        )
        logger.info("Chunks added to collections successfully")
        
    def search(self, query: str,n_results :int = 5, filters: Optional[Dict]=None)->List[Dict]:
        """Search for similar chunks"""
        embedded_query = self.embedder.embed(query)
        results = self.collection.query(
            query_embeddings=[embedded_query.tolist()],
            n_results=n_results,
            where=filters
        )
        
        final_results = []
        ids = results['ids'][0]
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        dists = results['distances'][0]
        
        for i in range(len(ids)):
            result = {
                'id': ids[i],
                'text': docs[i],
                'metadata': metas[i],
                'distance': dists[i]
            }
            
            final_results.append(result)

        logger.info("results against query returned successfully")
        return final_results
        
    def clear(self)->None:
        """Delete all items from collection"""
        count = self.collection.count()
        if count == 0:
            logger.info("collection already empty")
            return
            
        all_items = self.collection.get()
        if all_items['ids']:
            self.collection.delete(ids=all_items['ids'])
            logger.info(f"Cleared {count} items from collection")
        else:
            logger.info("No items to clear")
        
    def get_stats(self) -> Dict:
        """Get vector store statistics"""
        return {'collection_name':self.collection_name , 'total_chunks' : self.collection.count()
                , 'persist_directory' : self.persist_directory}
    
        
if __name__ =='__main__':
    from src.retrieval.embedder import Embedder
    
    # Initialize
    embedder = Embedder()
    store = VectorStore(embedder=embedder)
    
    # Clear any old data
    store.clear()
    
    # Test chunks
    test_chunks = [
        {
            'id': 'test_1',
            'text': 'async def authenticate_user(username, password):\n    """Verify user credentials"""\n    return check_password(username, password)',
            'metadata': {'file': 'auth.py', 'function': 'authenticate_user', 'is_async': True}
        },
        {
            'id': 'test_2',
            'text': 'def parse_json(data):\n    """Parse JSON string"""\n    return json.loads(data)',
            'metadata': {'file': 'utils.py', 'function': 'parse_json', 'is_async': False}
        },
        {
            'id': 'test_3',
            'text': 'async def verify_token(token):\n    """Check if token is valid"""\n    return jwt.decode(token)',
            'metadata': {'file': 'auth.py', 'function': 'verify_token', 'is_async': True}
        }
    ]
    
    # Add chunks
    print("Adding chunks...")
    store.add_chunks(test_chunks)
    
    # Get stats
    print(f"Stats: {store.get_stats()}")
    
    # Test search
    print("\n" + "="*50)
    print("Search: 'user authentication'")
    print("="*50)
    results = store.search("user authentication", n_results=3)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['metadata']['function']} (distance: {result['distance']:.4f})")
        print(f"   File: {result['metadata']['file']}")
        print(f"   Text preview: {result['text'][:100]}...")
    
    # Test filtering
    print("\n" + "="*50)
    print("Search: 'authentication' filtered to auth.py")
    print("="*50)
    results = store.search("authentication", n_results=2, filters={'file': 'auth.py'})
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['metadata']['function']}")
    