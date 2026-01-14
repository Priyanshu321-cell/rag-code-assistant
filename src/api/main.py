from src.generation.rag_pipeline import RAGPipeline
from src.generation.conversation import ConversationManager
from src.retrieval.vector_store import VectorStore
from src.retrieval.embedder import Embedder
from fastapi import FastAPI

app = FastAPI()
rag_pipeline = RAGPipeline()
conversation = ConversationManager()
store = VectorStore(embedder=Embedder())

@app.get("/")
async def root():
    return {"message" : "Hello world"}

@app.post("/query/{query}")
async def query_search(query:str = "Hello ?"):
    return rag_pipeline.query(question=query)

@app.post("/chat")
async def query_chat():
    pass

@app.get("/health")
async def get_health():
    return store.get_stats()

@app.post("/feedback")
async def feedback():
    pass

@app.get("/stats")
async def stats():
    return store.get_stats()