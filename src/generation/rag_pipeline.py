# src/generation/rag_pipeline.py

"""
Complete RAG pipeline: search + generation.
"""

import pickle
from pathlib import Path
from typing import Dict, List
from loguru import logger
import time
import sys
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_search import BM25Search
from src.retrieval.adaptive_search import AdaptiveSearch
from src.retrieval.reranker import Reranker
from src.generation.rag_generator import RAGGenerator


class RAGPipeline:
    """
    End-to-end RAG pipeline: query → retrieve → generate.
    """
    
    def __init__(self):
        """Initialize all components"""
        
        logger.info("Initializing RAG pipeline...")
        
        # Retrieval components
        self.embedder = Embedder()
        self.vector_store = VectorStore(self.embedder)
        
        with open("data/processed/bm25_index.pkl", 'rb') as f:
            self.bm25 = pickle.load(f)
        
        self.reranker = Reranker()
        
        # Search system (adaptive)
        self.search = AdaptiveSearch(
            bm25_searcher=self.bm25,
            vector_store=self.vector_store,
            reranker=self.reranker
        )
        
        # Generation
        self.generator = RAGGenerator()
        
        logger.info("RAG pipeline ready")
        
    

    def query_stream(
        self,
        question: str,
        n_results: int = 5,
        include_code: bool = True
    ):
        """Stream with progress indicators"""
    
        yield {'type': 'query', 'content': question}
        
        # Show spinner during retrieval
        yield {'type': 'status', 'content': 'Searching', 'spinner': True}
        
        start = time.time()
        retrieved_chunks = self.search.search(question, n_results=n_results)
        elapsed = time.time() - start
        
        yield {
            'type': 'status',
            'content': f'Found {len(retrieved_chunks)} functions in {elapsed:.1f}s',
            'spinner': False
        }
        
        logger.info(f"RAG streaming query: '{question}'")
        
        # Yield query info
        yield {
            'type': 'query',
            'content': question
        }
        
        # Step 1: Retrieve (blocking, but fast)
        logger.debug("Retrieving chunks...")
        
        yield {
            'type': 'status',
            'content': 'Searching codebase...'
        }
        
        retrieved_chunks = self.search.search(question, n_results=n_results)
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
        
        yield {
            'type': 'status',
            'content': f'Found {len(retrieved_chunks)} relevant functions'
        }
        
        # Step 2: Stream answer generation
        logger.debug("Streaming answer...")
        
        for event in self.generator.generate_answer_stream(
            query=question,
            retrieved_chunks=retrieved_chunks,
            include_code=include_code
        ):
            yield event


    def query_stream_display(self, question: str):
        """
        Stream query with live display in terminal.
        """
        
        print("\n" + "="*80)
        print(f"Question: {question}")
        print("="*80)
        
        answer_parts = []
        sources = None
        
        for event in self.query_stream(question):
            
            if event['type'] == 'status':
                print(f"\n📡 {event['content']}")
            
            elif event['type'] == 'sources':
                sources = event['content']
                print(f"\n📚 Using {len(sources)} sources:")
                for i, source in enumerate(sources, 1):
                    print(f"   {i}. {source['function']}() - {source['file']}")
                print("\n💬 Answer:")
                print("-" * 80)
            
            elif event['type'] == 'answer_chunk':
                # Print chunk without newline
                print(event['content'], end='', flush=True)
                answer_parts.append(event['content'])
            
            elif event['type'] == 'answer':
                # Non-streaming fallback
                print(event['content'])
                answer_parts.append(event['content'])
            
            elif event['type'] == 'done':
                print("\n" + "="*80)
            
            elif event['type'] == 'error':
                print(f"\n❌ Error: {event['content']}")
        
        return {
            'question': question,
            'answer': ''.join(answer_parts),
            'sources': sources
        }
    
    
    def query(
        self,
        question: str,
        n_results: int = 5,
        include_code: bool = True
    ) -> Dict:
        """
        Complete RAG query: retrieve and generate answer.
        
        Args:
            question: User's question
            n_results: Number of chunks to retrieve
            include_code: Include code in answer
            
        Returns:
            Dict with 'question', 'answer', 'sources', 'retrieved_chunks'
        """
        
        logger.info(f"RAG query: '{question}'")
        
        # Step 1: Retrieve relevant chunks
        logger.debug("Retrieving relevant chunks...")
        retrieved_chunks = self.search.search(question, n_results=n_results)
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
        
        # Step 2: Generate answer
        logger.debug("Generating answer...")
        generation_result = self.generator.generate_answer(
            query=question,
            retrieved_chunks=retrieved_chunks,
            include_code=include_code
        )
        
        # Combine results
        result = {
            'question': question,
            'answer': generation_result['answer'],
            'sources': generation_result['sources'],
            'retrieved_chunks': retrieved_chunks,
            'num_chunks_used': len(retrieved_chunks)
        }
        
        logger.info("RAG query complete")
        return result
    
    
    def query_with_display(self, question: str):
        """Query and display results in formatted way"""
        
        result = self.query(question)
        
        print("\n" + "="*80)
        print(f"Question: {result['question']}")
        print("="*80)
        
        print(f"\nAnswer:\n{result['answer']}")
        
        print(f"\n{'='*80}")
        print(f"Sources ({len(result['sources'])} chunks used):")
        print('-'*80)
        
        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. {source['function']}()")
            print(f"   File: {source['file']}")
            if source['line']:
                print(f"   Line: {source['line']}")
        
        print("="*80)
        
        return result


if __name__ == "__main__":
    # Test complete RAG pipeline
    
    pipeline = RAGPipeline()
    
    # Test queries
    test_queries = [
        "How do I create an API endpoint in FastAPI?",
        "What is APIRouter?",
        "How do I handle authentication?",
    ]
    
    for query in test_queries:
        pipeline.query_with_display(query)
        print("\n")