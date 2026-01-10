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
from src.generation.conversation import ConversationManager

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
        
        self.conversation = ConversationManager(max_history=10)
    
        logger.info("RAG pipeline ready with conversation support")
        
    def query_conversational(
        self,
        query: str,
        n_results: int = 5,
        resolve_references: bool = True
    ) -> Dict:
        """
        Query with conversation context.
        
        Args:
            query: User's query (may be follow-up)
            n_results: Number of chunks
            resolve_references: Whether to resolve follow-up queries
            
        Returns:
            Result dict with conversation context
        """
        
        logger.info(f"Conversational query: '{query}'")
        
        # Detect and resolve follow-ups
        is_followup = self.conversation.is_follow_up(query)
        
        if is_followup and resolve_references:
            logger.info("Detected follow-up question, resolving references...")
            original_query = query
            query = self.conversation.resolve_query(query, use_llm=True)
            logger.info(f"Resolved: '{original_query}' → '{query}'")
        else:
            original_query = query
        
        # Standard RAG query
        result = self.query(query, n_results=n_results)
        
        # Add to conversation history
        self.conversation.add_turn(
            query=original_query,
            answer=result['answer'],
            sources=result['sources']
        )
        
        # Add conversation metadata
        result['original_query'] = original_query
        result['resolved_query'] = query if is_followup else None
        result['is_followup'] = is_followup
        result['conversation_stats'] = self.conversation.get_stats()
        
        return result


    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation.clear()
        logger.info("Conversation cleared")
    
    
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
        include_code: bool = True,
        timeout: int = 30
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
        
        # Validate question
        if not question or not question.strip():
            logger.warning("Empty question provided")
            return {
                'question': question,
                'answer': "Please provide a valid question.",
                'sources': [],
                'error': 'empty_query'
            }
            
        # Check question length
        if len(question) > 1000:
            logger.warning(f"Question too long: {len(question)} chars")
            question = question[:1000]  # Truncate
            logger.info("Truncated question to 1000 chars")
        
        logger.info(f"RAG query: '{question}'")
        
        try:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Retrieval timeout")
            
            # Set timeout (Unix only)
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)
            
            try:
                retrieved_chunks = self.search.search(question, n_results=n_results)
            finally:
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)  # Cancel alarm
            
        except TimeoutError:
            logger.error(f"Retrieval timeout after {timeout}s")
            return {
                'question': question,
                'answer': "The search took too long. Please try a simpler query.",
                'sources': [],
                'error': 'timeout'
            }
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return {
                'question': question,
                'answer': f"Search failed: {str(e)}. Please try again.",
                'sources': [],
                'error': 'retrieval_error'
            }
            
        # Check retrieval results
        if not retrieved_chunks:
            logger.warning("No chunks retrieved")
            return {
                'question': question,
                'answer': self._get_no_results_suggestion(question),
                'sources': [],
                'error': 'no_results',
                'num_chunks_used': 0
            }
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
        
        # Generate answer
        generation_result = self.generator.generate_answer(
            query=question,
            retrieved_chunks=retrieved_chunks,
            include_code=include_code
        )
        
        # Check for generation errors
        if generation_result.get('error'):
            logger.warning(f"Generation error: {generation_result['error']}")
            
            
        # Combine results
        result = {
            'question': question,
            'answer': generation_result['answer'],
            'sources': generation_result['sources'],
            'retrieved_chunks': retrieved_chunks,
            'num_chunks_used': len(retrieved_chunks),
            'error': generation_result.get('error')
        }
        
        return result
    
    
    def _get_no_results_suggestion(self, question: str) -> str:
        """Suggest alternatives when no results found"""
        
        return f"""I couldn't find relevant information for: "{question}"

    Suggestions:
    - Try a more general query (e.g., "routing" instead of "advanced routing patterns")
    - Check spelling and terminology
    - Ask about core FastAPI concepts
    - Try breaking down into simpler questions

    The indexed codebase focuses on FastAPI core functionality."""
    
    def query_safe(self, question: str, **kwargs) -> Dict:
        """
        Safe query wrapper that never throws exceptions.
        Always returns a valid result dict.
        """
        try:
            return self.query(question, **kwargs)
        except Exception as e:
            logger.error(f"Unhandled error in query: {e}")
            return {
                'question': question,
                'answer': f"An unexpected error occurred: {str(e)}\n\nPlease try again or rephrase your question.",
                'sources': [],
                'error': 'unhandled_exception',
                'num_chunks_used': 0
            }
    
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