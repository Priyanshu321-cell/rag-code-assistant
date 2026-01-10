"""
RAG answer generation using LLMs 
"""

import os
from typing import List, Dict, Optional
from loguru import logger
from dotenv import load_dotenv
from google import genai
load_dotenv()

class RAGGenerator:
    """Generates answers from retrieved code chunks using LLM"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str= "gemini-3-flash-preview",
        max_tokens: int = 2000
    ):
        """Initialize RAG generator"""
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        self.model = model
        self.max_tokens = max_tokens
        
        logger.info(f"RAGGenerator initialized with {model}")
        
    def generate_answer(
        self, query: str, retrieved_chunks: List[Dict],
        include_code : bool = True
    )-> Dict:
        """Generate answer from query and retrieved chunks."""
        
        # Validate input
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return {
                'answer': "Please provide a valid question.",
                'sources': [],
                'error': 'empty_query'
            }
            
        # checking for results
        if not retrieved_chunks:
            return{
                'answer':"I couldn't find relevant information to answer your question Please try rephrasing or ask about a different topic.",
                'sources':[],
                'prompt_used': None
            }
            
        # Build prompt
        try :
            prompt = self._build_prompt(query, retrieved_chunks, include_code)
        except Exception as e:
            logger.error(f"Prompt building failed: {e}")
            return{
                'answer': "Error preparing the query. Please try rephrasing.",
                'sources': [],
                'error': 'prompt_error'
            }
        
        # check context size 
        estimated_tokens = len(prompt) // 4   
            
        if estimated_tokens > 180000:  
            logger.warning(f"Context too large: ~{estimated_tokens} tokens")
            # Retry with fewer chunks
            return self.generate_answer(
                query, 
                retrieved_chunks[:3],  # Use only top 3
                include_code
            )   
            
            
        # llm call retries
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model = self.model,
                    contents=prompt
                )
                
                answer = response.text
                if not answer or len(answer) < 10:
                    logger.warning("Generated answer too short")
                    answer = "I couldn't generate a complete answer. Please try rephrasing your question."                
                    
                sources = self._extract_sources(retrieved_chunks)
                
                logger.info(f"Answer generated successfully ({len(answer)} chars)")
                return{
                    'answer' : answer,
                    'sources' : sources,
                    'prompt_used' : prompt,
                    'error':None
                }
            
            except Exception as e:
                error_str = str(e)
                
                # Rate limit
                if 'rate_limit' in error_str.lower() or 'overloaded' in error_str.lower():
                    logger.warning(f"Rate limited, attempt {attempt + 1}/{max_retries}")

                    if attempt < max_retries - 1:
                        import time
                        wait_time = (attempt + 1) * 2  # Exponential backoff
                        logger.info(f"Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                
                # Other errors
                logger.error(f"API error (attempt {attempt + 1}): {e}")
                
                if attempt == max_retries - 1:
                    return {
                        'answer': self._get_error_message(error_str),
                        'sources': self._extract_sources(retrieved_chunks),
                        'error': 'api_error',
                        'error_details': error_str
                    }
                    
        # Should not reach here
        return {
            'answer': "An unexpected error occurred. Please try again.",
            'sources': [],
            'error': 'unknown'
        }
    def _get_no_results_message(self, query: str) -> str:
        """Generate helpful message when no results found"""
        
        return f"""I couldn't find relevant information to answer: "{query}"

    This could mean:
    - The topic isn't covered in the indexed FastAPI codebase
    - Try rephrasing your question
    - Try a more general query

    Would you like to try asking in a different way?"""      
    
    def _get_error_message(self, error: str) -> str:
        """Generate user-friendly error message"""
        
        if 'rate_limit' in error.lower():
            return """The AI service is currently busy. Please wait a moment and try again.

    This usually resolves within a few seconds."""
        
        elif 'timeout' in error.lower():
            return """The request timed out. This might be due to network issues.

    Please try again."""
        
        elif 'context_length' in error.lower():
            return """The query context is too large. Try:
    - Asking a more specific question
    - Breaking down into smaller questions"""
        
        else:
            return f"""An error occurred while generating the answer.

    Please try:
    - Rephrasing your question
    - Asking something more specific
    - Trying again in a moment

    Technical details: {error[:100]}"""  

    def generate_answer_stream(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        include_code: bool = True
    ):
        """
        Generate answer with streaming (yields chunks as they come).
        
        Args:
            query: User's question
            retrieved_chunks: Retrieved code chunks
            include_code: Include code in answer
            
        Yields:
            Dict with 'type' and 'content' for each chunk
        """
        
        if not retrieved_chunks:
            yield {
                'type': 'answer',
                'content': "I couldn't find relevant information to answer your question."
            }
            return
        
        # First, yield sources immediately
        sources = self._extract_sources(retrieved_chunks)
        yield {
            'type': 'sources',
            'content': sources
        }
        
        # Build prompt
        prompt = self._build_prompt(query, retrieved_chunks, include_code)
        
        # Stream answer from LLM
        logger.debug(f"Streaming answer for: '{query}'")
        
        try:
            stream = self.client.models.generate_content_stream(
                model=self.model,
                contents=prompt,
                config={
                    "max_output_tokens": self.max_tokens
                }
            )

            for event in stream:
                # Text comes inside model output events
                if event.text:
                    yield {
                        'type': 'answer_chunk',
                        'content': event.text
                    }

            yield {
                'type': 'done',
                'content': None
            }

            logger.info("Streaming complete")

        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield {
                'type': 'error',
                'content': str(e)
            }
        
    def _build_prompt(
        self, query: str, chunks: List[Dict], include_code: bool
    ) -> str:
        """Build the RAG prompt"""
        system = """You are a helpful assistant that answers questions about FastAPI code.

                Your task:
                1. Answer the user's question using ONLY the provided code context
                2. Cite specific functions and files when referencing code
                3. If the context doesn't contain enough information, say so honestly
                4. Keep answers concise and practical
                5. Include code examples from the context when helpful

                Citation format: [Function: function_name, File: filename.py]"""

        context_parts = []
        
        for i, chunk in enumerate(chunks[:5], 1):
            metadata = chunk['metadata']
            function_name = metadata.get('function', 'unknown')
            file_name = metadata.get('file', 'unknown')
            
            # extract clean text
            text = chunk['text']
            
            # formating as numbered
            context_item = f"""[Context {i}]
                Function: {function_name}
                File: {file_name}

                {text}
                """
                
        context_parts.append(context_item)
        context = "\n---\n".join(context_parts)
        
        prompt = f"""{system}
                ## Context
                Here is relevant code from the FastAPI codebase:
                {context}
                ## Question
                {query}
                ## Instructions
                Answer the question using the context provided above. Cite specific functions and files. If you include code examples, keep them short and relevant."""    
                
        return prompt
    
    def _extract_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Extract source information from chunks"""
        sources = []
        
        for chunk in chunks[:5]:
            metadata = chunk['metadata']
            
            source = {
                'function':metadata.get('function', 'unknown'),
                'file': metadata.get('file', 'unknown'),
                'line': metadata.get('line_start', None),
                'chunk_id': chunk.get('id', None)
            }
            
            sources.append(source)
        
        return sources
    
class RAGException(Exception):
    """Base exception for rag errors"""
    pass

class NoRelevantResultsError(RAGException):
    """No relevant chunks found"""
    pass

class ContextTooLargeError(RAGException):
    """Context exceeds model limits"""
    pass

class APIError(RAGException):
    """API call failed"""
    pass

if __name__ == "__main__":
    # Test RAG generator
    
    # Mock retrieved chunks
    test_chunks = [
        {
            'id': 'chunk_1',
            'text': '''async def add_api_route(
    path: str,
    endpoint: Callable,
    **kwargs
):
    """
    Add an API route to the application.
    
    Args:
        path: URL path for the route
        endpoint: Function to handle the route
        **kwargs: Additional route options
    """
    self.router.add_api_route(path, endpoint, **kwargs)''',
            'metadata': {
                'function': 'add_api_route',
                'file': 'applications.py',
                'line_start': 234
            }
        },
        {
            'id': 'chunk_2',
            'text': '''class APIRouter:
    """
    Main routing class for FastAPI applications.
    Groups related routes together.
    """
    
    def __init__(self):
        self.routes = []''',
            'metadata': {
                'function': 'APIRouter',
                'file': 'routing.py',
                'line_start': 45
            }
        }
    ]
    
    # Initialize generator
    generator = RAGGenerator()
    
    # Generate answer
    query = "How do I create an API endpoint in FastAPI?"
    
    result = generator.generate_answer(query, test_chunks)
    
    print("\n" + "="*80)
    print("RAG GENERATION TEST")
    print("="*80)
    print(f"\nQuery: {query}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources:")
    for i, source in enumerate(result['sources'], 1):
        print(f"  {i}. {source['function']} ({source['file']}:{source['line']})")