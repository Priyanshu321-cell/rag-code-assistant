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
        if not retrieved_chunks:
            return{
                'answer':"I couldn't find relevant information to answer your question Please try rephrasing or ask about a different topic.",
                'sources':[],
                'prompt_used': None
            }
            
        # Build prompt
        prompt = self._build_prompt(query, retrieved_chunks, include_code)
        
        logger.debug(f"Generating answer for: '{query}'")
        
        try:
            response = self.client.models.generate_content(
                model = self.model,
                contents=prompt
            )
            
            answer = response
            
            sources = self._extract_sources(retrieved_chunks)
            
            return{
                'answer' : answer.text,
                'sources' : sources,
                'prompt_used' : prompt
            }
        
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return {
                'answer' : f"Error generating answer: {str(e)}",
                'sources' : [],
                'prompt_used': prompt
            }

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