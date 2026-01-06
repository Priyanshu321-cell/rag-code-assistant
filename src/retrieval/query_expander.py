from typing import List
from loguru import logger
import os
from dotenv import load_dotenv
from google import genai
load_dotenv()

class QueryExpander:
    """Expands search query into multiple variations using LLM"""
    def __init__(self, api_key: str = os.getenv(key='GOOGLE_API_KEY'), model: str = "gemini-3-flash-preview"):
        """Initialize query expander"""
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("API key required. Set API_KEY env var or pass api_key parameter")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.cache = {}
        logger.info(f"QueryExpander initialized with model: {model}")
        
    def expand(
        self, 
        query:str,
        n_variations: int = 3,
        context: str = "Python code search"
    )->List[str]:
        """Expand query into variations"""
        
        # Check cache first
        if query in self.cache:
            logger.debug(f"Using cached expansion for '{query}'")
            return self.cache[query]
        
        if(len(query.split(" "))>5):
            logger.debug(f"Query already detailed, skipping expansion: '{query}'")
            return [query]
        
        logger.debug(f"Expanding query: '{query}'")
        
        try:
            prompt = self._build_prompt(query, n_variations, context=context)
            
            response = self.client.models.generate_content(
                model=self.model, contents=prompt
            )
            variations = self._parse_variations(response=response)
            
            all_queries = [query] + variations
            logger.info(f"Expanded '{query}' into {len(all_queries)} queries")
            
            self.cache[query] = all_queries

            return all_queries
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return [query]
    
    def _build_prompt(self, query: str, n_variations: int, context:str)->str:
        """Build prompt for LLM query expansion"""
        prompt = f"""You are helping with code search. Given a user's search query, generate alternative phrasings that capture the same intent.
                        Context: {context}
                        
                        Original query: "{query}"
                        Generate {n_variations} alternative search queries that:
                        1. Capture the same intent as the original
                        2. Use different wording or terminology
                        3. Are concise (2-6 words each)
                        4. Are relevant to code/programming

                        Examples:
                        Query: "auth"
                        Alternatives:
                        - user authentication
                        - login verification
                        - credential checking

                        Query: "routing"
                        Alternatives:
                        - API endpoint routing
                        - request path handling
                        - URL route management

                        Now generate {n_variations} alternatives for: "{query}"

                        Output ONLY the alternatives, one per line, no numbering or extra text."""
        return prompt 
    
    def _parse_variations(self, response: str) -> List[str]:
        """Parse LLM response into list of variations."""
        lines = response.text.split("\n")
        variations = []
        for line in lines:
            line = line.strip()
            
            line = line.lstrip('0123456789.- ')
            
            if not line:
                continue
            
            line = line.strip('"\'')
            
            variations.append(line)
            
        return variations
        
if __name__=="__main__":
    query_expander = QueryExpander()
    print(query_expander.expand("authenticate"))
    print(query_expander.expand("authenticate"))