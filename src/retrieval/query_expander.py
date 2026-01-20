from typing import List
from loguru import logger
import os
from dotenv import load_dotenv
from google import genai
load_dotenv()

class QueryExpander:
    """Expands search query into multiple variations using LLM"""
    def __init__(self, api_key: str | None = os.getenv(key='GOOGLE_API_KEY'), model: str = "gemini-3-flash-preview"):
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
        """Expand query into variations with noise filtering"""
        
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
            variations = self._parse_variations(response=response.text if response.text else "")
            
            # Filter out noisy variations
            filtered_variations = self._filter_variations(query, variations)
            
            all_queries = [query] + filtered_variations
            logger.info(f"Expanded '{query}' into {len(all_queries)} queries after filtering")
            
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
        lines = response.split("\n")
        variations = []
        for line in lines:
            line = line.strip()
            
            line = line.lstrip('0123456789.- ')
            
            if not line:
                continue
            
            line = line.strip('"\'')
            
            variations.append(line)
            
        return variations
    
    def _filter_variations(self, original_query: str, variations: List[str]) -> List[str]:
        """Filter out noisy variations that don't match original intent"""
        filtered = []
        
        for variation in variations:
            # Skip if too similar to original (duplicate)
            if variation.lower() == original_query.lower():
                continue
                
            # Skip if too long (likely noisy)
            if len(variation.split()) > 8:
                continue
                
            # Skip if contains irrelevant keywords
            noise_keywords = {'example', 'tutorial', 'guide', 'learn', 'basic', 'simple'}
            if any(keyword in variation.lower() for keyword in noise_keywords):
                continue
                
            # Skip if no semantic overlap with original
            original_words = set(original_query.lower().split())
            variation_words = set(variation.lower().split())
            
            # Require at least some word overlap or semantic similarity
            overlap = len(original_words & variation_words)
            if overlap == 0 and len(original_words) > 1:
                continue
                
            # Additional semantic filtering
            if not self._is_semantically_similar(original_query, variation):
                continue
                
            filtered.append(variation)
            
        return filtered
    
    def _is_semantically_similar(self, query1: str, query2: str) -> bool:
        """Check semantic similarity using simple heuristics"""
        # Convert to lowercase and split
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        if union == 0:
            return False
            
        jaccard = intersection / union
        
        # Consider similar if Jaccard > 0.2 or they share important terms
        if jaccard > 0.2:
            return True
            
        # Check for shared programming terms
        programming_terms = {
            'function', 'method', 'class', 'api', 'endpoint', 'route',
            'request', 'response', 'parameter', 'argument', 'return',
            'async', 'await', 'decorator', 'middleware', 'handler'
        }
        
        shared_terms = (words1 & programming_terms) & (words2 & programming_terms)
        if len(shared_terms) > 0:
            return True
            
        return False
        
if __name__=="__main__":
    query_expander = QueryExpander()
    print(query_expander.expand("authenticate"))
    print(query_expander.expand("authenticate"))