"""
Conversation context management for multi-turn RAG.
Tracks conversation history and resolves follow-up queries.
"""

from typing import List, Dict, Optional
from loguru import logger


class ConversationManager:
    """
    Manages multi-turn conversation context.
    Tracks history and resolves references in follow-up queries.
    """
    
    def __init__(self, max_history: int = 10):
        """
        Initialize conversation manager.
        
        Args:
            max_history: Maximum conversation turns to keep
        """
        self.max_history = max_history
        self.history: List[Dict] = []
        
        logger.info(f"ConversationManager initialized (max_history={max_history})")
    
    
    def add_turn(self, query: str, answer: str, sources: List[Dict]):
        """
        Add a conversation turn to history.
        
        Args:
            query: User's query
            answer: System's answer
            sources: Sources used
        """
        turn = {
            'query': query,
            'answer': answer,
            'sources': sources,
            'turn_number': len(self.history) + 1
        }
        
        self.history.append(turn)
        
        # Trim if exceeds max
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        logger.debug(f"Added turn {turn['turn_number']} to history")
    
    
    def get_history(self, n_turns: Optional[int] = None) -> List[Dict]:
        """
        Get conversation history.
        
        Args:
            n_turns: Number of recent turns (None = all)
            
        Returns:
            List of conversation turns
        """
        if n_turns is None:
            return self.history.copy()
        
        return self.history[-n_turns:] if self.history else []
    
    def validate_state(self) -> bool:
        """
        Validate conversation state integrity.
        
        Returns:
            True if state is valid
        """
        try:
            # Check history structure
            if not isinstance(self.history, list):
                logger.error("History is not a list")
                return False
            
            # Validate each turn
            for i, turn in enumerate(self.history):
                if not isinstance(turn, dict):
                    logger.error(f"Turn {i} is not a dict")
                    return False
                
                required_keys = ['query', 'answer', 'sources', 'turn_number']
                if not all(key in turn for key in required_keys):
                    logger.error(f"Turn {i} missing required keys")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"State validation error: {e}")
            return False
        
    def safe_add_turn(self, query: str, answer: str, sources: List[Dict]) -> bool:
        """
        Safely add turn with validation.
        
        Returns:
            True if successful
        """
        try:
            # Validate inputs
            if not query or not isinstance(query, str):
                logger.warning("Invalid query")
                return False
            
            if not answer or not isinstance(answer, str):
                logger.warning("Invalid answer")
                return False
            
            if not isinstance(sources, list):
                logger.warning("Invalid sources")
                sources = []
            
            # Add turn
            self.add_turn(query, answer, sources)
            
            # Validate state after
            if not self.validate_state():
                logger.error("State invalid after add_turn")
                # Rollback
                self.history = self.history[:-1]
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding turn: {e}")
            return False


    def resolve_query(self, query: str, use_llm: bool = True) -> str:
        """Resolve with fallback on error"""
        
        try:
            return super().resolve_query(query, use_llm)
        except Exception as e:
            logger.error(f"Query resolution failed: {e}")
            # Fallback: return original
            return query
    
    
    def get_context_summary(self, n_turns: int = 3) -> str:
        """
        Get a text summary of recent conversation context.
        
        Args:
            n_turns: Number of recent turns to include
            
        Returns:
            Formatted context string
        """
        recent = self.get_history(n_turns)
        
        if not recent:
            return ""
        
        context_parts = []
        
        for turn in recent:
            context_parts.append(f"User: {turn['query']}")
            # Include short answer summary (first 100 chars)
            answer_summary = turn['answer'][:100] + "..." if len(turn['answer']) > 100 else turn['answer']
            context_parts.append(f"Assistant: {answer_summary}")
        
        return "\n".join(context_parts)
    
    
    def is_follow_up(self, query: str) -> bool:
        """
        Detect if query is a follow-up question.
        
        Args:
            query: User's query
            
        Returns:
            True if appears to be follow-up
        """
        if not self.history:
            return False
        
        query_lower = query.lower().strip()
        
        # Follow-up indicators
        follow_up_patterns = [
            # Pronouns
            'it', 'that', 'this', 'them', 'those', 'these',
            # References
            'the same', 'also', 'too', 'as well',
            # Short questions
            'how', 'why', 'what', 'where', 'when',
            # Continuations
            'and', 'but', 'or', 'so',
            # Examples
            'example', 'show me', 'demonstrate'
        ]
        
        # Very short queries are likely follow-ups
        if len(query.split()) <= 3:
            return True
        
        # Check for follow-up patterns at start
        for pattern in follow_up_patterns:
            if query_lower.startswith(pattern):
                return True
        
        # Check if query contains references without context
        if any(ref in query_lower for ref in ['it', 'that', 'this', 'the same']):
            return True
        
        return False
    
    
    def resolve_query(self, query: str, use_llm: bool = True) -> str:
        """
        Resolve references in follow-up query using context.
        
        Args:
            query: User's potentially ambiguous query
            use_llm: Whether to use LLM for resolution (vs simple rules)
            
        Returns:
            Resolved, standalone query
        """
        if not self.is_follow_up(query):
            return query
        
        if not self.history:
            return query
        
        logger.debug(f"Resolving follow-up query: '{query}'")
        
        if use_llm:
            return self._resolve_with_llm(query)
        else:
            return self._resolve_with_rules(query)
    
    
    def _resolve_with_rules(self, query: str) -> str:
        """Simple rule-based resolution"""
        
        # Get last query topic
        last_turn = self.history[-1]
        last_query = last_turn['query']
        
        # Extract main topic from last query (simple heuristic)
        # e.g., "How do I create an endpoint" → "endpoint"
        last_words = last_query.lower().split()
        
        # Find key nouns (simple: last few words)
        topic = ' '.join(last_words[-3:]) if len(last_words) >= 3 else last_query
        
        # Replace pronouns with topic
        resolved = query.lower()
        
        replacements = {
            'it': topic,
            'that': topic,
            'this': topic,
            'them': topic,
            'those': topic,
            'the same': topic
        }
        
        for pronoun, replacement in replacements.items():
            if pronoun in resolved:
                resolved = resolved.replace(pronoun, replacement)
                break
        
        logger.debug(f"Rule-based resolution: '{query}' → '{resolved}'")
        return resolved
    
    
    def _resolve_with_llm(self, query: str) -> str:
        """LLM-based resolution (more accurate)"""
        
        from google import genai
        import os
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("No API key for LLM resolution, falling back to rules")
            return self._resolve_with_rules(query)
        
        # Get recent context
        context = self.get_context_summary(n_turns=3)
        
        # Build resolution prompt
        prompt = f"""Given this conversation history:

{context}

The user just asked: "{query}"

Rewrite the user's question as a standalone query that includes necessary context from the conversation history. The rewritten query should be understandable without seeing the conversation.

Output ONLY the rewritten query, nothing else."""
        
        try:
            client = genai(api_key=api_key)
            
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                max_tokens=200,
                contents=prompt
            )
            
            resolved = response.content[0].text.strip()
            
            logger.debug(f"LLM resolution: '{query}' → '{resolved}'")
            return resolved
            
        except Exception as e:
            logger.error(f"LLM resolution failed: {e}, falling back to rules")
            return self._resolve_with_rules(query)
    
    
    def clear(self):
        """Clear conversation history"""
        self.history = []
        logger.info("Conversation history cleared")
    
    
    def get_stats(self) -> Dict:
        """Get conversation statistics"""
        return {
            'total_turns': len(self.history),
            'max_history': self.max_history,
            'has_context': len(self.history) > 0
        }


if __name__ == "__main__":
    # Test conversation manager
    
    conv = ConversationManager()
    
    # Simulate conversation
    print("\n" + "="*80)
    print("CONVERSATION MANAGER TEST")
    print("="*80)
    
    # Turn 1
    query1 = "How do I create an API endpoint?"
    answer1 = "Use app.get() decorator or add_api_route() method."
    sources1 = [{'function': 'add_api_route', 'file': 'app.py'}]
    
    conv.add_turn(query1, answer1, sources1)
    print(f"\nTurn 1:")
    print(f"  Query: {query1}")
    print(f"  Stats: {conv.get_stats()}")
    
    # Turn 2 (follow-up)
    query2 = "How do I add authentication to it?"
    
    print(f"\nTurn 2:")
    print(f"  Original query: {query2}")
    print(f"  Is follow-up? {conv.is_follow_up(query2)}")
    
    resolved = conv.resolve_query(query2, use_llm=False)
    print(f"  Resolved query: {resolved}")
    
    # Turn 3 (another follow-up)
    answer2 = "Add Depends() to your endpoint parameters."
    sources2 = [{'function': 'Depends', 'file': 'security.py'}]
    conv.add_turn(resolved, answer2, sources2)
    
    query3 = "Show me an example"
    print(f"\nTurn 3:")
    print(f"  Original query: {query3}")
    print(f"  Is follow-up? {conv.is_follow_up(query3)}")
    
    resolved3 = conv.resolve_query(query3, use_llm=False)
    print(f"  Resolved query: {resolved3}")
    
    # Show context
    print(f"\nContext summary:")
    print(conv.get_context_summary())
    
    print("\n" + "="*80)