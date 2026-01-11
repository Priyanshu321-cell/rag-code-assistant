# test_error_handling.py

"""
Test error handling and edge cases.
"""

from src.generation.rag_pipeline import RAGPipeline
from loguru import logger


def test_error_scenarios():
    """Test various error scenarios"""
    
    print("\n" + "="*80)
    print("ERROR HANDLING TESTS")
    print("="*80)
    
    pipeline = RAGPipeline()
    
    test_cases = [
        # (query, description, expected_behavior)
        ("", "Empty query", "Should reject gracefully"),
        ("   ", "Whitespace only", "Should reject gracefully"),
        ("x" * 2000, "Very long query", "Should truncate and handle"),
        ("asdfghjkl qwerty zxcvbn", "Gibberish", "Should return no results gracefully"),
        ("quantum computing in FastAPI", "Irrelevant topic", "Should return no results with suggestions"),
        ("How do I create an endpoint?", "Valid query", "Should work normally"),
    ]
    
    for query, description, expected in test_cases:
        print(f"\n{'='*80}")
        print(f"Test: {description}")
        print(f"Query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        print(f"Expected: {expected}")
        print('-'*80)
        
        try:
            result = pipeline.query_safe(query, n_results=5)
            
            if result.get('error'):
                print(f"✓ Error handled: {result['error']}")
                print(f"Message: {result['answer'][:100]}...")
            else:
                print(f"✓ Success: {len(result['answer'])} chars")
            
        except Exception as e:
            print(f"✗ Unhandled exception: {e}")
    
    print("\n" + "="*80)


def test_api_resilience():
    """Test API failure recovery"""
    
    print("\n" + "="*80)
    print("API RESILIENCE TEST")
    print("="*80)
    
    pipeline = RAGPipeline()
    
    # Test with invalid API key (simulates API failure)
    import os
    original_key = os.getenv("ANTHROPIC_API_KEY")
    
    try:
        # Temporarily use invalid key
        os.environ["ANTHROPIC_API_KEY"] = "invalid_key"
        pipeline.generator.api_key = "invalid_key"
        
        print("\nTesting with invalid API key...")
        result = pipeline.query_safe("How do I create an endpoint?")
        
        if result.get('error'):
            print(f"✓ API error handled gracefully")
            print(f"Error type: {result['error']}")
            print(f"User message: {result['answer'][:150]}...")
        else:
            print("✗ Should have failed with invalid key")
    
    finally:
        # Restore original key
        if original_key:
            os.environ["ANTHROPIC_API_KEY"] = original_key
    
    print("\n" + "="*80)


def test_conversation_errors():
    """Test conversation error handling"""
    
    print("\n" + "="*80)
    print("CONVERSATION ERROR TESTS")
    print("="*80)
    
    from src.generation.conversation import ConversationManager
    
    conv = ConversationManager()
    
    # Test invalid inputs
    print("\n1. Adding invalid turn...")
    success = conv.safe_add_turn("", "Valid answer", [])
    print(f"   Result: {'✓ Rejected' if not success else '✗ Should reject'}")
    
    print("\n2. Adding valid turn...")
    success = conv.safe_add_turn("Valid query", "Valid answer", [])
    print(f"   Result: {'✓ Accepted' if success else '✗ Should accept'}")
    
    print("\n3. Validating state...")
    valid = conv.validate_state()
    print(f"   State valid: {'✓ Yes' if valid else '✗ No'}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Run all tests
    test_error_scenarios()
    test_api_resilience()
    test_conversation_errors()