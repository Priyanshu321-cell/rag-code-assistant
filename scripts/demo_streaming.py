
"""
Interactive demo of streaming RAG responses.
"""

import sys
from src.generation.rag_pipeline import RAGPipeline


def demo_streaming():
    """Demonstrate streaming RAG"""
    
    print("\n" + "="*80)
    print("STREAMING RAG DEMO")
    print("="*80)
    print("\nInitializing RAG pipeline...")
    
    pipeline = RAGPipeline()
    
    print("✓ Ready!\n")
    
    # Demo queries
    demo_queries = [
        "How do I create an API endpoint?",
        "What is dependency injection in FastAPI?",
        "How do I handle file uploads?",
    ]
    
    if len(sys.argv) > 1:
        # Use command line query
        query = ' '.join(sys.argv[1:])
        pipeline.query_stream_display(query)
    else:
        # Interactive mode
        print("Demo Queries (or type your own):")
        for i, q in enumerate(demo_queries, 1):
            print(f"  {i}. {q}")
        
        print("\nType a number (1-3), your own question, or 'quit' to exit:")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                # Check if number
                if user_input.isdigit() and 1 <= int(user_input) <= len(demo_queries):
                    query = demo_queries[int(user_input) - 1]
                else:
                    query = user_input
                
                if not query:
                    continue
                
                # Stream answer
                pipeline.query_stream_display(query)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    demo_streaming()