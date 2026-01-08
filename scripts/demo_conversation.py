# demo_conversation.py

"""
Interactive multi-turn conversation demo.
"""

from src.generation.rag_pipeline import RAGPipeline
from loguru import logger


def demo_conversation():
    """Interactive conversation demo"""
    
    print("\n" + "="*80)
    print("MULTI-TURN CONVERSATION DEMO")
    print("="*80)
    print("\nInitializing RAG pipeline with conversation support...")
    
    pipeline = RAGPipeline()
    
    print("✓ Ready!")
    print("\nYou can now have a conversation. Ask questions and follow-ups.")
    print("Type 'clear' to reset conversation, 'quit' to exit.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                pipeline.clear_conversation()
                print("✓ Conversation cleared\n")
                continue
            
            # Query with conversation context
            result = pipeline.query_conversational(user_input)
            
            # Show if follow-up was detected
            if result['is_followup']:
                print(f"\n💡 (Detected follow-up, resolved to: '{result['resolved_query']}')\n")
            
            # Show answer
            print(f"\nAssistant: {result['answer']}\n")
            
            # Show sources (compact)
            if result['sources']:
                print(f"📚 Sources: {', '.join([s['function'] for s in result['sources'][:3]])}\n")
            
            # Show conversation stats
            stats = result['conversation_stats']
            print(f"💬 Turn {stats['total_turns']}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    demo_conversation()