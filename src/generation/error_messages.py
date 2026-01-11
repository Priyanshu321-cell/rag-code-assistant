# src/generation/error_messages.py

"""
User-friendly error messages for different failure scenarios.
"""

ERROR_MESSAGES = {
    'empty_query': """I didn't receive a question. Please ask me something about FastAPI!

Examples:
- How do I create an API endpoint?
- What is dependency injection?
- How do I handle file uploads?""",

    'no_results': """I couldn't find relevant information in the FastAPI codebase.

Try:
- Using different keywords
- Being more specific or more general
- Asking about core FastAPI concepts
- Breaking down complex questions""",

    'timeout': """The search is taking too long. This usually means:
- The query is too complex
- The system is under load

Try:
- Simplifying your question
- Asking about a specific aspect
- Waiting a moment and trying again""",

    'api_error': """I'm having trouble connecting to the AI service.

This could be:
- Temporary service issues
- Rate limiting (too many requests)
- Network problems

Try:
- Waiting 10-30 seconds
- Rephrasing your question
- Trying again later""",

    'context_too_large': """Your query requires too much context to process.

Try:
- Asking a more focused question
- Breaking down into smaller questions
- Being more specific about what you need""",

    'generation_failed': """I had trouble generating an answer, but here are the relevant code sections I found.

You can:
- Review the sources below
- Try rephrasing your question
- Ask a more specific follow-up""",
}


def get_error_message(error_type: str, **kwargs) -> str:
    """
    Get user-friendly error message.
    
    Args:
        error_type: Type of error
        **kwargs: Additional context
        
    Returns:
        Formatted error message
    """
    
    message = ERROR_MESSAGES.get(error_type, ERROR_MESSAGES['generation_failed'])
    
    # Add context if provided
    if 'query' in kwargs:
        message = f"Query: \"{kwargs['query']}\"\n\n" + message
    
    if 'details' in kwargs:
        message += f"\n\nTechnical details: {kwargs['details'][:100]}"
    
    return message