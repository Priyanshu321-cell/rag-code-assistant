"""
Main pipeline for building and searching the code intelligence system.
Connects all components: parser, chunker, embedder, vector store.
"""

from pathlib import Path
from loguru import logger
from typing import List, Dict
import sys

from src.ingestion.parser import parse_directory
from src.ingestion.chunker import chunk_by_function
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore


def build_index(
    repo_path: str = "data/raw/fastapi/fastapi",
    rebuild: bool = False
):
    """
    Build the searchable index from a codebase.
    """
    
    logger.info("="*60)
    logger.info("BUILDING CODE INTELLIGENCE INDEX")
    logger.info("="*60)
    
    # Step 1: Parse the codebase
    logger.info(f"\n[1/4] Parsing codebase: {repo_path}")
    repo_path = Path(repo_path)
    
    if not repo_path.exists():
        logger.error(f"Repository not found at {repo_path}")
        logger.error("Make sure you've cloned FastAPI:")
        logger.error("  cd data/raw && git clone https://github.com/tiangolo/fastapi.git")
        return
    
    functions = parse_directory(repo_path)
    logger.info(f"✓ Parsed {len(functions)} functions")
    
    if len(functions) == 0:
        logger.error("No functions found! Check your parser.")
        return
    
    # Step 2: Create chunks
    logger.info(f"\n[2/4] Creating chunks...")
    chunks = chunk_by_function(functions)
    logger.info(f"✓ Created {len(chunks)} chunks")
    
    # Step 3: Initialize embedder and vector store
    logger.info(f"\n[3/4] Initializing embedder and vector store...")
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    store = VectorStore(embedder=embedder)
    
    # Clear if rebuilding
    if rebuild:
        logger.info("Clearing existing index...")
        store.clear()
    
    # Step 4: Embed and store chunks
    logger.info(f"\n[4/4] Embedding and storing {len(chunks)} chunks...")
    logger.info("This may take a few minutes...")
    
    # Add in batches to show progress
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        store.add_chunks(batch)
        logger.info(f"  Processed {min(i+batch_size, len(chunks))}/{len(chunks)} chunks")
    
    # Final stats
    stats = store.get_stats()
    logger.info("\n" + "="*60)
    logger.info("INDEX BUILD COMPLETE!")
    logger.info("="*60)
    logger.info(f"Total chunks indexed: {stats['total_chunks']}")
    logger.info(f"Collection name: {stats['collection_name']}")
    logger.info(f"Database location: {stats['persist_directory']}")
    logger.info("="*60)


def search_code(
    query: str,
    top_k: int = 5,
    file_filter: str = None
):
    """
    Search the indexed codebase.
    """
    
    logger.info(f"\nSearching for: '{query}'")
    if file_filter:
        logger.info(f"Filtered to file: {file_filter}")
    
    # Initialize
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    store = VectorStore(embedder=embedder)
    
    # Check if index exists
    stats = store.get_stats()
    if stats['total_chunks'] == 0:
        logger.error("No index found! Run 'python main_pipeline.py build' first.")
        return
    
    # Build filter
    filters = None
    if file_filter:
        filters = {'file': file_filter}
    
    # Search
    results = store.search(query, n_results=top_k, filters=filters)
    
    # Display results
    print("\n" + "="*80)
    print(f"SEARCH RESULTS FOR: '{query}'")
    print("="*80)
    
    if not results:
        print("No results found.")
        return
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {result['metadata']['function']}() ")
        print(f"    File: {result['metadata']['file']}")
        print(f"    Similarity: {result['distance']:.2f} (Less is better !)") 
        print(f"    Preview:")
        
        # Show first 200 chars of text
        preview = result['text'][:200].replace('\n', '\n    ')
        print(f"    {preview}")
        if len(result['text']) > 200:
            print(f"    ...")
        print()
    
    print("="*80)


def interactive_search():
    """Interactive search mode - keep asking for queries"""
    
    logger.info("Starting interactive search mode")
    logger.info("Type 'quit' or 'exit' to stop\n")
    
    # Initialize once
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    store = VectorStore(embedder=embedder)
    
    # Check if index exists
    stats = store.get_stats()
    if stats['total_chunks'] == 0:
        logger.error("No index found! Run 'python main_pipeline.py build' first.")
        return
    
    logger.info(f"Index loaded: {stats['total_chunks']} chunks available\n")
    
    while True:
        try:
            # Get query from user
            query = input("Search query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                logger.info("Exiting interactive mode")
                break
            
            if not query:
                continue
            
            # Search
            results = store.search(query, n_results=5)
            
            # Display results
            print("\n" + "-"*80)
            for i, result in enumerate(results, 1):
                print(f"[{i}] {result['metadata']['function']} - {result['metadata']['file']}")
                print(f"    Similarity: {result['distance']:.2f} (Less is better !)")
            print("-"*80 + "\n")
            
        except KeyboardInterrupt:
            logger.info("\nExiting interactive mode")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


def show_stats():
    """Show statistics about the index"""
    
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    store = VectorStore(embedder=embedder)
    stats = store.get_stats()
    
    print("\n" + "="*60)
    print("INDEX STATISTICS")
    print("="*60)
    print(f"Collection name: {stats['collection_name']}")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Database location: {stats['persist_directory']}")
    print("="*60 + "\n")


def main():
    """Main entry point with command-line interface"""
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python main_pipeline.py build              # Build the index")
        print("  python main_pipeline.py rebuild            # Rebuild from scratch")
        print("  python main_pipeline.py search 'query'     # Search the index")
        print("  python main_pipeline.py interactive        # Interactive search mode")
        print("  python main_pipeline.py stats              # Show index statistics")
        print("\nExamples:")
        print("  python main_pipeline.py build")
        print("  python main_pipeline.py search 'how to create endpoint'")
        print("  python main_pipeline.py search 'authentication' --file auth.py")
        print("  python main_pipeline.py interactive")
        print()
        return
    
    command = sys.argv[1].lower()
    
    if command == "build":
        build_index(rebuild=False)
    
    elif command == "rebuild":
        build_index(rebuild=True)
    
    elif command == "search":
        if len(sys.argv) < 3:
            print("Error: Please provide a search query")
            print("Example: python main_pipeline.py search 'authentication'")
            return
        
        query = sys.argv[2]
        
        # Check for file filter
        file_filter = None
        if "--file" in sys.argv:
            file_idx = sys.argv.index("--file")
            if file_idx + 1 < len(sys.argv):
                file_filter = sys.argv[file_idx + 1]
        
        search_code(query, top_k=5, file_filter=file_filter)
    
    elif command == "interactive":
        interactive_search()
    
    elif command == "stats":
        show_stats()
    
    else:
        print(f"Unknown command: {command}")
        print("Use 'build', 'rebuild', 'search', 'interactive', or 'stats'")


if __name__ == "__main__":
    main()