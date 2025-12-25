# In Active Development 🚧

# Code Intelligence Assistant - RAG System for FastAPI

A production-grade Retrieval-Augmented Generation (RAG) system that enables semantic search over the FastAPI codebase. Built from scratch to demonstrate ML engineering skills.


## 🎯 Project Overview

This project implements a complete RAG pipeline that:
- Parses Python codebases using AST (Abstract Syntax Tree)
- Creates semantic embeddings using sentence-transformers
- Stores vectors in ChromaDB for efficient similarity search
- Enables natural language queries over code

**Use Case**: Ask "how do I create an API endpoint?" and get relevant FastAPI functions with citations.

## 🏗️ Architecture
```
┌─────────────┐      ┌──────────┐      ┌───────────┐      ┌──────────────┐
│   Parser    │─────▶│ Chunker  │─────▶│ Embedder  │─────▶│ VectorStore  │
│   (AST)     │      │ (Format) │      │ (Vectors) │      │  (ChromaDB)  │
└─────────────┘      └──────────┘      └───────────┘      └──────────────┘
     │                                                              │
     └──────────────────────────────────────────────────────────────┘
                              Query Flow
```

## ✨ Features

- **Semantic Code Search**: Find relevant code using natural language
- **AST-based Parsing**: Intelligent extraction of functions, classes, and docstrings
- **Hybrid Search Ready**: Architecture supports vector + keyword search (Week 2)
- **Production Design**: Modular, testable, and extensible codebase
- **CLI Interface**: Simple commands for building and searching

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip
git
```

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/rag-code-assistant.git
cd rag-code-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download FastAPI codebase
cd data/raw
git clone https://github.com/tiangolo/fastapi.git
cd ../..
```

### Build the Index
```bash
# Parse FastAPI and build searchable index (~3-5 minutes)
python main_pipeline.py build
```

### Search
```bash
# Search with natural language
python main_pipeline.py search "how to create an API endpoint"
python main_pipeline.py search "user authentication"
python main_pipeline.py search "middleware"

# Interactive mode
python main_pipeline.py interactive
```

## Demo

![Demo](./demo/demo.gif)


## 📊 Performance

| Metric | Value |
|--------|-------|
| Functions Indexed | 327 |
| Embedding Dimension | 384 |
| Search Latency (p95) | <200ms |
| Model | all-MiniLM-L6-v2 |

## 🛠️ Tech Stack

- **Parsing**: Python AST module
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: ChromaDB
- **Logging**: loguru
- **Language**: Python 3.8+

## 📁 Project Structure
```
rag-code-assistant/
├── src/
│   ├── ingestion/
│   │   ├── parser.py          # AST-based code parsing
│   │   └── chunker.py         # Text formatting for embedding
│   └── retrieval/
│       ├── embedder.py        # Sentence-transformers wrapper
│       └── vector_store.py    # ChromaDB interface
├── data/
│   ├── raw/                   # Source repositories
│   └── vector_db/             # Persisted embeddings
├── main_pipeline.py           # CLI entry point
├── requirements.txt
└── README.md
```

## 💡 Usage Examples

### Search for API Routes
```bash
$ python main_pipeline.py search "create endpoint"

[1] add_api_route()
    File: applications.py (line 234)
    Similarity: 89.34%
    
[2] APIRouter()
    File: routing.py (line 45)
    Similarity: 85.67%
```

### Filter by File
```bash
$ python main_pipeline.py search "authentication" --file auth.py
```

### Interactive Mode
```bash
$ python main_pipeline.py interactive
Search query: dependency injection
[1] Depends - dependencies.py
[2] Security - security.py
```

## 🎓 What I Learned

Building this project taught me:

1. **Information Retrieval**: Semantic search vs keyword search tradeoffs
2. **Vector Databases**: Efficient similarity search at scale
3. **Code Analysis**: AST parsing for structured data extraction
4. **ML Engineering**: Building production-ready pipelines
5. **System Design**: Modular architecture for extensibility

## 🚧 Roadmap

**Week 2** (In Progress):
- [ ] Hybrid search (BM25 + vector)
- [ ] Cross-encoder reranking
- [ ] Query expansion with LLMs

**Week 3** (Planned):
- [ ] Evaluation framework with test queries
- [ ] Retrieval metrics (Recall@K, MRR, NDCG)
- [ ] A/B testing different strategies

**Week 4** (Planned):
- [ ] LLM integration for answer generation
- [ ] Citation system
- [ ] Multi-turn conversations

## 📝 Development Log

- **Week 1**: Core RAG pipeline (parser, embedder, vector store)
- Built AST parser for function extraction
- Implemented chunking strategies
- Integrated sentence-transformers
- Created ChromaDB interface
- Indexed full FastAPI codebase (327 functions)

## 🤝 Contributing

This is a learning project, but suggestions are welcome! Open an issue or PR.

## 📄 License

MIT License - feel free to use this for learning

## 👤 Author

**Priyanshu**
- GitHub: [@Priyanshu321-cell](https://github.com/Priyanshu321-cell)

## 🙏 Acknowledgments

- FastAPI team for the excellent codebase to index
- sentence-transformers for embedding models
- ChromaDB for vector database

---

**Status**:  | In Active Development 🚧