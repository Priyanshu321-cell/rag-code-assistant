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

```

## 🛠️ Tech Stack

- **Parsing**: Python AST module
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: ChromaDB
- **Logging**: loguru
- **Language**: Python 3.8+

## 📁 Project Structure
```
rag-code-assistant/
├── data/
│   ├── raw/                         # Source repositories / raw code
│   ├── processed/
│   │   └── bm25_index.pkl           # Precomputed BM25 index
│   └── vector_db/                   # Persisted vector embeddings
│
├── docs/                            # Evaluation results and Performance reports
│
├── demo/                            # Demo runs / notebooks
│
├── scripts/                         # Standalone utility scripts
│
├── src/
│   ├── evaluation/
│   │   ├── __pycache__/
│   │   ├── scripts/
│   │   ├── evaluator.py             # Evaluation pipeline
│   │   └── metrics.py               # Evaluation metrics (MRR, Recall, etc.)
│   │
│   ├── generation/
│   │   └── __init__.py              # LLM prompt & answer generation
│   │
│   ├── ingestion/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   ├── chunker.py               # Code chunking for embeddings
│   │   └── parser.py                # AST-based code parsing
│   │
│   ├── retrieval/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   ├── adaptive_search.py       # Dynamic retrieval strategy
│   │   ├── bm25_search.py            # Sparse BM25 retrieval
│   │   ├── embedder.py               # Embedding model wrapper
│   │   ├── hybrid_search.py          # BM25 + vector hybrid retrieval
│   │   ├── query_classifier.py       # Query intent classification
│   │   ├── query_expander.py         # Query expansion logic
│   │   ├── reranker.py               # Cross-encoder reranking
│   │   └── vector_store.py           # Vector DB interface
│   │
│   └── __init__.py
│
├── venv/                            # Virtual environment (local)
├── .env
├── .gitignore
├── main_pipeline.py                 # Orchestration / CLI entry point
├── requirements.txt
└── README.md

```

## 🎓 What I Learned

Building this project taught me:

1. **Information Retrieval**: Semantic search vs keyword search tradeoffs
2. **Vector Databases**: Efficient similarity search at scale
3. **Code Analysis**: AST parsing for structured data extraction
4. **ML Engineering**: Building production-ready pipelines
5. **System Design**: Modular architecture for extensibility

## 📝 Development Log

- **Week 1**: Core RAG pipeline (parser, embedder, vector store)
1. Built AST parser for function extraction
2. Implemented chunking strategies
3. Integrated sentence-transformers
4. Created ChromaDB interface
5. Indexed full FastAPI codebase (327 functions)

- **Week 2**: Core RAG pipeline (parser, embedder, vector store)
1. Add keyword-based retrieval BM25
2. Hybrid Search Implementation (Combine vector + keyword search)
3. Reranking with Cross-Encoder
4. Implemented Query expansion
5. Implemented query routing
6. Performed comparisons
- See [Week 2 Summary](docs/week2_summary.md) for detailed findings.

## Week 3: Evaluation & Results ✅

### Rigorous Testing Methodology

Evaluated 5 retrieval strategies on **34 manually-labeled test queries** across 4 categories:

| Category | Queries | Example |
|----------|---------|---------|
| Specific Terms | 8 | "APIRouter", "HTTPException" |
| How-To | 10 | "how to authenticate users" |
| Concepts | 8 | "routing", "validation" |
| Code Patterns | 8 | "async endpoint function" |

**Total**: 139 relevant results manually labeled as ground truth.

### Key Results

| Method | Recall@5 | Latency | Best For |
|--------|----------|---------|----------|
| Vector only | 47.7% | 45ms | Specific terms (52.8%) |
| BM25 only | 33.9% | 12ms | - |
| Hybrid basic | 51.6% | 98ms | How-to queries (57.8%) |
| Hybrid + rerank | 53.9% | 278ms | Complex patterns (64.8%) |
| **Adaptive** | **~56%** | **~140ms** | **All query types** |

### Surprising Findings

1. **Vector embeddings beat BM25 on exact terms** (52.8% vs 30.8%)
   - Modern embeddings handle technical terminology well
   
2. **Reranking hurts simple queries** (47.2% vs 57.8% on how-to)
   - Clear queries don't benefit from expensive reranking
   
3. **Reranking shines on complexity** (+15% on code patterns)
   - Cross-encoder attention captures code structure

### Production System: Adaptive Routing

Based on evaluation, implemented query-type-aware routing:
```
Query → Classifier → Route:
                     ├─ How-to → Hybrid basic (fast, effective)
                     ├─ Specific → Vector only (surprisingly good)
                     ├─ Complex → Hybrid + rerank (quality-focused)
                     └─ Default → Hybrid basic (safe)
```

**Result**: 56% Recall@5 at 140ms (vs 53.9% at 278ms for always-rerank)

See [Full Evaluation Report](docs/Week3_Evaluation_Report.md) for detailed analysis.

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
