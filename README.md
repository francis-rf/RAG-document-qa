# RAG Document Search

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-latest-green.svg)
![AWS](https://img.shields.io/badge/AWS-ECS%20Fargate-orange.svg)

A Retrieval Augmented Generation (RAG) system for semantic document search and Q&A over PDF files, powered by a LangGraph ReAct agent.

## Features

- Upload and index PDF documents
- Semantic search using FAISS + OpenAI embeddings
- LangGraph ReAct agent for intelligent Q&A
- Tavily web search fallback for questions beyond your documents
- Source citations with page references
- Custom HTML/CSS/JS frontend served via FastAPI
- AWS deployment — S3 for PDFs, Secrets Manager for API keys, ECS Fargate for hosting

## How It Works

1. PDFs are uploaded and split into chunks
2. Chunks are embedded using OpenAI `text-embedding-3-small`
3. Embeddings are stored in a FAISS vector database
4. User question is embedded and similarity search finds top-k relevant chunks
5. LangGraph ReAct agent generates an answer using retrieved context
6. If answer is not found in docs, agent falls back to Tavily web search

## Project Structure

```
2.RAG Document Search/
├── src/
│   ├── config/              # Settings — AWS Secrets Manager + .env fallback
│   ├── document_ingestion/  # PDF loading and chunking
│   ├── vectorstore/         # FAISS vector store management
│   ├── nodes/               # LangGraph retriever + ReAct agent nodes
│   ├── graph_builder/       # LangGraph workflow builder
│   ├── state/               # State schema (TypedDict)
│   └── utils/               # Logging
├── static/                  # Frontend (index.html, style.css, script.js)
├── data/                    # PDF documents (local dev only — S3 on AWS)
├── vectorstore/             # FAISS index (local dev only)
├── app.py                   # FastAPI application
├── requirements.txt
└── Dockerfile
```

## Local Development

### Prerequisites

- Python 3.12+
- OpenAI API key
- Tavily API key

### Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Add OPENAI_API_KEY and TAVILY_API_KEY to .env
```

### Run

```bash
uvicorn app:app --reload --port 8000
```

Open `http://localhost:8000`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serves frontend |
| GET | `/api/files` | List PDFs in data directory |
| POST | `/api/upload` | Upload a PDF file |
| POST | `/api/load` | Index documents into vector store |
| POST | `/api/query` | Query documents with a question |

## Docker

```bash
docker build -t rag-document-search .
docker run -p 8000:8000 rag-document-search
```

> On AWS, API keys are loaded from Secrets Manager automatically. No `.env` needed in the container.

## AWS Deployment

Deployed on **ECS Fargate** with:

| Service | Purpose |
|---------|---------|
| ECR | Container image registry |
| ECS Fargate | Serverless container hosting |
| Application Load Balancer | HTTPS traffic routing |
| S3 (`rag-documents-qa`) | PDF storage |
| Secrets Manager (`rag_document`) | API keys |
| CloudWatch | Logs and monitoring |
| IAM | Roles and permissions |

## 📸 Screenshots

![Application Interface](screenshots/image.png)
RAG Document Search Interface

## License

MIT
