# RAG Document Search

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-latest-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
[![CI/CD](https://github.com/francis-rf/RAG-document-qa/actions/workflows/deploy.yml/badge.svg)](https://github.com/francis-rf/RAG-document-qa/actions/workflows/deploy.yml)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-AWS%20ECS-orange?logo=amazonaws)](http://rag-alb-1726979633.us-east-1.elb.amazonaws.com/)

A Retrieval Augmented Generation (RAG) system for semantic document search and Q&A over PDF files, powered by a LangGraph ReAct agent.

> **Live Demo:** [http://rag-alb-1726979633.us-east-1.elb.amazonaws.com/](http://rag-alb-1726979633.us-east-1.elb.amazonaws.com/)

## 🎯 Features

- **Document Upload**: Upload and index PDF files directly from the browser
- **Semantic Search**: FAISS vector store with OpenAI `text-embedding-3-small`
- **ReAct Agent**: LangGraph ReAct agent for intelligent multi-step Q&A
- **Web Search Fallback**: Tavily search when answers aren't found in documents
- **Source Citations**: Every answer includes page-level source references
- **Cloud-Native**: S3 for PDFs, Secrets Manager for API keys, ECS Fargate for hosting

## 🛠️ Tech Stack

- **Backend**: FastAPI + Python 3.12
- **AI**: LangGraph ReAct agent, OpenAI embeddings, FAISS vector store
- **APIs**: OpenAI, Tavily
- **Frontend**: Vanilla JavaScript, HTML, CSS
- **Cloud**: AWS ECR + ECS Fargate + ALB + S3 + Secrets Manager
- **CI/CD**: GitHub Actions

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- API Keys:
  - OpenAI API key
  - Tavily API key

### Installation

1. Clone the repository:

```bash
git clone https://github.com/francis-rf/RAG-document-qa.git
cd RAG-document-qa
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` file:

```bash
cp .env.example .env
# Edit .env with your API keys
```

4. Run the application:

```bash
uvicorn app:app --reload --port 8000
```

5. Open browser:

`http://localhost:8000`

## 🐳 Docker Deployment

### Build and Run

```bash
docker build -t rag-document-search .
docker run -p 8000:8000 --env-file .env rag-document-search
```

## ☁️ AWS Deployment

### Services Used

| Service | Purpose |
|---------|---------|
| ECR | Container image registry |
| ECS Fargate | Serverless container hosting |
| Application Load Balancer | HTTP traffic routing |
| S3 (`rag-documents-qa`) | PDF document storage |
| Secrets Manager (`rag_document`) | API key storage |
| CloudWatch | Logs and monitoring |
| IAM | Task roles and permissions |

### Setup

1. Store API keys in **AWS Secrets Manager** under secret name `rag_document`
2. Upload PDFs to **S3** bucket `rag-documents-qa`
3. Push Docker image to **ECR**
4. Deploy via **ECS Fargate** with an ALB pointing to port 8000

### Live URL

The app is deployed and accessible at:

**[http://rag-alb-1726979633.us-east-1.elb.amazonaws.com/](http://rag-alb-1726979633.us-east-1.elb.amazonaws.com/)**

## ⚙️ GitHub Actions CI/CD

Automated deployment is configured via `.github/workflows/deploy.yml`.

### Workflow: Deploy to AWS ECS

On every push to `main`, the pipeline:

1. **Checks out** the code
2. **Configures** AWS credentials
3. **Logs in** to Amazon ECR
4. **Builds & pushes** the Docker image to ECR (tagged with commit SHA and `latest`)
5. **Triggers** a force new deployment on ECS

### Required GitHub Secrets

Add the following secrets to your GitHub repository (`Settings > Secrets > Actions`):

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |

### Workflow Status

[![Deploy to AWS ECS](https://github.com/francis-rf/RAG-document-qa/actions/workflows/deploy.yml/badge.svg)](https://github.com/francis-rf/RAG-document-qa/actions/workflows/deploy.yml)

## 📁 Project Structure

```
RAG-document-qa/
├── app.py                      # FastAPI application
├── src/
│   ├── config/                 # Settings — AWS Secrets Manager + .env fallback
│   ├── document_ingestion/     # PDF loading and chunking
│   ├── vectorstore/            # FAISS vector store management
│   ├── nodes/                  # LangGraph retriever + ReAct agent nodes
│   ├── graph_builder/          # LangGraph workflow builder
│   ├── state/                  # State schema (TypedDict)
│   └── utils/                  # Logging
├── static/                     # Frontend
│   ├── index.html
│   ├── script.js
│   └── style.css
├── data/                       # PDF documents (local only — S3 on AWS)
├── vectorstore/                # FAISS index (local only)
├── .github/workflows/          # CI/CD
│   └── deploy.yml
├── Dockerfile
├── .dockerignore
└── requirements.txt
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serves frontend |
| GET | `/api` | Health check |
| GET | `/api/files` | List PDFs (S3 or local) |
| POST | `/api/upload` | Upload a PDF file |
| POST | `/api/load` | Index documents into vector store |
| POST | `/api/query` | Query documents with a question |

## 📸 Screenshots

![Application Interface](screenshots/image.png)
RAG Document Search Interface

## 📄 License

MIT License
