"""
RAG Document Search - FastAPI Application
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import shutil
import boto3
from typing import List, Optional

from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder
from src.config.settings import settings
from src.utils.logger import get_logger
from langchain_openai import ChatOpenAI

logger = get_logger(__name__)

app = FastAPI(title="RAG ReAct Agent API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    retrieved_docs: List[dict]

class LoadResponse(BaseModel):
    status: str
    message: str
    chunks_count: Optional[int] = None
    docs_count: Optional[int] = None

class FileListResponse(BaseModel):
    files: List[str]

# Global vector store
vector_store = None


# ── S3 Helpers ─────────────────────────────────────────────────
def get_s3_client():
    return boto3.client("s3", region_name=settings.AWS_REGION)

def list_s3_pdfs() -> List[str]:
    """List PDF filenames from S3 bucket."""
    s3 = get_s3_client()
    response = s3.list_objects_v2(Bucket=settings.S3_BUCKET)
    return [
        Path(obj["Key"]).name
        for obj in response.get("Contents", [])
        if obj["Key"].lower().endswith(".pdf")
    ]

def download_pdfs_from_s3() -> List[str]:
    """Download all PDFs from S3 to local data dir. Returns list of filenames."""
    s3 = get_s3_client()
    response = s3.list_objects_v2(Bucket=settings.S3_BUCKET)
    keys = [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].lower().endswith(".pdf")]
    settings.DATA_DIR.mkdir(exist_ok=True, parents=True)
    for key in keys:
        local_path = settings.DATA_DIR / Path(key).name
        s3.download_file(settings.S3_BUCKET, key, str(local_path))
        logger.info(f"Downloaded {key} from S3")
    return [Path(k).name for k in keys]


# ── API Endpoints ───────────────────────────────────────────────
@app.get("/api")
def root():
    return {"status": "ok", "message": "RAG ReAct Agent API is running"}

@app.get("/api/files", response_model=FileListResponse)
def get_files_api():
    try:
        if settings.S3_BUCKET:
            return FileListResponse(files=list_s3_pdfs())
        else:
            data_dir = settings.DATA_DIR
            pdf_files = list(set(list(data_dir.glob("*.pdf")) + list(data_dir.glob("*.PDF"))))
            return FileListResponse(files=[f.name for f in pdf_files])
    except Exception as e:
        logger.error(f"Error getting file list: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        dest = settings.DATA_DIR / file.filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return {"status": "success", "filename": file.filename}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/load", response_model=LoadResponse)
def load_documents_api():
    global vector_store
    try:
        # On AWS — download PDFs from S3 first
        if settings.S3_BUCKET:
            logger.info(f"Downloading PDFs from S3 bucket: {settings.S3_BUCKET}")
            downloaded = download_pdfs_from_s3()
            if not downloaded:
                return LoadResponse(status="error", message="No PDF files found in S3 bucket")
            logger.info(f"Downloaded {len(downloaded)} file(s) from S3")

        processor = DocumentProcessor(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            extract_images=True
        )

        docs = processor.load_documents([settings.DATA_DIR])

        if not docs:
            return LoadResponse(status="error", message="No documents found")

        chunks = processor.split_documents(docs)
        vs = VectorStore()
        vs.create_retriever(chunks, save=True)
        vector_store = vs

        return LoadResponse(
            status="success",
            message=f"Indexed {len(docs)} document(s) into {len(chunks)} chunks",
            chunks_count=len(chunks),
            docs_count=len(docs)
        )

    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query", response_model=QueryResponse)
def query_documents_api(request: QueryRequest):
    global vector_store
    try:
        if vector_store is None:
            vs = VectorStore()
            if not vs.load():
                raise HTTPException(status_code=400, detail="Please load documents first")
            vector_store = vs

        retriever = vector_store.get_retriever()
        llm = ChatOpenAI(model=settings.OPENAI_MODEL, temperature=0)
        graph_builder = GraphBuilder(retriever=retriever, llm=llm)

        result = graph_builder.run(request.question)

        answer = result.get("answer", "No answer generated.")
        retrieved_docs = result.get("retrieved_docs", [])

        docs_formatted = []
        for doc in retrieved_docs:
            meta = doc.metadata if hasattr(doc, 'metadata') else {}
            docs_formatted.append({
                "source": Path(meta.get('source', 'Unknown')).name,
                "page": meta.get('page', 'N/A'),
                "content": doc.page_content[:300]
            })

        return QueryResponse(answer=answer, retrieved_docs=docs_formatted)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
