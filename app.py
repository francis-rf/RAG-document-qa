"""
Combined FastAPI + Gradio application
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import gradio as gr
from pathlib import Path
import sys
from typing import List, Tuple, Optional, Union, Dict

# Add src to path

from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder
from src.config.settings import settings
from src.utils.logger import get_logger
from langchain_openai import ChatOpenAI

logger = get_logger(__name__)

# FastAPI app
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

# API Endpoints
@app.get("/api")
def root():
    return {"status": "ok", "message": "RAG ReAct Agent API is running"}

@app.get("/api/files", response_model=FileListResponse)
def get_files_api():
    try:
        data_dir = settings.DATA_DIR
        pdf_files = list(set(list(data_dir.glob("*.pdf")) + list(data_dir.glob("*.PDF"))))
        file_names = [f.name for f in pdf_files]
        return FileListResponse(files=file_names)
    except Exception as e:
        logger.error(f"Error getting file list: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/load", response_model=LoadResponse)
def load_documents_api():
    global vector_store

    try:
        vs = VectorStore()

        if vs.exists():
            if vs.load():
                vector_store = vs
                return LoadResponse(
                    status="success",
                    message=f"Loaded existing vector store from disk ({vs.persist_directory})"
                )

        processor = DocumentProcessor(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            extract_images=True
        )

        docs = processor.load_documents([settings.DATA_DIR])

        if not docs:
            return LoadResponse(
                status="error",
                message="No documents found in the data directory"
            )

        chunks = processor.split_documents(docs)
        vs.create_retriever(chunks, save=True)
        vector_store = vs

        return LoadResponse(
            status="success",
            message="Successfully loaded and indexed documents",
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

# Gradio UI Functions
def load_documents_action():
    try:
        vs = VectorStore()

        if vs.exists():
            if vs.load():
                return f"✅ Loaded existing vector store from disk ({vs.persist_directory})"

        processor = DocumentProcessor(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            extract_images=True
        )

        docs = processor.load_documents([settings.DATA_DIR])

        if not docs:
            return "❌ No documents found in the data directory!"

        chunks = processor.split_documents(docs)
        vs.create_retriever(chunks, save=True)

        return f"✅ Successfully loaded and indexed {len(chunks)} chunks from {len(docs)} documents."

    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        return f"❌ Error loading documents: {str(e)}"

def get_file_list():
    data_dir = settings.DATA_DIR
    pdf_files = list(set(list(data_dir.glob("*.pdf")) + list(data_dir.glob("*.PDF"))))
    if not pdf_files:
        return "No PDF files found in data directory."
    return "\n".join([f"- {f.name}" for f in pdf_files])

def chatbot_response(message: str, history: Optional[Union[List[Tuple[str, str]], List[Dict]]]):
    try:
        # Normalize history to messages format: list of {'role':..., 'content':...}
        history_msgs: List[Dict] = []
        if history:
            for item in history:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    user, assistant = item
                    history_msgs.append({"role": "user", "content": user})
                    history_msgs.append({"role": "assistant", "content": assistant})
                elif isinstance(item, dict) and "role" in item and "content" in item:
                    history_msgs.append(item)

        vs = VectorStore()
        if not vs.load():
            history_msgs.append({"role": "user", "content": message})
            history_msgs.append({"role": "assistant", "content": "⚠️ Please load documents first!"})
            return history_msgs, ""

        retriever = vs.get_retriever()

        llm = ChatOpenAI(model=settings.OPENAI_MODEL, temperature=0)
        graph_builder = GraphBuilder(retriever=retriever, llm=llm)

        result = graph_builder.run(message)

        answer = result.get("answer", "No answer generated.")
        retrieved_docs = result.get("retrieved_docs", [])

        docs_display = ""
        for i, doc in enumerate(retrieved_docs[:5], 1):
            meta = doc.metadata if hasattr(doc, 'metadata') else {}
            source = Path(meta.get('source', 'Unknown')).name
            page = meta.get('page', 'N/A')
            content_preview = doc.page_content[:300].replace('\n', ' ') + "..."
            docs_display += f"**Source {i}:** {source} (Page {page})\n> {content_preview}\n\n"

        history_msgs.append({"role": "user", "content": message})
        history_msgs.append({"role": "assistant", "content": answer})

        return history_msgs, docs_display

    except Exception as e:
        logger.error(f"Error in chatbot: {str(e)}")
        error_msg = f"❌ Error: {str(e)}"
        try:
            history_msgs.append({"role": "user", "content": message})
            history_msgs.append({"role": "assistant", "content": error_msg})
        except Exception:
            pass
        return history_msgs, ""

# Gradio UI
with gr.Blocks(title="RAG ReAct Agent") as demo:
    gr.Markdown("# 🤖 RAG ReAct Agent")
    gr.Markdown("Ask questions about your documents using AI-powered retrieval.")

    with gr.Row():
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("## ⚙️ Settings")

            gr.Markdown("### 📁 Documents")
            file_list_display = gr.Markdown(value=get_file_list())
            refresh_files_btn = gr.Button("🔄 Refresh File List", size="sm")

            gr.Markdown("### Actions")
            load_btn = gr.Button("📂 Load Documents", variant="primary")

            status_output = gr.Textbox(label="Status", interactive=False, lines=3)

            gr.Markdown("### 🤖 Model Info")
            gr.Markdown(f"**Model:** `{settings.OPENAI_MODEL}`")
            gr.Markdown(f"**Chunk Config:** `{settings.CHUNK_SIZE}` / `{settings.CHUNK_OVERLAP}`")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, label="Chat History")
            msg = gr.Textbox(placeholder="Ask a question about your documents...", label="Your Question")
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                clear_btn = gr.Button("Clear Chat")

            with gr.Accordion("📄 Retrieved Context", open=False):
                context_display = gr.Markdown("No context retrieved yet.")

    refresh_files_btn.click(fn=get_file_list, outputs=file_list_display)
    load_btn.click(fn=load_documents_action, outputs=status_output)
    submit_btn.click(
        fn=chatbot_response,
        inputs=[msg, chatbot],
        outputs=[chatbot, context_display]
    ).then(fn=lambda: "", outputs=msg)

    msg.submit(
        fn=chatbot_response,
        inputs=[msg, chatbot],
        outputs=[chatbot, context_display]
    ).then(fn=lambda: "", outputs=msg)

    clear_btn.click(lambda: ([], ""), outputs=[chatbot, context_display])

# Mount Gradio app
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
