"""
Gradio frontend for RAG ReAct Agent application
"""
import gradio as gr
from pathlib import Path
import sys
from typing import List, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from document_ingestion.document_processor import DocumentProcessor
from vectorstore.vectorstore import VectorStore
from graph_builder.graph_builder import GraphBuilder
from config.settings import settings
from utils.logger import get_logger
from langchain_openai import ChatOpenAI

logger = get_logger(__name__)

# --- Logic Functions ---

def load_documents_action():
    """Load documents and return status message."""
    try:
        vs = VectorStore()

        # Use existing store if available
        if vs.exists():
            if vs.load():
                return f"✅ Loaded existing vector store from disk ({vs.persist_directory})"

        # Initialize document processor
        processor = DocumentProcessor(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            extract_images=True
        )

        # Load documents from data directory
        docs = processor.load_documents([settings.DATA_DIR])

        if not docs:
            return "❌ No documents found in the data directory!"

        # Split and index documents
        chunks = processor.split_documents(docs)
        vs.create_retriever(chunks, save=True)

        return f"✅ Successfully loaded and indexed {len(chunks)} chunks from {len(docs)} documents."

    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        return f"❌ Error loading documents: {str(e)}"

def get_file_list():
    """Get list of PDF files in data directory."""
    data_dir = settings.DATA_DIR
    pdf_files = list(set(list(data_dir.glob("*.pdf")) + list(data_dir.glob("*.PDF"))))
    if not pdf_files:
        return "No PDF files found in data directory."
    return "\n".join([f"- {f.name}" for f in pdf_files])

def chatbot_response(message: str, history: List[Tuple[str, str]]):
    """
    Process user message and generate response using RAG.
    Returns: (updated_history, retrieved_docs_text)
    """
    try:
        # Check if vector store exists/is loaded
        vs = VectorStore()
        if not vs.load():
             return history + [(message, "⚠️ Please load documents first!")], ""

        retriever = vs.get_retriever()
        
        # Initialize LLM & Graph
        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0
        )
        graph_builder = GraphBuilder(retriever=retriever, llm=llm)
        
        # Run workflow
        result = graph_builder.run(message)
        
        answer = result.get("answer", "No answer generated.")
        retrieved_docs = result.get("retrieved_docs", [])
        
        # Format retrieved docs for display
        docs_display = ""
        for i, doc in enumerate(retrieved_docs[:5], 1):
            meta = doc.metadata if hasattr(doc, 'metadata') else {}
            source = Path(meta.get('source', 'Unknown')).name
            page = meta.get('page', 'N/A')
            content_preview = doc.page_content[:300].replace('\n', ' ') + "..."
            docs_display += f"**Source {i}:** {source} (Page {page})\n> {content_preview}\n\n"
            
        history.append((message, answer))
        return history, docs_display

    except Exception as e:
        logger.error(f"Error in chatbot: {str(e)}")
        error_msg = f"❌ Error: {str(e)}"
        history.append((message, error_msg))
        return history, ""

# --- UI Construction ---

with gr.Blocks(title="RAG ReAct Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 RAG ReAct Agent")
    gr.Markdown("Ask questions about your documents using AI-powered retrieval.")
    
    with gr.Row():
        # Sidebar-like Column
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

        # Main Chat Column
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, label="Chat History")
            msg = gr.Textbox(placeholder="Ask a question about your documents...", label="Your Question")
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                clear_btn = gr.Button("Clear Chat")
            
            with gr.Accordion("📄 Retrieved Context", open=False):
                context_display = gr.Markdown("No context retrieved yet.")

    # --- Event Wiring ---
    
    # File list
    refresh_files_btn.click(fn=get_file_list, outputs=file_list_display)
    
    # Actions
    load_btn.click(
        fn=load_documents_action,
        outputs=status_output
    )

    # Chat
    submit_btn.click(
        fn=chatbot_response,
        inputs=[msg, chatbot],
        outputs=[chatbot, context_display]
    ).then(
        fn=lambda: "", outputs=msg  # Clear input box
    )
    
    msg.submit(
        fn=chatbot_response,
        inputs=[msg, chatbot],
        outputs=[chatbot, context_display]
    ).then(
        fn=lambda: "", outputs=msg
    )
    
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, context_display])

if __name__ == "__main__":
    demo.queue().launch()
