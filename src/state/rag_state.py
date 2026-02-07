"""
State management for RAG workflow
"""
from typing import TypedDict, List
from langchain_core.documents import Document


class RAGState(TypedDict):
    """
    State object for RAG Workflow
    """
    question: str
    retrieved_docs: List[Document]
    answer: str
