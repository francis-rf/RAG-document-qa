"""Vector store module for document embedding and retrieval"""

import shutil
from typing import List, Optional
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)

class VectorStore:
    """
    Vector store with persistent storage support.
    Saves and loads FAISS index to/from disk.
    """

    DEFAULT_PERSIST_PATH = settings.VECTORSTORE_DIR / "faiss_index"

    def __init__(self, persist_directory: Optional[Path] = None):
        try:
            self.embeddings = OpenAIEmbeddings()
            self.vectorstore = None
            self.retriever = None
            self.persist_directory = persist_directory or self.DEFAULT_PERSIST_PATH
            logger.info(f"VectorStore initialized with persist_directory={self.persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize VectorStore: {str(e)}")
            raise

    def create_retriever(self, documents: List[Document], save: bool = True):
        """
        Create vector store and retriever from documents.

        Args:
            documents: List of documents to index
            save: Whether to save the index to disk (default: True)
        """
        try:
            if not documents:
                raise ValueError("No documents provided for indexing")

            logger.info(f"Creating vector store with {len(documents)} documents")
            self.vectorstore = FAISS.from_documents(documents=documents, embedding=self.embeddings)
            self.retriever = self.vectorstore.as_retriever()
            logger.info("Vector store and retriever created successfully")

            if save:
                self.save()

        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}")
            raise

    def save(self, path: Optional[Path] = None):
        """
        Save the vector store to disk.

        Args:
            path: Optional path to save to (uses default if not provided)
        """
        save_path = path or self.persist_directory
        try:
            if self.vectorstore is None:
                raise ValueError("No vector store to save. Create one first.")

            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.vectorstore.save_local(str(save_path))
            logger.info(f"Vector store saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save vector store: {str(e)}")
            raise

    def load(self, path: Optional[Path] = None) -> bool:
        """
        Load the vector store from disk.

        Args:
            path: Optional path to load from (uses default if not provided)

        Returns:
            True if loaded successfully, False otherwise
        """
        load_path = path or self.persist_directory
        try:
            if not Path(load_path).exists():
                logger.info(f"No existing vector store found at {load_path}")
                return False

            self.vectorstore = FAISS.load_local(
                str(load_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vectorstore.as_retriever()
            logger.info(f"Vector store loaded from {load_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            return False

    def exists(self, path: Optional[Path] = None) -> bool:
        """
        Check if a persistent vector store exists.

        Args:
            path: Optional path to check (uses default if not provided)

        Returns:
            True if exists, False otherwise
        """
        check_path = path or self.persist_directory
        return Path(check_path).exists()

    def add_documents(self, documents: List[Document], save: bool = True):
        """
        Add documents to an existing vector store.

        Args:
            documents: List of documents to add
            save: Whether to save after adding (default: True)
        """
        try:
            if self.vectorstore is None:
                raise ValueError("No vector store initialized. Load or create one first.")

            self.vectorstore.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")

            if save:
                self.save()
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise

    def delete(self, path: Optional[Path] = None):
        """
        Delete the persistent vector store.

        Args:
            path: Optional path to delete (uses default if not provided)
        """
        delete_path = path or self.persist_directory
        try:
            if Path(delete_path).exists():
                shutil.rmtree(delete_path)
                logger.info(f"Deleted vector store at {delete_path}")
        except Exception as e:
            logger.error(f"Failed to delete vector store: {str(e)}")
            raise

    def get_retriever(self):
        """
        Get the retriever instance.

        Returns:
            Retriever instance

        Raises:
            ValueError: If retriever not initialized
        """
        if self.retriever is None:
            logger.error("Retriever not initialized")
            raise ValueError('Vector store not initialized. Call create_retriever first.')
        return self.retriever



