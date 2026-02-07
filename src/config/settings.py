"""
Configuration settings for the RAG application.
"""
import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    VECTORSTORE_DIR: Path = PROJECT_ROOT / "vectorstore"

    # Document processing settings
    CHUNK_SIZE: int = Field(default=1000, description="Size of text chunks")
    CHUNK_OVERLAP: int = Field(default=200, description="Overlap between chunks")

    # OpenAI settings
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    OPENAI_MODEL: str = Field(default="gpt-5-nano-2025-08-07", description="OpenAI model name")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small", description="Embedding model")

    # Tavily settings
    TAVILY_API_KEY: Optional[str] = Field(default=None, description="Tavily API key for web search")

    # Vector store settings
    VECTORSTORE_TYPE: str = Field(default="faiss", description="Vector store type")
    SIMILARITY_THRESHOLD: float = Field(default=0.7, description="Similarity threshold")
    TOP_K_RESULTS: int = Field(default=5, description="Number of results to retrieve")

    # Logging settings
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_TO_FILE: bool = Field(default=True, description="Enable file logging")
    LOG_TO_CONSOLE: bool = Field(default=True, description="Enable console logging")

    # Application settings
    APP_NAME: str = Field(default="RAG ReAct Agent", description="Application name")
    APP_VERSION: str = Field(default="0.1.0", description="Application version")

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"

    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        self.DATA_DIR.mkdir(exist_ok=True, parents=True)
        self.LOGS_DIR.mkdir(exist_ok=True, parents=True)
        self.VECTORSTORE_DIR.mkdir(exist_ok=True, parents=True)


# Global settings instance
settings = Settings()
settings.ensure_directories()
