"""
Configuration settings for the RAG application.
"""
import sys
import json
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
import os
import boto3
from botocore.exceptions import ClientError


def get_secret(secret_name: str = "rag_document", region: str = "us-east-1") -> dict:
    """Fetch secrets from AWS Secrets Manager. Returns empty dict if unavailable."""
    try:
        client = boto3.client("secretsmanager", region_name=region)
        response = client.get_secret_value(SecretId=secret_name)
        return json.loads(response["SecretString"])
    except Exception:
        return {}


_secrets = get_secret()
# Inject into os.environ so LangChain/OpenAI clients can find them



class Settings(BaseSettings):
    """Application settings with environment variable support."""

    def __init__(self, **data):
        for key, value in _secrets.items():
            if key not in data:
                data[key] = value
        super().__init__(**data)

    # Project paths — relative to working directory (works locally and in Docker)
    DATA_DIR: Path = Path("data")
    LOGS_DIR: Path = Path("logs")
    VECTORSTORE_DIR: Path = Path("vectorstore")

    # Document processing settings
    CHUNK_SIZE: int = Field(default=1000, description="Size of text chunks")
    CHUNK_OVERLAP: int = Field(default=200, description="Overlap between chunks")

    # OpenAI settings
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    OPENAI_MODEL: str = Field(default="gpt-5-nano", description="OpenAI model name")
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

    # AWS settings (optional — only needed on AWS deployment)
    S3_BUCKET: Optional[str] = Field(default=None, description="S3 bucket for PDFs (AWS only)")
    AWS_REGION: str = Field(default="us-east-1", description="AWS region")

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

    def validate_api_keys(self):
        """Validate that required API keys are set."""
        missing_keys = []

        if not self.OPENAI_API_KEY:
            missing_keys.append("OPENAI_API_KEY")

        if not self.TAVILY_API_KEY:
            missing_keys.append("TAVILY_API_KEY")

        if missing_keys:
            print("\nERROR: Missing required API keys:")
            for key in missing_keys:
                print(f"  - {key}")
            sys.exit(1)


# Global settings instance
settings = Settings()
settings.ensure_directories()
settings.validate_api_keys()

if settings.OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
if settings.TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_KEY
