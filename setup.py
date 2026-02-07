from setuptools import setup, find_packages

setup(
    name="rag-react-agent",
    version="0.1.0",
    description="A RAG system with ReAct agent and simple UI",
    author="Francis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "langchain",
        "langchain-community",
        "langchain-openai",
        "langgraph",
        "openai",
        "faiss-cpu",
        "pydantic",
        "pydantic-settings",
        "python-dotenv",
        "beautifulsoup4",
        "requests",
        "streamlit",
        "wikipedia",
        "pypdf",
        "pymupdf",
    ],
)
