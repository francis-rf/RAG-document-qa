"""
ReAct agent node with tools for RAG workflow
"""

from typing import List, Optional
from src.state.rag_state import RAGState
from src.utils.logger import get_logger

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Tavily search tool
from tavily import TavilyClient


logger = get_logger(__name__)

class RAGNodes:
    """Handles retrieval and answer generation nodes for the RAG workflow."""
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None
        logger.info("RAGNodes (ReAct) initialized") 

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """
        Retrieve relevant documents.

        Args:
            state: Current RAG state

        Returns:
            Updated state with retrieved documents
        """
        try:
            logger.info(f"Retrieving docs for: {state['question']}")
            docs = self.retriever.invoke(state['question'])
            logger.info(f"Retrieved {len(docs)} documents")
            return {
                "question": state['question'],
                "retrieved_docs": docs,
                "answer": state.get('answer', '')
            }
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {str(e)}")
            return {
                "question": state['question'],
                "retrieved_docs": [],
                "answer": f"Error retrieving documents: {str(e)}"
            }
    
    ## build tools

    def _build_tools(self) -> List[Tool]:
        """
        Build tools for the ReAct agent.

        Returns:
            List of tools
        """
        def retriever_tools_fn(query: str) -> str:
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found"
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, 'metadata') else {}
                title = meta.get("title") or meta.get('source') or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n".join(merged)

        retriever_tool = Tool(
            name="retriever",
            description='Fetch passages from indexed vectorstore',
            func=retriever_tools_fn
        )

        # Tavily web search tool for questions beyond the knowledge base
        tavily_client = TavilyClient()

        def tavily_search_fn(query: str) -> str:
            results = tavily_client.search(query=query, max_results=3)
            if not results or "results" not in results:
                return "No search results found"
            return "\n".join([f"- {r['title']}: {r['content']}" for r in results["results"]])

        tavily_tool = Tool(
            name="tavily_search",
            description="Search the web for current information and general knowledge",
            func=tavily_search_fn
        )

        logger.info("Built 2 tools: retriever and tavily_search")
        return [retriever_tool, tavily_tool]


    
    ## build agent

    def _build_agent(self):
        """
        Build ReAct agent with tools.
        """
        try:
            logger.info("Building ReAct agent")
            system_prompt = (
                "You are a helpful RAG agent. "
                "Prefer 'retriever' for user-provided docs; use 'tavily_search' for general knowledge. "
                "Return only the final useful answer."
            )

            self._agent = create_react_agent(
                model=self.llm,
                prompt=system_prompt,
                tools=self._build_tools(),
            )
            logger.info("ReAct agent built successfully")
        except Exception as e:
            logger.error(f"Failed to build ReAct agent: {str(e)}")
            raise

    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate answer using ReAct agent.

        Args:
            state: Current RAG state

        Returns:
            Updated state with generated answer
        """
        try:
            if self._agent is None:
                self._build_agent()

            logger.info(f"Generating answer with ReAct agent for: {state['question']}")
            result = self._agent.invoke({"messages": [HumanMessage(content=state['question'])]})

            messages = result.get("messages", [])

            answer: Optional[str] = None
            if messages:
                answer_msg = messages[-1]
                answer = getattr(answer_msg, "content", None)

            logger.info("Answer generated successfully")
            return {
                "question": state['question'],
                "retrieved_docs": state.get('retrieved_docs', []),
                "answer": answer or "No answer found."
            }
        except Exception as e:
            logger.error(f"Failed to generate answer with ReAct agent: {str(e)}")
            return {
                "question": state['question'],
                "retrieved_docs": state.get('retrieved_docs', []),
                "answer": f"Error generating answer: {str(e)}"
            }





    
        