"""
Graph builder for RAG workflow
"""
from langgraph.graph import StateGraph, END
from state.rag_state import RAGState
from nodes.react_node import RAGNodes
from utils.logger import get_logger

logger = get_logger(__name__)

class GraphBuilder:
    """Builds and runs the LangGraph RAG workflow."""
    def __init__(self, retriever, llm):
        """
        Initialize GraphBuilder.

        Args:
            retriever: Document retriever
            llm: Language model
        """
        try:
            self.nodes = RAGNodes(retriever=retriever, llm=llm)
            self.graph = None
            logger.info("GraphBuilder initialized")
        except Exception as e:
            logger.error(f"Failed to initialize GraphBuilder: {str(e)}")
            raise
       
    def build(self):
        """
        Build the RAG workflow graph.

        Returns:
            Compiled graph
        """
        try:
            logger.info("Building workflow graph")
            # Create state graph
            builder = StateGraph(RAGState)

            # Add nodes
            builder.add_node("retriever", self.nodes.retrieve_docs)
            builder.add_node("responder", self.nodes.generate_answer)

            # Set entry point
            builder.set_entry_point("retriever")

            # Add edges
            builder.add_edge("retriever", "responder")
            builder.add_edge('responder', END)

            # Compile graph
            self.graph = builder.compile()
            logger.info("Workflow graph built successfully")
            return self.graph
        except Exception as e:
            logger.error(f"Failed to build workflow graph: {str(e)}")
            raise
    
    def run(self, question: str) -> dict:
        """
        Run the RAG workflow.

        Args:
            question: User question

        Returns:
            Final state as dictionary
        """
        try:
            if self.graph is None:
                self.build()

            logger.info(f"Running workflow for question: {question}")
            initial_state: RAGState = {
                "question": question,
                "retrieved_docs": [],
                "answer": ""
            }
            result = self.graph.invoke(initial_state)
            logger.info("Workflow completed")
            return result
        except Exception as e:
            logger.error(f"Failed to run workflow: {str(e)}")
            raise
    
    
