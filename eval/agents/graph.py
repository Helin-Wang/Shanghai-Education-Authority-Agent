from app.workflow.state import AgentState
from app.workflow.node import retrieve_node, rerank_node, generate_node
from langgraph.graph import StateGraph, END

def create_retriever_test_workflow_graph() -> StateGraph:
    """Create the LangGraph workflow for the retriever test"""
        
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)

    # Define the flow
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app

# Create the workflow instance
retriever_test_workflow_app = create_retriever_test_workflow_graph()