from langgraph.graph import StateGraph, END
from workflow.state import AgentState
from workflow.node import retrieve_node, rerank_node, generate_node

def create_workflow_graph() -> StateGraph:
    """Create the LangGraph workflow for the Shanghai Education Authority Agent"""
    
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate", generate_node)
    
    # Define the flow
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app

# Create the workflow instance
workflow_app = create_workflow_graph()
