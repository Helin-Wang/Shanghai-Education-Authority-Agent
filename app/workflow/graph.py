from langgraph.graph import StateGraph, END
from workflow.state import AgentState
from workflow.node import entry_node, attribute_extraction_node,clarification_node, summarize_conversation_node, retrieve_node, rerank_node, generate_node, should_continue_to_retrieve

def create_workflow_graph() -> StateGraph:
    """Create the LangGraph workflow for the Shanghai Education Authority Agent"""
    
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("entry", entry_node)
    workflow.add_node("attribute_extraction", attribute_extraction_node)
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("summarize", summarize_conversation_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate", generate_node)
    
    # Define the flow
    workflow.set_entry_point("entry")
    workflow.add_edge("entry", "summarize")
    workflow.add_edge("summarize", "attribute_extraction")
    workflow.add_edge("attribute_extraction", "clarification")
    workflow.add_conditional_edges(
        "clarification",
        should_continue_to_retrieve,
        {
            "end": END,
            "retrieve": "retrieve"
        }
    )
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app

# Create the workflow instance
workflow_app = create_workflow_graph()
