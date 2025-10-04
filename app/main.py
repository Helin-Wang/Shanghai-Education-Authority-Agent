import os
from workflow.graph import workflow_app
from langchain_core.messages import HumanMessage, AIMessage
import sqlite3

# Initialize API configuration
api_key_r1 = 'ebe4d4b6-00ae-4ea7-9890-9356d6a29570'
os.environ["OPENAI_API_BASE"] = 'https://ark.cn-beijing.volces.com/api/v3'
os.environ["OPENAI_API_KEY"] = api_key_r1

def run_langgraph_workflow(query: str, conversation_state: dict = None):
    """Run the LangGraph workflow for a given query with optional conversation state"""
    
    # Initialize or use existing state
    if conversation_state is None:
        conn = sqlite3.connect("../data/shanghai_education_authority_agent.db")
        initial_state = {
            "query": query,
            "docs": [],
            "history": [],
            "answer": None,
            "retriever": None,
            "llm": None,
            "conn": conn,
            "faiss_db_path": "../data/faiss_index",
            "needs_clarification": None
        }
    else:
        # Use existing conversation state
        initial_state = conversation_state.copy()
        initial_state["query"] = query
        # Add new user message to history
        initial_state["history"].append(HumanMessage(content=query))
    
    # Run the workflow
    result = workflow_app.invoke(initial_state)
    
    # Close connection if it was newly created
    if conversation_state is None:
        result["conn"].close()
    
    return result

def run_multi_round_conversation():
    """Run a multi-round conversation with the Shanghai Education Authority Agent"""
    
    print("=" * 60)
    print("上海教育考试院智能助手")
    print("=" * 60)
    print("您好！我是上海教育考试院的智能助手，可以回答关于考试相关的问题。")
    print("请输入您的问题，输入 'quit' 或 'exit' 退出。")
    print("=" * 60)
    
    conversation_state = None
    conversation_round = 0
    
    while True:
        try:
            # Get user input
            user_input = input(f"\n[第{conversation_round + 1}轮] 您的问题: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("\n感谢使用！再见！")
                break
            
            if not user_input:
                print("请输入有效的问题。")
                continue
            
            print(f"\n正在处理您的问题: {user_input}")
            print("-" * 40)
            
            # Run the workflow
            result = run_langgraph_workflow(user_input, conversation_state)
            
            # Update conversation state for next round
            conversation_state = {
                "docs": result.get("docs", []),
                "history": result.get("history", []),
                "retriever": result.get("retriever"),
                "llm": result.get("llm"),
                "conn": result.get("conn"),
                "faiss_db_path": result.get("faiss_db_path"),
                "needs_clarification": result.get("needs_clarification")
            }
            
            # Handle the result
            if result.get("needs_clarification", False):
                print(f"🤔 需要澄清: {result.get('answer', '')}")
                print("请提供更多详细信息，以便我能更好地帮助您。")
            else:
                print(f"✅ 回答: {result.get('answer', '')}")
                
                # Show retrieved documents if available
                if result.get("docs"):
                    print(f"\n📚 参考了 {len(result['docs'])} 个相关文档")
            
            conversation_round += 1
            
        except KeyboardInterrupt:
            print("\n\n程序被用户中断。")
            break
        except Exception as e:
            print(f"\n❌ 处理过程中出现错误: {str(e)}")
            print("请重试或联系技术支持。")

def run_single_query(query: str):
    """Run a single query and display the result"""
    
    print(f"运行查询: {query}")
    print("=" * 50)
    
    # Run the workflow
    result = run_langgraph_workflow(query)
    
    # Display results
    if result.get("needs_clarification", False):
        print(f"🤔 需要澄清: {result.get('answer', '')}")
    else:
        print(f"✅ 回答: {result.get('answer', '')}")
        
        # Show retrieved documents
        if result.get("docs"):
            print(f"\n📚 检索到的文档数量: {len(result['docs'])}")
            for i, doc in enumerate(result["docs"][:3]):  # Show first 3 docs
                print(f"\n文档 {i+1}:")
                print(f"内容预览: {doc.page_content[:200]}...")
                if hasattr(doc, 'metadata') and doc.metadata:
                    print(f"元数据: {doc.metadata}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Single query mode
        query = sys.argv[1]
        run_single_query(query)
    else:
        # Interactive conversation mode
        run_multi_round_conversation()