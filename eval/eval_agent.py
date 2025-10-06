import sys
import os
import time
import json
import sqlite3
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, AIMessage
import pandas as pd
from tqdm import tqdm
from langchain_openai import ChatOpenAI

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))

from app.main import run_langgraph_workflow

@dataclass
class ConversationMetrics:
    """Track metrics for a single conversation"""
    question: str
    ground_truth: str
    iterations: int
    total_time: float
    final_answer: str
    conversation_history: List[Dict]
    stop_reason: str  # "satisfied", "max_iterations", "timeout", "error"

class UserSimulationAgent:
    """Agent that simulates a real user in multi-round conversations"""
    
    def __init__(self, max_iterations: int = 5, max_time: float = 300.0):
        """
        Initialize the user simulation agent
        
        Args:
            max_iterations: Maximum number of conversation rounds
            max_time: Maximum time allowed for conversation (seconds)
        """
        self.max_iterations = max_iterations
        self.max_time = max_time
        
        # Initialize LLM for user simulation
        api_key = 'sk-hmqokjrhfszsquludqhbdzftggjriimfelvjjqwzccxnqxmn'
        os.environ["OPENAI_API_BASE"] = 'https://api.siliconflow.cn/v1'
        os.environ["OPENAI_API_KEY"] = api_key
        llm = ChatOpenAI(
            model='Qwen/Qwen2.5-7B-Instruct', 
            openai_api_key=api_key,
            openai_api_base='https://api.siliconflow.cn/v1',
            streaming=True
        )
        self.llm = llm
    
    def should_continue_conversation(self, 
                                   conversation_history: List[Dict], 
                                   ground_truth: str,
                                   iteration: int,
                                   elapsed_time: float) -> Tuple[bool, str]:
        """
        Decide whether to continue the conversation
        
        Args:
            conversation_history: List of conversation turns
            ground_truth: Expected answer
            iteration: Current iteration number
            elapsed_time: Time elapsed so far
            
        Returns:
            Tuple of (should_continue, reason)
        """
        # Check time limit
        if elapsed_time > self.max_time:
            return False, "timeout"
        
        # Check iteration limit
        if iteration >= self.max_iterations:
            return False, "max_iterations"
        
        # Check if user is satisfied with the answer (after first round)
        if len(conversation_history) >= 1:
            last_answer = conversation_history[-1].get("answer", "")
            
            # Use LLM to evaluate if the answer is satisfactory
            satisfaction_prompt = f"""
            你是一个用户，正在与上海教育考试院的智能助手对话。
            
            你的原始问题是: {conversation_history[0].get("query", "")}
            你期望的答案是: {ground_truth}
            
            助手的最新回答是: {last_answer}
            
            请判断这个回答是否满足你的需求。考虑以下因素：
            1. 回答是否准确回答了你的问题
            2. 回答是否提供了足够的信息
            3. 回答是否清晰易懂
            
            请只回答 "满意" 或 "不满意"，不要提供其他内容。
            """
            
            try:
                response = self.llm.invoke(satisfaction_prompt)
                satisfaction = response.content.strip()
                
                if "满意" in satisfaction:
                    return False, "satisfied"
            except Exception as e:
                print(f"Error evaluating satisfaction: {e}")
                # If evaluation fails, continue conversation
        
        return True, "continue"
    
    def generate_user_input(self, 
                          conversation_history: List[Dict], 
                          ground_truth: str,
                          iteration: int,
                          subsection: str = "",
                          section: str = "") -> str:
        """
        Generate the next user input based on conversation history and ground truth
        
        Args:
            conversation_history: List of conversation turns
            ground_truth: Expected answer
            iteration: Current iteration number
            subsection: 部分 information
            section: 考试类型 information
            
        Returns:
            Next user input
        """
        if iteration == 0:
            # First iteration: use the original question
            return conversation_history[0].get("query", "")
        
        # Generate follow-up question based on conversation
        last_answer = conversation_history[-1].get("answer", "")
        
        follow_up_prompt = f"""
        你是一个用户，正在与上海教育考试院的智能助手进行多轮对话。
        
        你的原始问题是: 【{section}-{subsection}】{conversation_history[0].get("query", "")}
        你期望的答案是: {ground_truth}
       
        对话历史:
        """
        
        for i, turn in enumerate(conversation_history):
            follow_up_prompt += f"\n第{i+1}轮:\n"
            follow_up_prompt += f"用户: {turn.get('query', '')}\n"
            follow_up_prompt += f"助手: {turn.get('answer', '')}\n"
        
        follow_up_prompt += f"""
        
        基于以上对话，请生成下一个用户输入。考虑以下情况：
        1. 如果助手需要澄清，请提供更具体的信息
        2. 如果回答不够完整，请询问更多细节
        3. 如果回答有误，请指出错误并重新提问
        4. 如果回答基本正确但需要补充，请询问相关问题
        
        请生成一个自然、真实的用户输入，长度控制在50字以内。
        """
        
        try:
            response = self.llm.invoke(follow_up_prompt)
            return response.content.strip()
        except Exception as e:
            print(f"Error generating user input: {e}")
            # Fallback: ask for clarification
            return "请提供更详细的信息"
    
    def simulate_conversation(self, question: str, ground_truth: str, subsection: str = "", section: str = "") -> ConversationMetrics:
        """
        Simulate a complete multi-round conversation
        
        Args:
            question: Initial question
            ground_truth: Expected answer
            subsection: 部分 information
            section: 考试类型 information
            
        Returns:
            ConversationMetrics object with results
        """
        start_time = time.time()
        conversation_history = []
        conversation_state = None
        iteration = 0
        stop_reason = "error"
        
        try:
            while True:
                # Generate user input
                if iteration == 0:
                    user_input = question
                else:
                    user_input = self.generate_user_input(conversation_history, ground_truth, iteration, subsection, section)
                
                # Run the workflow
                result = run_langgraph_workflow(user_input, conversation_state)
                
                # Extract retrieved document IDs
                retrieved_docs_ids = []
                if result.get("docs"):
                    for doc in result.get("docs", []):
                        if hasattr(doc, 'metadata') and doc.metadata:
                            # Try different possible metadata keys for document ID
                            doc_id = (doc.metadata.get('chunk_index') or 
                                    doc.metadata.get('id') or 
                                    doc.metadata.get('doc_id') or 
                                    doc.metadata.get('source'))
                            if doc_id:
                                retrieved_docs_ids.append(str(doc_id))
                
                # Record this turn
                turn = {
                    "query": user_input,
                    "retrieved_docs_ids": retrieved_docs_ids,
                    "answer": result.get("answer", ""),
                    "needs_clarification": result.get("needs_clarification", False),
                    "docs_count": len(result.get("docs", []))
                }
                conversation_history.append(turn)
                
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
                
                iteration += 1
                elapsed_time = time.time() - start_time
                
                # Check if should continue
                should_continue, reason = self.should_continue_conversation(
                    conversation_history, ground_truth, iteration, elapsed_time
                )
                
                if not should_continue:
                    stop_reason = reason
                    break
            
            total_time = time.time() - start_time
            final_answer = conversation_history[-1].get("answer", "") if conversation_history else ""
            
            return ConversationMetrics(
                question=question,
                ground_truth=ground_truth,
                iterations=iteration,
                total_time=total_time,
                final_answer=final_answer,
                conversation_history=conversation_history,
                stop_reason=stop_reason
            )
            
        except Exception as e:
            print(f"Error in conversation simulation: {e}")
            total_time = time.time() - start_time
            return ConversationMetrics(
                question=question,
                ground_truth=ground_truth,
                iterations=iteration,
                total_time=total_time,
                final_answer="",
                conversation_history=conversation_history,
                stop_reason="error"
            )

def evaluate_multi_round_conversations(qapair_filepath: str, 
                                     results_filepath: str,
                                     max_iterations: int = 5,
                                     max_time: float = 300.0,
                                     sample_size: Optional[int] = None):
    """
    Evaluate multi-round conversations using user simulation
    
    Args:
        qapair_filepath: Path to QA pairs file
        results_filepath: Path to save results
        max_iterations: Maximum iterations per conversation
        max_time: Maximum time per conversation (seconds)
        sample_size: Number of QA pairs to evaluate (None for all)
    """
    print("Loading QA Pairs...")
    
    # Load QA pairs
    qapairs = []
    if qapair_filepath.endswith('.json'):
        with open(qapair_filepath, "r", encoding="utf-8") as f:
            qapairs = json.load(f)
    else:
        df = pd.read_csv(qapair_filepath)
        for index, row in df.iterrows():
            qapair = {
                "question": row['问题'],
                'answer': row['答案'],
                'subsection': row.get('部分', ''),
                'section': row.get('考试类型', '')
            }
            qapairs.append(qapair)
    
    # Sample if specified
    if sample_size and sample_size < len(qapairs):
        import random
        qapairs = random.sample(qapairs, sample_size)
    
    print(f"Evaluating {len(qapairs)} QA pairs...")
    
    # Initialize user simulation agent
    agent = UserSimulationAgent(max_iterations=max_iterations, max_time=max_time)
    
    # Evaluate each QA pair
    eval_results = []
    for qapair in tqdm(qapairs):
        question = qapair["question"]
        ground_truth = qapair["answer"]
        subsection = qapair.get("部分", "")
        section = qapair.get("考试类型", "")
        
        # print(f"\nEvaluating: {question[:50]}...")
        
        # Simulate conversation
        metrics = agent.simulate_conversation(question, ground_truth, subsection, section)
        
        # Convert to serializable format
        result = {
            "question": metrics.question,
            "ground_truth": metrics.ground_truth,
            "iterations": metrics.iterations,
            "total_time": metrics.total_time,
            "final_answer": metrics.final_answer,
            "conversation_history": metrics.conversation_history,
            "stop_reason": metrics.stop_reason
        }
        
        eval_results.append(result)
        
        # print(f"  Iterations: {metrics.iterations}")
        # print(f"  Time: {metrics.total_time:.2f}s")
        # print(f"  Stop reason: {metrics.stop_reason}")

    # Create results directory if not exists
    os.makedirs(os.path.dirname(results_filepath), exist_ok=True)
    
    # Save results
    with open(results_filepath, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=4)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    total_conversations = len(eval_results)
    avg_iterations = sum(r["iterations"] for r in eval_results) / total_conversations
    avg_time = sum(r["total_time"] for r in eval_results) / total_conversations
    
    stop_reasons = {}
    for r in eval_results:
        reason = r["stop_reason"]
        stop_reasons[reason] = stop_reasons.get(reason, 0) + 1
    
    print(f"Total conversations: {total_conversations}")
    print(f"Average iterations: {avg_iterations:.2f}")
    print(f"Average time: {avg_time:.2f}s")
    print(f"Stop reasons: {stop_reasons}")
    
    print(f"\nResults saved to: {results_filepath}")

def interactive_terminal_mode():
    """
    Interactive terminal mode for testing multi-round conversations manually
    """
    print("=" * 60)
    print("Multi-Round Conversation Evaluation - Interactive Mode")
    print("=" * 60)
    print("This mode allows you to manually test multi-round conversations.")
    print("You can act as the user and interact with the Shanghai Education Authority Agent.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60)
    
    # Initialize the agent
    agent = UserSimulationAgent(max_iterations=10, max_time=600.0)
    
    while True:
        try:
            # Get user input
            print("\n" + "-" * 40)
            question = input("Enter your question (or 'quit' to exit): ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                print("Please enter a valid question.")
                continue
            
            # Get ground truth (optional)
            print("\nEnter the expected answer (ground truth) - press Enter to skip:")
            ground_truth = input("Ground truth: ").strip()
            if not ground_truth:
                ground_truth = "No ground truth provided"
            
            # Get 部分 and 考试类型 (optional)
            print("\nEnter 部分 (section) - press Enter to skip:")
            subsection = input("部分: ").strip()
            print("\nEnter 考试类型 (exam type) - press Enter to skip:")
            section = input("考试类型: ").strip()
            
            print(f"\nStarting conversation with question: {question}")
            print("-" * 40)
            
            # Simulate the conversation
            metrics = agent.simulate_conversation(question, ground_truth, subsection, section)
            
            # Display results
            print(f"\n📊 CONVERSATION RESULTS:")
            print(f"   Iterations: {metrics.iterations}")
            print(f"   Total time: {metrics.total_time:.2f}s")
            print(f"   Stop reason: {metrics.stop_reason}")
            
            print(f"\n💬 CONVERSATION HISTORY:")
            for i, turn in enumerate(metrics.conversation_history):
                print(f"\n   Turn {i+1}:")
                print(f"   User: {turn['query']}")
                print(f"   Retrieved docs: {turn['retrieved_docs_ids']}")
                print(f"   Agent: {turn['answer'][:200]}{'...' if len(turn['answer']) > 200 else ''}")
                if turn['needs_clarification']:
                    print(f"   ⚠️  Needs clarification")
            
            print(f"\n✅ Final Answer:")
            print(f"   {metrics.final_answer}")
            
            # Ask if user wants to save this conversation
            save_choice = input("\nSave this conversation to file? (y/n): ").strip().lower()
            if save_choice in ['y', 'yes']:
                filename = input("Enter filename (default: interactive_conversation.json): ").strip()
                if not filename:
                    filename = "interactive_conversation.json"
                
                if not filename.endswith('.json'):
                    filename += '.json'
                
                # Save the conversation
                result = {
                    "question": metrics.question,
                    "ground_truth": metrics.ground_truth,
                    "iterations": metrics.iterations,
                    "total_time": metrics.total_time,
                    "final_answer": metrics.final_answer,
                    "conversation_history": metrics.conversation_history,
                    "stop_reason": metrics.stop_reason,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump([result], f, ensure_ascii=False, indent=4)
                
                print(f"Conversation saved to: {filename}")
            
        except KeyboardInterrupt:
            print("\n\nProgram interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Error occurred: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate multi-round conversations")
    parser.add_argument("--qapair_filepath", 
                       help="QA pairs file path", 
                       default="../data/eval/official_faq.csv")
    parser.add_argument("--results_filepath", 
                       help="Results file path", 
                       default="./results/multi_round_eval.json")
    parser.add_argument("--max_iterations", 
                       type=int, 
                       help="Maximum iterations per conversation", 
                       default=5)
    parser.add_argument("--max_time", 
                       type=float, 
                       help="Maximum time per conversation (seconds)", 
                       default=300.0)
    parser.add_argument("--sample_size", 
                       type=int, 
                       help="Number of QA pairs to evaluate (None for all)", 
                       default=None)
    parser.add_argument("--interactive", 
                       action="store_true", 
                       help="Run in interactive terminal mode")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_terminal_mode()
    else:
        evaluate_multi_round_conversations(
            qapair_filepath=args.qapair_filepath,
            results_filepath=args.results_filepath,
            max_iterations=args.max_iterations,
            max_time=args.max_time,
            sample_size=args.sample_size
        )
