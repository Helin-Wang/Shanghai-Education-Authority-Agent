from typing import Dict, Any, Optional, List
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from models.M3eEmbedding import M3eEmbeddings
from models.BgeReranker import BgeReranker
from models.HybridRetriever import HybridRetriever
import os
from workflow.state import AgentState
from workflow.utils import extract_year, extract_category, generate_hypothetical_document
from langchain_core.messages import HumanMessage, AIMessage
import json
from langchain_core.pydantic_v1 import BaseModel, Field
from workflow.exam_aliases import EXAM_CATEGORIES, EXAM_ALIASES

# Initialize API configuration
api_key = 'sk-hmqokjrhfszsquludqhbdzftggjriimfelvjjqwzccxnqxmn'
os.environ["OPENAI_API_BASE"] = 'https://api.siliconflow.cn/v1'
os.environ["OPENAI_API_KEY"] = api_key

def entry_node(state: AgentState) -> AgentState:
    query = state["query"]
    # Add query into history
    state["history"].append(
        HumanMessage(content=query)
    )
    return state

def clarification_node(state: AgentState) -> AgentState:
    """Check if the user query is clear and needs clarification"""
    query = state["query"]
    
    # Quick exam category detection using aliases
    query_lower = query.lower().strip()
    
    # Check if query contains any exam category aliases
    detected_categories = []
    for category, aliases in EXAM_ALIASES.items():
        for alias in aliases:
            if alias.lower() in query_lower:
                detected_categories.append(category)
                break
    
    # If we can detect exam category, proceed without clarification
    if detected_categories:
        print(f"Detected exam categories: {detected_categories} - proceeding without clarification")
        state["needs_clarification"] = False
        return state
    
    # Additional check: if query is very short and doesn't contain exam terms, might need clarification
    if len(query.strip()) <= 3:
        print(f"Query too short ({len(query.strip())} chars) and no exam category detected - may need clarification")
        # Continue to LLM analysis for short queries
    
    # Initialize LLM if not already done
    if "llm" not in state or state["llm"] is None:
        llm = ChatOpenAI(
            model='Qwen/Qwen2.5-7B-Instruct', 
            openai_api_key=api_key,
            openai_api_base='https://api.siliconflow.cn/v1',
            streaming=True
        )
        state["llm"] = llm
    
    # Create prompt to analyze query clarity with focus on exam category detection
    clarity_prompt = f"""
    你是一个智能助手，专门分析用户关于上海教育考试院的查询。
    你的主要任务是判断用户的查询是否能够确定具体的考试类型，从而决定是否需要澄清。
    
    ## 任务
    分析用户查询，判断是否能够确定具体的考试类型。如果无法确定考试类型，则需要澄清。
    
    ## 可识别的考试类型列表
    {EXAM_CATEGORIES}
    
    ## 评估标准
    只有在以下情况下才需要澄清：
    1. 查询过于模糊，无法确定任何考试类型（如："考试"、"报名"、"什么时候"）
    2. 查询包含多个可能的考试类型，且无法从上下文推断
    3. 查询只包含科目信息，但无法确定是哪个考试类型的科目（如："英语满分多少"）
    4. 查询的考试类型不在上述列表中
    
    ## 不需要澄清的情况
    - 查询明确包含考试类型（如："高考报名时间"、"考研成绩查询"、"中考分数线"）
    - 查询包含明确的考试相关术语（如："四六级考试"、"教师资格证"、"自考"）
    - 查询包含时间、年份等上下文信息（如："2024年高考"、"今年考研"）
    
    ## 输入
    用户查询: "{query}"

    ## 输出
    输出必须是JSON格式，内容用中文：
    {{
        "needs_clarification": true/false,
        "reason": "说明为什么需要澄清或为什么足够清晰",
        "clarification_question": "具体的澄清问题（如果needs_clarification为false则为空）"
    }}
    
    现在分析给定的查询并按要求JSON格式回复。
    """
    
    class ClarityAnalysis(BaseModel):
        needs_clarification: bool = Field(..., description="Whether the query needs clarification")
        reason: str = Field(..., description="Explanation of why clarification is needed or why it's clear")
        clarification_question: Optional[str] = Field(None, description="Specific question to ask for clarification. Blank if needs_clarification is false.")
    
    llm = state["llm"]
    structured_llm = llm.with_structured_output(ClarityAnalysis)
    clarity_analysis = structured_llm.invoke(clarity_prompt)
    
    try:
        if clarity_analysis.needs_clarification:
            # Add clarification question to history
            state["history"].append(
                AIMessage(content=f"{clarity_analysis.clarification_question}")
            )
            state["answer"] = clarity_analysis.clarification_question
            state["needs_clarification"] = True
        else:
            state["needs_clarification"] = False
            
    except Exception as e:
        print(f"Clarity analysis error: {e}")
        # Fallback: default to no clarification needed (more lenient)
        state["needs_clarification"] = False
        print("Defaulting to no clarification needed due to error")
    
    return state

def should_continue_to_retrieve(state: AgentState) -> str:
    """Routing function to decide whether to continue to retrieve or ask for clarification"""
    if state.get("needs_clarification", False):
        return "end"  # Go to user interaction instead of ending
    else:
        return "retrieve"  # Go to conversation summarization first

def summarize_conversation_node(state: AgentState) -> AgentState:
    """Summarize conversation history and generate final query for retrieval"""
    history = state["history"]
    
    # Initialize LLM if not already done
    if "llm" not in state or state["llm"] is None:
        llm = ChatOpenAI(
            model='Qwen/Qwen2.5-7B-Instruct', 
            openai_api_key=api_key,
            openai_api_base='https://api.siliconflow.cn/v1',
            streaming=True
        )
        state["llm"] = llm
    
    # Create conversation summary prompt
    conversation_text = "\n".join([f"{msg.__class__.__name__}: {msg.content}" for msg in history])
    
    summary_prompt = f"""
    你是一个检索查询生成器(Query Generator)，精通RAG与提示工程。你的任务是从对话中提炼关键信息，并生成用于检索数据库的精准中文查询。
    
    ## 输入：用户与助手的对话历史
    {conversation_text}
    
    ## 输出
    输出必须是**JSON格式**，内容用中文，包括：
    {{
        "query": "最终的查询语句",
        "reasoning": "简要依据：列出从对话中提取的关键信息字段及其取值；不要包含思维链或逐步推理细节。"
    }}
    
    ## 生成准则
    1. 涵盖上下文信息：在查询中尽量纳入对话中出现的关键信息字段及其取值，比如考试类型、时间、流程等等
    2. 格式要求：**JSON格式**，`query`为最终的查询语句，`reasoning`为简要依据
    """
    
    class QueryGeneration(BaseModel):
        query: str = Field(..., description="The final query to retrieve documents")
        reasoning: str = Field(..., description="The reasoning process")
    
    llm = state["llm"]
    structured_llm = llm.with_structured_output(QueryGeneration)
    response = structured_llm.invoke(summary_prompt)
    print(response)
    
    # Set the final query for retrieval
    state["query"] = response.query.strip()
    
    return state

def attribute_extraction_node(state: AgentState) -> AgentState:
    """Extract year and category from query"""
    query = state["query"]
    years = extract_year(query)
    categories = extract_category(query, state["llm"])
    state["years"] = years
    state["categories"] = categories
    return state

def need_clarification(state: AgentState) -> str:
    """Check if the query needs clarification"""
    # If categories is not empty, return "no_clarification"
    if state["categories"]:
        return "no_clarification"
    
    # Otherwise, use llm to determine if the query needs clarification
    query = state["query"]
    # Initialize LLM if not already done
    if "llm" not in state or state["llm"] is None:
        llm = ChatOpenAI(
            model='Qwen/Qwen2.5-7B-Instruct', 
            openai_api_key=api_key,
            openai_api_base='https://api.siliconflow.cn/v1',
            streaming=True
        )
        state["llm"] = llm
    
    # Create prompt to analyze query clarity with focus on exam category detection
    clarity_prompt = f"""
    你是一个智能助手，专门分析用户关于上海教育考试院的查询。
    
    请严格按照以下步骤执行，并最终输出一个包含所有字段的完整 JSON 对象。
    
    ## 1. 判断用户输入相关的考试类型列表
    - 从下列考试类型列表中，抽取与用户输入相关的考试类型列表
        {EXAM_CATEGORIES}
    - 如果用户没有提及明确的考试类型，则返回空数组`[]`
    - 字段：
        `"categories": ["考试类型1", "考试类型2", "考试类型3"]`

    ## 2. 判断用户输入是否清晰
    - 清晰的查询输入需要满足：
        - 在第一步抽取的考试类型列表非空
        - 查询没有歧义，没有多重含义、模糊替代等问题
    - 字段：
        `"needs_clarification": true/false`

    ## 3. 如果用户输入不清晰，则生成澄清问题
    - 如果用户输入清晰，则返回空字符串`""`
    - 如果用户输入不清晰，则生成具体的澄清问题
    - 字段：
        `"clarification_question": "具体的澄清问题"`
    
    ## 输出格式：**JSON格式**
    {{
        "categories": ["考试类型1", "考试类型2", "考试类型3"],
        "needs_clarification": true/false,
        "clarification_question": "具体的澄清问题"
    }}


    ## 例子
    ### 示例1
    用户查询："什么时候考数学？"
    输出：
        {{
            "categories": [],
            "needs_clarification": true,
            "clarification_question": "请问您想咨询哪个考试类型的数学考试时间？"
        
        }}
    
    ### 示例2
    用户查询："秋考和春考的数学什么时候考？"
    输出：
        {{
            "categories": ["高考学考_秋季高考", "高考学考_春季高考"],
            "needs_clarification": false,
            "clarification_question": ""
        }}
    

    ## 输入
    用户查询: "{query}"
    """
    
    class ClarityAnalysis(BaseModel):
        categories: List[str] = Field(..., description="The list of exam categories")
        needs_clarification: bool = Field(..., description="Whether the query needs clarification")
        clarification_question: Optional[str] = Field(None, description="Specific question to ask for clarification. Blank if needs_clarification is false.")
    
    llm = state["llm"]
    structured_llm = llm.with_structured_output(ClarityAnalysis)
    clarity_analysis = structured_llm.invoke(clarity_prompt)
    
    try:
        if clarity_analysis.needs_clarification:
            # Add clarification question to history
            state["history"].append(
                AIMessage(content=f"{clarity_analysis.clarification_question}")
            )
            state["answer"] = clarity_analysis.clarification_question
            state["needs_clarification"] = True
        else:
            state["needs_clarification"] = False
            # Update categories
            state["categories"].extend(cat for cat in clarity_analysis.categories if cat not in state["categories"])
            
    except Exception as e:
        print(f"Clarity analysis error: {e}")
        # Fallback: default to no clarification needed (more lenient)
        state["needs_clarification"] = False
        print("Defaulting to no clarification needed due to error")
    
    return state


def retrieve_node(state: AgentState) -> AgentState:
    """根据查询从不同数据库中检索相关文档 - 使用混合检索方法"""
    # Use final query if available (after conversation summarization), otherwise use original query
    query = state["query"]
    print(f"Retrieving documents for query: {query}")
    
    # Initialize hybrid retriever if not already done
    if "retriever" not in state or state["retriever"] is None:
        # Use hybrid retriever combining BM25 and FAISS
        hybrid_retriever = HybridRetriever(
            faiss_db_path=state["faiss_db_path"],
            conn=state["conn"],
            k=6  # Retrieve more documents initially, will be filtered by reranker
        )
        state["retriever"] = hybrid_retriever
    
    # Extract year and category from query
    years = extract_year(query)
    state["years"] = years
    categories = extract_category(query, state["llm"])
    state["categories"] = categories
    
    # Retrieve documents using hybrid approach (raw query first)
    retriever = state["retriever"]

    # Check if HyDE should be used
    use_hyde = state.get("use_hyde", False)  # Default to False if not specified
    llm = state.get("llm")
    
    if use_hyde and llm is not None:
        try:
            hyde_query = generate_hypothetical_document(query, llm)
            print(f"Generated HyDE query: {hyde_query[:200]}...")
            docs, scores = retriever.retrieve(hyde_query, years=years, categories=categories, alpha=0.1, return_scores=True)
        except Exception as e:
            print(f"HyDE retrieval failed: {e}")
            print("Falling back to original query")
            docs, scores = retriever.retrieve(query, years=years, categories=categories, alpha=0.1, return_scores=True)
    else:
        print("No LLM available for HyDE, keeping original results")
        docs, scores = retriever.retrieve(query, years=years, categories=categories, alpha=0.1, return_scores=True)

    state["docs"] = docs
    
    return state

def rerank_node(state: AgentState) -> AgentState:
    """Rerank retrieved documents using BGE reranker"""
    query = state["query"]
    docs = state["docs"]
    
    # Initialize reranker if not already done
    if "reranker" not in state or state["reranker"] is None:
        reranker = BgeReranker()
        state["reranker"] = reranker
    
    # Rerank documents
    reranker = state["reranker"]
    reranked_docs = reranker.rerank(query, docs, top_k=3)  # Keep top 3 after reranking
    state["reranked_docs"] = reranked_docs
    
    return state

def generate_node(state: AgentState) -> AgentState:
    """Generate answer based on reranked documents"""
    query = state["query"]
    docs = state["reranked_docs"]  # Use reranked documents instead of original docs
    
    # Initialize LLM if not already done
    if "llm" not in state or state["llm"] is None:
        llm = ChatOpenAI(
            model='Qwen/Qwen2.5-7B-Instruct', 
            openai_api_key=api_key,
            openai_api_base='https://api.siliconflow.cn/v1',
            streaming=True
        )
        state["llm"] = llm
    
    # Create prompt with retrieved documents
    prompt = f"""
    You are a helpful assistant that can answer questions about Shanghai Education Authority information.
    Based on the following retrieved documents:
    {docs}
    
    Question: {query}
    
    Please provide a comprehensive answer based on the retrieved information. If the information is not sufficient, please indicate that.
    
    Answer:
    """
    
    # Generate response
    llm = state["llm"]
    response = llm.invoke(prompt)
    state["answer"] = response.content
    state["history"].append(
        AIMessage(content=response.content)
    )
    
    return state
