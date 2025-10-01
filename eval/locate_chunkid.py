import json
import pandas as pd
import ast
from openai import OpenAI
from tqdm import tqdm
import os
import sys
# add ../ to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))
from models.HybridRetriever import HybridRetriever
import sqlite3
import argparse

api_key_r1 = 'sk-hmqokjrhfszsquludqhbdzftggjriimfelvjjqwzccxnqxmn'
llm_client = OpenAI(
        api_key=api_key_r1,
        base_url="https://api.siliconflow.cn/v1"
    )

# Mapping of official category to document category
CATEGORY_MAPPING = {
    "高考学考_秋考": "高考学考_秋季高考",
    "高考学考_春考": "高考学考_春季高考",
    "高考学考_艺术类统一考试": "高考学考_艺体类加试",
    "高考学考_体育类统一考试": "高考学考_艺体类加试",
    "高考学考_三校生高考": "高考学考_三校生高考",
    '高考学考_专科自主招生': '高考学考_专科自主招生',
    '高考学考_高中学业水平考试': '高考学考_高中学业考',
    '高考学考_中职校学业水平考试': '高考学考_中职校学业考',
    '高考学考_其他考试—专升本考试': '高考学考_其他考试',
    '高考学考_其他考试—普通高校联合招收华侨港澳台考试': '高考学考_其他考试',
    '中考中招_中考中招': '中考中招',
    '研考成考_研究生招生考试': '研考成考_研究生招生考试',
    '研考成考_成人高考': '研考成考_成人高考招生考试',
    '研考成考_同等学力人员申请硕士学位外国语水平和学科综合水平全国统一考试': '研考成考_同等学力人员申请硕士学位外国语水平和学科综合水平全国统一考试',
    '自学考试_自学考试': '自学考试',
    '证书考试_全国大学英语四、六级考试（CET)': '证书考试_全国大学英语四、六级考试（CET）',
    '证书考试_全国中小学教师资格考试笔试': '证书考试_全国中小学教师资格考试',
    '证书考试_全国计算机等级考试（NCRE)': '证书考试_全国计算机等级考试（NCRE）',
    '证书考试_上海市高校信息技术水平考试': '证书考试_上海市高等学校信息技术水平考试',
}
DOCUMENT_RELEVANCE_SYSTEM_PROMPT = "You are an expert assistant for a Retrieval-Augmented Generation (RAG) system used in the Shanghai Education Examination Authority's Q&A assistant."
DOCUMENT_RELEVANCE_USER_PROMPT_TEMPLATE = """
### Task
Given a QAPair (with a Question in Chinese and its corresponding Answer in Chinese) and a document text(also in Chinese), determine whether the document could reasonably contain the information needed to answer the Question. 
- If the document text is relevant and could lead to the correct Answer, return `true`.
- If the document text is irrelevant or unlikely to help answer the Question, return `false`.

### Important Notes
1. All inputs (Question, Answer, Document Text) will be in **Chinese**. Do not translate them.  
2. The system is focused on the **Shanghai Education Examination Authority** domain (e.g., exams, policies, admissions, notices, official information).  
3. Your output must be strictly one of the following:
   - `True`
   - `False`
   Nothing else. No explanations. No additional words.

### Input
Question: {question}
Answer: {answer}
Document Text: {text}
"""
DOCUMENT_RELEVANCE_OUTPUT_SCHEMA = {
    "type": "boolean"
}


def check_relevance(chunk_text, question, answer):
    # Use llm to check the relevance between chunk and qa pair
    system_prompt = DOCUMENT_RELEVANCE_SYSTEM_PROMPT
    user_prompt = DOCUMENT_RELEVANCE_USER_PROMPT_TEMPLATE.format(text=chunk_text, question=question, answer=answer)
    response = llm_client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "document_relevance",
                "schema": DOCUMENT_RELEVANCE_OUTPUT_SCHEMA
            }
        }
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    args = parser.parse_args()
    start_index = args.start_index
    end_index = args.end_index
    
    # 1. Load chunks
    with open("../data/v1_chunks.json", "r") as f:
        chunks = json.load(f)
        
        
    # 2. Use HybridRetriever to get the relevant chunks
    if not os.path.exists("./data/official_faq_with_relevant_chunks_by_hybridretriever.csv"):
        print("Building relevant chunks by HybridRetriever...")
        # load the official faq
        officialfaq_pd = pd.read_csv("./data/official_faq.csv")
        print("Total Number of OfficialQA Pairs:", len(officialfaq_pd))
        
        # Map the category to the document category
        category_list = []
        for index, row in officialfaq_pd.iterrows():
            category = CATEGORY_MAPPING[row['部分'] + '_' + row['考试类型']]
            category_list.append(category)
        officialfaq_pd['category'] = category_list
        
        hybrid_retriever = HybridRetriever(
            faiss_db_path="../data/faiss_index",
            conn=sqlite3.connect("../data/shanghai_education_authority_agent.db"),
            k=10
        )
            
        relevant_chunks_list = []
        # Check the relevance between documents and qa pairs
        for index, row in tqdm(officialfaq_pd.iterrows(), total=len(officialfaq_pd)):
            category = row['category']
            question = row['问题']
            answer = row['答案']
            
            # HybridRetriever to get the relevant chunks
            query = str(question)+"\n"+str(answer)
            relevant_chunks = hybrid_retriever.retrieve(query, categories=[category])
            relevant_chunks_list.append([chunk.metadata['chunk_index'] for chunk in relevant_chunks])

        officialfaq_pd['relevant_chunks'] = relevant_chunks_list
        officialfaq_pd.to_csv("./data/official_faq_with_relevant_chunks_by_hybridretriever.csv", index=False)
    
    else:
        # load officialfaq_pd with relevant_chunks
        officialfaq_pd = pd.read_csv("./data/official_faq_with_relevant_chunks_by_hybridretriever.csv")
    
    
    # 3. Use llm to check the relevance of the chunks
    
    # Reshape the chunks
    chunks_dict = {chunk['metadata']['chunk_index']: chunk for chunk in chunks}
    
    # Convert relevant_chunks col to list using ast.literal_eval
    officialfaq_pd['relevant_chunks'] = officialfaq_pd['relevant_chunks'].apply(ast.literal_eval)
    end_index = len(officialfaq_pd) if end_index == -1 else end_index
    officialfaq_pd = officialfaq_pd[start_index:end_index]
    
    relevant_chunks_list_verified_with_llm = []
    for index, row in tqdm(officialfaq_pd.iterrows(), total=len(officialfaq_pd)):
        question = row['问题']
        answer = row['答案']
        relevant_chunks_id_list = row['relevant_chunks']
        relevant_chunks_id_list_verified_with_llm = []
        for chunk_id in relevant_chunks_id_list:
            # Find the chunk text
            chunk_text = chunks_dict[chunk_id]['text']
            if check_relevance(chunk_text, question, answer):
                relevant_chunks_id_list_verified_with_llm.append(chunk_id)
        relevant_chunks_list_verified_with_llm.append(relevant_chunks_id_list_verified_with_llm)
    
    officialfaq_pd['relevant_chunks_verified_with_llm'] = relevant_chunks_list_verified_with_llm
    officialfaq_pd.to_csv(f"./data/official_faq_with_relevant_chunks_relevance_by_llm_{start_index}_{end_index}.csv", index=False)