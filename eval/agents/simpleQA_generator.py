"""
Generate simple questions from a given text. It only relates to current content, not the whole knowledge base.
"""
# modify running path: add the parent directory to the path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.doc.chunk import Chunk
import json
from typing import List, Dict
import random
SIMPLE_QA_GENERATION_SYSTEM_PROMPT = """\
You are a helpful assistant that generates **question–answer pairs** from notices of the Shanghai Education Authority.  

Your task:  
1. Generate simple and clear **questions in Chinese** based on the given text.  
2. The **question must explicitly include the document year (doc_year) and exam type**.  
   - Example: "2024年上海市普通高中学业水平考试的报名时间是什么时候？"  
3. Provide corresponding **answers in Chinese**, derived only from the text.  
4. Use the provided **question type** and example detailed types as guidance.  
   - You may propose new detailed types if needed, as long as they are consistent with the text.  

Important rules:  
- All questions and answers must strictly relate to the given text (no external information).  
- If no reasonable Q&A pair can be generated, return the JSON literal `null`.  
- Return output as a **STRICT JSON array** with the following structure:  
- Each element must be an object with the following keys:  
    - `"question"`: string, the generated question in Chinese
    - `"answer"`: string, the generated answer in Chinese
    - `"type"`: string, the type of the question
    - `"detailed_type"`: string, the detailed type of the question
    - `"confidence"`: float, confidence score between 0.0 and 1.0
    - `"rationale"`: string, a concise explanation in English of why this Q&A fits the text
"""

SIMPLE_QA_GENERATION_USER_PROMPT_TEMPLATE = """\
You will be given:  
- The **question type**  
- Some **example detailed types** (you may also create new detailed types if needed)  
- An excerpt from a Shanghai Education Authority notice  

Details:  
- Question type: {type}  
- Example detailed types: {detailed_types}  
- Document title: {doc_title}  
- Document year: {doc_year}  
- Document category: {doc_category} 

Text to analyze:  
{text}  

Your task:  
Based on this text, generate appropriate **question–answer pairs** following the system instructions.  
Return the result as a **STRICT JSON array** in the required format.   
"""

SIMPLE_QA_GENERATION_OUTPUT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "answer": {"type": "string"},
            "type": {"type": "string"},
            "detailed_type": {"type": "string"},
            "confidence": {"type": "number"},
            "rationale": {"type": "string"}
        },
        "required": ["question", "answer", "type", "detailed_type", "confidence", "rationale"]
    }
}


class SimpleQAGenerator:
    def __init__(self, llm_client=None, predefined_types_dict: Dict[str, List[str]] = None):
        self.llm_client = llm_client
        self.predefined_types_dict = predefined_types_dict

    def generate_simple_QA(self, chunk: Chunk, type: str, detailed_types:str) -> str:
        if type not in self.predefined_types_dict:
            return None
        text = chunk['text']
        doc_title = chunk['metadata']['doc_title']
        doc_year = chunk['metadata']['year']
        doc_category = chunk['metadata']['category']
        
        system_prompt = SIMPLE_QA_GENERATION_SYSTEM_PROMPT
        user_prompt = SIMPLE_QA_GENERATION_USER_PROMPT_TEMPLATE.format(type=type, 
                                                                       detailed_types=detailed_types, 
                                                                       text=text, 
                                                                       doc_title=doc_title,
                                                                       doc_year=doc_year, 
                                                                       doc_category=doc_category)
        
        try:
            response = self.llm_client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "simple_QA_generation",
                        "schema": SIMPLE_QA_GENERATION_OUTPUT_SCHEMA
                    }
                }
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error: {e}")
            return None
        