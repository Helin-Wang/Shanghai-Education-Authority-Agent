"""
For a given text, analyze the possible question types and detailed types.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.doc.chunk import Chunk
from typing import List, Dict
import json
import random

TYPE_CLASSIFIER_SYSTEM_PROMPT = """\
You are an expert in classifying question types and detailed subtypes for Chinese education authority FAQs.  

Your task:  
1. Read the given text carefully.  
2. Identify all possible **question types** (from the predefined list).  
3. For each identified question type, generate one or more possible **detailed types** in Chinese.  
4. For each (question_type, detailed_types) pair, provide a **rationale in Chinses** explaining why it fits.  

Important rules:  
- The possible question types are:  
{question_types}  
- Example detailed types: {example_detailed_types}  
- You must only select question types from this list (no new categories).  
- Only classify if the text itself provides enough information to support the classification.  
  - If the type cannot be reasonably inferred **based only on the given text**, do not classify it.
- If no reasonable classification can be made, return an **empty JSON array**: `[]`.  
- `detailed_types` and `rationale` must always be in Chinese.  
- Multiple `detailed_types` values may be returned for the same `question_type`, but each must be separated by ';'.  
- Output must be a **STRICT JSON array** (no comments, no trailing commas, no text outside the array).  
- Each element must be an object with the following keys:  
  - `"question_type"`: string, one of the provided question types  
  - `"detailed_types"`: string, the inferred detailed subtype separated by ';' in Chinese  
  - `"rationale"`: string, a concise Chinese explanation of the classification  
"""

TYPE_CLASSIFIER_USER_PROMPT_TEMPLATE = """\
You will be given an excerpt from an official notice published by the Shanghai Education Authority.  

Document information:  
- Title: {doc_title}  
- Year: {doc_year}  
- Category: {doc_category}  

Text to analyze:  
{section}  
{text}  

Your task:  
Analyze this text and classify it into one or more **question types** and **detailed types** according to the system instructions.  
Return the result as a **STRICT JSON array** following the required format.
"""

TYPE_CLASSIFIER_OUTPUT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "question_type": {"type": "string"},
            "detailed_types": {"type": "string"},
            "rationale": {"type": "string"}
        },
        "required": ["question_type", "detailed_types", "rationale"]
    }
}

class TypeClassifier:
    def __init__(self, llm_client=None, predefined_types_dict: Dict[str, List[str]] = None):
        self.llm_client = llm_client
        self.predefined_types_dict = predefined_types_dict

    def classify_type(self, chunk: Chunk) -> str:
        system_prompt = TYPE_CLASSIFIER_SYSTEM_PROMPT.format(question_types=self.predefined_types_dict.keys(), example_detailed_types=';'.join(random.choices(list(self.predefined_types_dict[random.choice(list(self.predefined_types_dict.keys()))]), k=3)))
        user_prompt = TYPE_CLASSIFIER_USER_PROMPT_TEMPLATE.format(text=chunk['text'], section=chunk['metadata']['section_breadcrumb'], doc_title=chunk['metadata']['doc_title'], doc_year=chunk['metadata']['doc_year'], doc_category=chunk['metadata']['doc_category'])
        try:
            response = self.llm_client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "type_classifier",
                    "schema": TYPE_CLASSIFIER_OUTPUT_SCHEMA
                }
            })
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            print(f"Error: {e}")
            return None