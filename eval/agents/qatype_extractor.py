from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import json
from tqdm import tqdm
from collections import Counter, defaultdict

# TODO: Extract 'field' from ground truth QA
@dataclass
class FieldInfo:
    """
    Represents a field with its name and value extracted from the answer.
    """
    name: str               # Field name in Chinese
    value: str              # Actual value from the answer

@dataclass
class QuestionTypeExtractionResult:
    """
    Complete result of question type extraction for a QA pair.
    """
    question: str
    answer: str
    type: str               # High-level type (e.g., "TIME", "PROCESS")
    detailed_type: str      # Detailed type (e.g., "报名时间", "考试时间")
    # field: List[FieldInfo]  # Fields with names and values extracted from answer
    confidence: float       # Overall confidence score
    rationale: str          # Explanation for the classification

# JSON Schema for structured response
QUESTION_TYPE_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "description": "High-level category from predefined list or new type if needed"
        },
        "detailed_type": {
            "type": "string", 
            "description": "Specific subtype in Chinese (e.g., 报名时间, 考试时间, 报名流程, 资格条件)"
        },
        # "field": {
        #     "type": "array",
        #     "description": "List of fields with names and values extracted from answer",
        #     "items": {
        #         "type": "object",
        #         "properties": {
        #             "name": {
        #                 "type": "string",
        #                 "description": "Field name in Chinese"
        #             },
        #             "value": {
        #                 "type": "string",
        #                 "description": "Actual value from the answer"
        #             }
        #         },
        #         "required": ["name", "value"]
        #     }
        # },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Confidence score between 0 and 1"
        },
        "rationale": {
            "type": "string",
            "description": "Explanation for the classification"
        }
    },
    "required": ["type", "detailed_type", "confidence", "rationale"]
}

# Prompts for comprehensive question type extraction
QUESTION_TYPE_EXTRACTION_SYSTEM_PROMPT = """\
You are an expert question type classifier for the Shanghai education system FAQs. Your task is to analyze question-answer pairs and extract:

1. **Type**: High-level category from the predefined list: {question_types}
   - If the question cannot be categorized into any of these types, create a new appropriate type name
2. **Detailed Type**: Specific subtype in Chinese (e.g., 报名时间, 考试时间, 报名流程, 报名条件. etc.; add more types if needed)

Return STRICT JSON with keys:
- type: string (high-level category from predefined list or new type if needed)
- detailed_type: string (specific subtype in Chinese; add more detailed types if needed)
- confidence: float (0-1)
- rationale: string (explanation)

Guidelines:
- Use Chinese for detailed_type
- Consider the context of Shanghai education system
- Do NOT include exam types (e.g., 中考, 高考, 教师资格考试) in Detailed Type
- Be precise and avoid generic classifications
- If creating a new type, make it descriptive and follow the naming convention
"""

# Prompts for comprehensive question type extraction
QUESTION_TYPE_EXTRACTION_SYSTEM_PROMPT_GIVEN_TYPES = """\
You are an expert question type classifier for the Shanghai education system FAQs. Your task is to analyze question-answer pairs and extract:

1. **Type**: High-level category from the predefined list: {question_types}
2. **Detailed Type**: Specific subtype in Chinese (e.g., 报名时间, 考试时间, 报名流程, 报名条件. etc.; add more types if needed)

Return STRICT JSON with keys:
- type: string (high-level category from predefined list)
- detailed_type: string (specific subtype in Chinese; add more detailed types if needed)
- confidence: float (0-1)
- rationale: string (explanation)

Guidelines:
- Use Chinese for detailed_type
- Consider the context of Shanghai education system
- You can NOT create new types, only use the types from the predefined list; But you can create new detailed types
- Do NOT include exam types (e.g., 中考, 高考, 教师资格考试) in Detailed Type
- Be precise and avoid generic classifications
- If creating a new type, make it descriptive and follow the naming convention
"""

QUESTION_TYPE_EXTRACTION_USER_PROMPT_TEMPLATE = """\
Question: {question}
Answer: {answer}
Section: {section}
Exam Type: {exam_type}

Analyze this QA pair and extract the question type information according to the schema above.
"""

class QuestionTypeExtractor:
    """
    Comprehensive question type extraction agent for Chinese education FAQs.
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize the question type extractor.
        
        Args:
            llm_client: Language model client for question type extraction
        """
        self.llm_client = llm_client
        self.question_types = [
            "TIME", "PROCESS", "ELIGIBILITY", "LOCATION", "FEES", 
            "DOCUMENTS", "POLICIES", "CONTACT", "RESULTS", "GENERAL_INFO"
        ]
    
    def extract_question_type(self, question: str, answer: str, section: str = "", exam_type: str = "") -> QuestionTypeExtractionResult:
        """
        Extract comprehensive question type information from a QA pair.
        
        Args:
            question: The question text
            answer: The answer text
            section: Section/category of the question
            exam_type: Type of exam (e.g., 秋考, 春考)
            
        Returns:
            QuestionTypeExtractionResult with type, and detailed_type information
        """
        if self.llm_client:
            return self._extract_with_llm(question, answer, section, exam_type)
        else:
            # Fallback to default result when no LLM client is available
            return self._create_default_result(question, answer)
    
    def _extract_with_llm(self, question: str, answer: str, section: str, exam_type: str) -> QuestionTypeExtractionResult:
        """
        Extract question type using LLM.
        """
        # Format the system prompt with available question types
        system_prompt = QUESTION_TYPE_EXTRACTION_SYSTEM_PROMPT.format(
            question_types=", ".join(self.question_types)
        )
        
        prompt = QUESTION_TYPE_EXTRACTION_USER_PROMPT_TEMPLATE.format(
            question=question,
            answer=answer,
            section=section,
            exam_type=exam_type
        )
        
        try:
            response = self.llm_client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "question_type_extraction",
                        "schema": QUESTION_TYPE_EXTRACTION_SCHEMA
                    }
                }
            )
            
            # Parse JSON response
            result_data = json.loads(response.choices[0].message.content)

            return QuestionTypeExtractionResult(
                question=question,
                answer=answer,
                type=result_data.get("type", "GENERAL_INFO"),
                detailed_type=result_data.get("detailed_type", "基本信息"),
                confidence=result_data.get("confidence", 0.7),
                rationale=result_data.get("rationale", "LLM-based classification")
            )
        except Exception as e:
            print(f"Error: {e}")
            # Fallback to default result if LLM fails
            return None
    
    def _create_default_result(self, question: str, answer: str, error: str = None) -> QuestionTypeExtractionResult:
        """
        Create a default result when extraction fails.
        """
        return QuestionTypeExtractionResult(
            question=question,
            answer=answer,
            type="GENERAL_INFO",
            detailed_type="基本信息",
            confidence=0.5,
            rationale=f"Default classification due to extraction failure. Error: {error}" if error else "Default classification"
        )
    
    def batch_extract(self, qa_pairs: List[Dict[str, str]]) -> List[QuestionTypeExtractionResult]:
        """
        Extract question types for multiple QA pairs.
        
        Args:
            qa_pairs: List of dictionaries with keys: question, answer, section, exam_type
            
        Returns:
            List of QuestionTypeExtractionResult objects
        """
        results = []
        for qa_pair in tqdm(qa_pairs):
            result = self.extract_question_type(
                question=qa_pair.get("question", ""),
                answer=qa_pair.get("answer", ""),
                section=qa_pair.get("section", ""),
                exam_type=qa_pair.get("exam_type", "")
            )
            if result is not None:
                results.append(result)
        return results
    
    def aggregate_and_deduplicate_specifictype(self, type_list: List[str]) -> Dict[str, str]:
        """
        Aggregate and deduplicate one specific type using LLM.
        
        Args:
            type_list: List of strings in format 'type'
            
        Returns:
            Dictionary:
            - type_mapping: Maps original types to consolidated types
        """
        system_prompt = """You are an expert at aggregating and deduplicating question types for Chinese education authority FAQs. 
Your task is to analyze a list of question types, then consolidate similar ones.

Guidelines:
1. Group similar types together (e.g., "RESULTS" and "成绩" should be consolidated)
2. Use English for types
3. Ensure you return all mappings for the type list
4. Return mappings that show how original types map to consolidated types"""
        user_prompt = f"""Please analyze and consolidate these question types:

Type list: {type_list}

Return all mappings for the type list."""
        try:
            response = self.llm_client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "type_aggregation_specific",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "type_mapping": {
                                "type": "object",
                                "description": "Mapping from original types to consolidated types"
                            }
                        },
                        "required": ["type_mapping"]
                    }
                }
            }
            )
            result_data = json.loads(response.choices[0].message.content)
            return result_data.get("type_mapping", {})
        except Exception as e:
            print(f"Error in type aggregation: {e}")
            return {t: t for t in type_list}
        
    
    
    def _extract_with_llm_given_types(self, question: str, answer: str, section: str, exam_type: str, types: List[str]) -> QuestionTypeExtractionResult:
        """
        Extract question type using LLM.
        """
        # Format the system prompt with available question types
        system_prompt = QUESTION_TYPE_EXTRACTION_SYSTEM_PROMPT_GIVEN_TYPES.format(
            question_types=", ".join(types)
        )
        
        prompt = QUESTION_TYPE_EXTRACTION_USER_PROMPT_TEMPLATE.format(
            question=question,
            answer=answer,
            section=section,
            exam_type=exam_type
        )
        
        try:
            response = self.llm_client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "question_type_extraction",
                        "schema": QUESTION_TYPE_EXTRACTION_SCHEMA
                    }
                }
            )
            
            # Parse JSON response
            result_data = json.loads(response.choices[0].message.content)

            return QuestionTypeExtractionResult(
                question=question,
                answer=answer,
                type=result_data.get("type", "GENERAL_INFO"),
                detailed_type=result_data.get("detailed_type", "基本信息"),
                confidence=result_data.get("confidence", 0.7),
                rationale=result_data.get("rationale", "LLM-based classification")
            )
        except Exception as e:
            print(f"Error: {e}")
            # Fallback to default result if LLM fails
            return None