import re
import json
import datetime
from typing import Optional, List, Annotated
from workflow.exam_aliases import EXAM_ALIASES
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

def extract_year(query: str) -> list[str]:
    """
    Extract all year information from the user query.

    Returns a list of 4-digit years as strings, in order of appearance,
    de-duplicated. Supports Arabic numerals, Chinese numerals (including 〇),
    relative references, and academic/fiscal year formats.
    """
    if not query or not isinstance(query, str):
        return []

    query = query.strip()
    current_year = datetime.datetime.now().year
    found: list[str] = []

    def add_year(year_value: int) -> None:
        if 1900 <= year_value <= 2099:
            year_str = str(year_value)
            if year_str not in found:
                found.append(year_str)

    # Pre-process: convert Chinese digits to Arabic numerals
    chinese_digit_map = {
        "零": "0", "〇": "0",
        "一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
        "六": "6", "七": "7", "八": "8", "九": "9",
    }
    
    # Convert Chinese digits to Arabic numerals
    normalized_query = query
    for chinese_digit, arabic_digit in chinese_digit_map.items():
        normalized_query = normalized_query.replace(chinese_digit, arabic_digit)

    # Pattern 1: Full 4-digit years (1900-2099)
    for m in re.finditer(r"(19\d{2}|20\d{2})", normalized_query):
        add_year(int(m.group(1)))

    # Pattern 2: 2-digit years with context (年|届|级|-|/|.)
    for m in re.finditer(r"(?<!\d)(\d{2})(?=年|届|级|[-/\.])", normalized_query):
        year_suffix = int(m.group(1))
        if year_suffix <= 30:
            add_year(int(f"20{year_suffix:02d}"))
        else:
            add_year(int(f"19{year_suffix:02d}"))

    # Pattern 3: Relative time references (allow with/without 年)
    relative_patterns = [
        (r"(今年|今|当)(?:年)?", current_year),
        (r"(去年|去|上)(?:年)?", current_year - 1),
        (r"(明年|下|未来)(?:年)?", current_year + 1),
        (r"(前年|前|过去)(?:年)?", current_year - 2),
        (r"(后年|后|未来)(?:年)?", current_year + 2),
        (r"(大前年|三年前)", current_year - 3),
        (r"(大后年|三年后)", current_year + 3),
    ]
    for pattern, base_year in relative_patterns:
        if re.search(pattern, query):  # Use original query for Chinese relative terms
            add_year(base_year)

    # Pattern 4: Academic year references (e.g., "2023-2024学年")
    for m in re.finditer(r"(\d{4})[-/](\d{4})学年", normalized_query):
        start_year = int(m.group(1))
        end_year = int(m.group(2))
        add_year(start_year)
        add_year(end_year)

    # Pattern 5: Fiscal year references
    for m in re.finditer(r"(\d{4})财年|(\d{4})年度", normalized_query):
        year_str = m.group(1) or m.group(2)
        if year_str and year_str.isdigit():
            add_year(int(year_str))

    # If no year is found, use recent 3 years
    if not found:
        add_year(current_year)
        add_year(current_year - 1)
        add_year(current_year - 2)

    return found

def extract_year_using_LLM(query: str, llm: ChatOpenAI) -> List[str]:
    """
    Extract year information from user query using LLM.
    
    Args:
        query: The user's input text to extract years from
        llm: ChatOpenAI instance for processing the query
        
    Returns:
        List of 4-digit years as strings, in order of appearance, de-duplicated.
        Returns empty list if LLM is None or if extraction fails.
    """
    # Initialize LLM if not already done
    if llm is None:
        return []
    
    YEAR_EXTRACTION_SYSTEM_PROMPT = """You are an assistant for a SHANGHAI EDUCATION AUTHORITY information service.
    
    TASK: Extract ONLY explicit 4-digit calendar years mentioned in the user's text. 
    
    Rules: 
    1) Return only Common Era 4-digit years between 1000 and 2100 (inclusive). 
    2) Do not infer or resolve relative dates (e.g., 'last year'); ignore them unless a 4-digit year is explicitly present. 
    3) Do not include ranges like '2018-19' unless a 4-digit year appears (e.g., return '2018' if explicitly written). 
    4) No additional commentary. 
    
    Follow the schema exactly. """ 
    YEAR_EXTRACTION_USER_PROMPT = f"""User query: {query}
    Return the result as the 'years' array per the schema."""
    
    schema = {
        "name": "YearExtraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "years": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "pattern": r"^(20\d{2})$"
                    },
                    "description": "4-digit CE years as strings, deduplicated, in mentioned order."
                }
            },
            "required": ["years"],
            "additionalProperties": False
        }
    }
    try:
        response = llm.invoke(
            [
                {"role": "system", "content": YEAR_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": YEAR_EXTRACTION_USER_PROMPT},
            ],
            response_format={"type": "json_schema", "json_schema": schema},
            temperature=0,
        )
        
        # Parse the response and extract years
        if hasattr(response, 'content') and response.content:
            try:
                result = json.loads(response.content)
                return result.get('years', [])
            except json.JSONDecodeError as json_error:
                print(f"JSON parsing error: {json_error}")
                return []
        else:
            print("No content in LLM response")
            return []
        
    except Exception as e:
        print(f"Error: {e}")
        return []

def extract_category_using_LLM(query: str, llm: ChatOpenAI) -> List[str]:
    """
    Extract exam category information from user query(in Chinese) using LLM.
    
    Args:
        query: The user's input text to extract exam category from
        llm: ChatOpenAI instance for processing the query
        
    Returns:
        List of exam category as strings, in order of appearance, de-duplicated.
        Returns empty list if LLM is None or if extraction fails.
    """
    if llm is None:
        return []
    
    EXAM_CATEGORIES_LIST = ['高考学考_秋季高考', '高考学考_高中学业考', '高考学考_春季高考', '高考学考_三校生高考', '高考学考_艺体类加试', '高考学考_专科自主招生', '高考学考_其他考试', 
                   '高考学考_中职校学业考', '中考中招', '研考成考_研究生招生考试', '研考成考_成人高考招生考试', '研考成考_同等学力人员申请硕士学位外国语水平和学科综合水平全国统一考试', 
                   '自学考试', '证书考试_全国大学英语四、六级考试（CET）', '证书考试_全国中小学教师资格考试', '证书考试_全国英语等级考试（PETS）', 
                   '证书考试_全国计算机等级考试（NCRE）', '证书考试_上海市高等学校信息技术水平考试', '证书考试_普通话水平测试', '证书考试_上海市高等学校教师资格专业课程考试']
    EXAM_CATEGORIES_PATTERN = "|".join(re.escape(cat) for cat in EXAM_CATEGORIES_LIST)
    EXAM_CATEGORIES_LIST_STR = "\n".join(EXAM_CATEGORIES_LIST)
   
    EXAM_CATEGORY_EXTRACTION_SYSTEM_PROMPT = f"""You are an assistant for a SHANGHAI EDUCATION AUTHORITY information service.
    
    TASK: Extract ONLY explicit exam category mentioned in the user's text. 
    
    Rules: 
    1) Return only the exam category names as plain strings.
    2) You may only choose from the predefined list of categories. Do not create or infer new categories.
    3) If multiple categories are mentioned, return each as a separate string.
    4) If no category is mentioned, return an empty list.
    5) Do not add explanations, commentary, or extra text.
    
    Predefined exam categories:
    {EXAM_CATEGORIES_LIST_STR}
    
    Follow the schema exactly.
    """
    EXAM_CATEGORY_EXTRACTION_USER_PROMPT = f"""User query: {query}
    Return the result as the 'categories' array per the schema."""
    
    schema = {
        "name": "ExamCategoryExtraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "categories": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "pattern": r"^({EXAM_CATEGORIES_PATTERN})$"
                    },
                    "description": "Exam category as strings, deduplicated, in mentioned order."
                }
            },
            "required": ["categories"],
            "additionalProperties": False
        }
    }
    try:
        response = llm.invoke(
            [
                {"role": "system", "content": EXAM_CATEGORY_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": EXAM_CATEGORY_EXTRACTION_USER_PROMPT},
            ],
            response_format={"type": "json_schema", "json_schema": schema},
            temperature=0,
        )
        
        # Parse the response and extract years
        if hasattr(response, 'content') and response.content:
            try:
                result = json.loads(response.content)
                if 'categories' not in result:
                    return result
                return result.get('categories', [])
            except json.JSONDecodeError as json_error:
                print(f"JSON parsing error: {json_error}")
                return []
        else:
            print("No content in LLM response")
            return []
        
    except Exception as e:
        print(f"Error: {e}")
        return []

def test_extract_year():
    """
    Test function to validate extract_year() with various inputs
    """
    test_cases = [
        # Basic 4-digit years
        ("2023年", ["2023"]),
        ("1999年", ["1999"]),
        ("2024", ["2024"]),

        # Multiple 4-digit years
        ("2020到2022年", ["2020", "2022"]),

        # 2-digit years with context
        ("23年", ["2023"]),
        ("24届", ["2024"]),
        ("99级", ["1999"]),
        ("01-02", ["2001"]),

        # Chinese numerals (including 〇)
        ("二零二三年", ["2023"]),
        ("二〇二四年", ["2024"]),
        ("一九八四", ["1984"]),

        # Relative time references
        ("今年", [str(datetime.datetime.now().year)]),
        ("去年", [str(datetime.datetime.now().year - 1)]),

        # Academic year references (both years included)
        ("2023-2024学年", ["2023", "2024"]),
        ("2022/2023学年", ["2022", "2023"]),

        # Fiscal year references
        ("2023财年", ["2023"]),
        ("2024年度", ["2024"]),

        # Mixed content with multiple years
        ("2023和二〇二四年安排", ["2023", "2024"]),

        # Edge cases
        ("", []),
        ("没有年份信息", []),
        ("abc123def", []),
        ("3000年", []),  # Invalid year
        ("1800年", []),  # Too old
    ]
    
    print("Testing extract_year() function:")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for input_text, expected in test_cases:
        result = extract_year(input_text)
        status = "✓" if result == expected else "✗"
        print(f"{status} Input: '{input_text}' -> Expected: {expected}, Got: {result}")
        
        if result == expected:
            passed += 1
        else:
            failed += 1
    
    print("=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    return passed, failed

def extract_category(query: str, llm: ChatOpenAI) -> list[str]:
    """
    Extract all category information from the user query.
    """
    result = []
    for standard_name, aliases in EXAM_ALIASES.items():
        for alias in aliases:
            if alias in query:  # 子串匹配
                result.append(standard_name)
                break  # 已经命中一个别名，跳过这个标准词的其他别名
            
    # If no category is found, use LLM to extract
    if not result:
        result = extract_category_using_LLM(query, llm)
    return result

def generate_hypothetical_document(query: str, llm: ChatOpenAI) -> str:
    """
    Generate a hypothetical document that would contain the answer to the query using HyDE approach.
    
    Args:
        query: The original user query
        llm: ChatOpenAI instance for generating the hypothetical document
        
    Returns:
        A hypothetical document string that would contain the answer to the query
    """
    if llm is None:
        return query  # Fallback to original query if no LLM available
    
    HYDE_SYSTEM_PROMPT = """You are an assistant for a SHANGHAI EDUCATION AUTHORITY information service.

TASK: Generate a hypothetical document that would contain the answer to the user's question.

Rules:
1) Write a comprehensive document that would answer the user's question about Shanghai education policies, exams, or procedures.
2) Include relevant details, specific information, and context that would be found in official documents.
3) Use formal, official language appropriate for government documents.
4) Include specific dates, regulations, procedures, and requirements when relevant.
5) Write in Chinese as this is for Shanghai Education Authority.
6) Make the document detailed and informative, as if it were an official FAQ or policy document.
7) Do not include disclaimers or meta-commentary about the document being hypothetical.

The document should be written as if it were an official Shanghai Education Authority document that directly answers the user's question."""
    
    HYDE_USER_PROMPT = f"""User question: {query}

Generate a hypothetical document that would contain the answer to this question. Write it as an official Shanghai Education Authority document."""
    
    try:
        response = llm.invoke([
            {"role": "system", "content": HYDE_SYSTEM_PROMPT},
            {"role": "user", "content": HYDE_USER_PROMPT}
        ], temperature=0.3)  # Lower temperature for more consistent, factual content
        
        if hasattr(response, 'content') and response.content:
            return response.content.strip()
        else:
            print("No content in HyDE LLM response, falling back to original query")
            return query
            
    except Exception as e:
        print(f"Error generating hypothetical document: {e}")
        return query  # Fallback to original query

