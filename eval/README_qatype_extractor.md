# Question Type Extraction Agent

This module provides a comprehensive question type extraction agent for Chinese education system FAQs.

## Features

The `QuestionTypeExtractor` agent extracts two levels of information from QA pairs:

1. **Type**: High-level category from predefined list (TIME, PROCESS, ELIGIBILITY, LOCATION, FEES, DOCUMENTS, POLICIES, CONTACT, RESULTS, GENERAL_INFO) or new type if needed
2. **Detailed Type**: Specific subtype in Chinese (e.g., 报名时间, 考试时间, 报名流程, 资格条件)

## Usage

### Basic Usage

```python
from agents.qatype_extractor import QuestionTypeExtractor

# Initialize the extractor
extractor = QuestionTypeExtractor()

# Extract question type from a QA pair
result = extractor.extract_question_type(
    question="20XX年高考大报名时间？",
    answer="考生信息采集时间一般是在每年的10月份...",
    section="高考学考",
    exam_type="秋考"
)

print(f"Type: {result.type}")           # TIME
print(f"Detailed Type: {result.detailed_type}")  # 报名时间
print(f"Confidence: {result.confidence}")  # 0.9
```

### Batch Processing

```python
# Process multiple QA pairs
qa_pairs = [
    {
        "question": "20XX年高考大报名时间？",
        "answer": "考生信息采集时间一般是在每年的10月份...",
        "section": "高考学考",
        "exam_type": "秋考"
    },
    {
        "question": "考生今年高三怎么报名参加20XX年高考？",
        "answer": "本市学籍应届高中阶段毕业生...",
        "section": "高考学考",
        "exam_type": "秋考"
    }
]

results = extractor.batch_extract(qa_pairs)
```

## Question Type Categories

### TIME (时间相关)
- **Detailed Types**: 报名时间, 考试时间, 成绩公布时间, 录取时间, 确认时间, 修改时间
- **Common Fields**: 具体日期, 时间段, 截止时间, 开放时间, 持续时间

### PROCESS (流程相关)
- **Detailed Types**: 报名流程, 考试流程, 录取流程, 确认流程, 申诉流程, 转学流程
- **Common Fields**: 步骤, 要求, 材料, 地点, 方式, 注意事项

### ELIGIBILITY (资格相关)
- **Detailed Types**: 报名资格, 考试资格, 录取资格, 转学资格, 免试资格
- **Common Fields**: 年龄要求, 学历要求, 户籍要求, 成绩要求, 特殊条件

### LOCATION (地点相关)
- **Detailed Types**: 考试地点, 报名地点, 确认地点, 咨询地点
- **Common Fields**: 具体地址, 交通方式, 联系方式, 开放时间

### FEES (费用相关)
- **Detailed Types**: 报名费用, 考试费用, 材料费用, 服务费用
- **Common Fields**: 费用标准, 缴费方式, 缴费时间, 退费政策

### DOCUMENTS (材料相关)
- **Detailed Types**: 报名材料, 考试材料, 录取材料, 转学材料
- **Common Fields**: 材料清单, 格式要求, 提交方式, 审核标准

### POLICIES (政策相关)
- **Detailed Types**: 招生政策, 考试政策, 录取政策, 转学政策
- **Common Fields**: 政策内容, 适用范围, 执行标准, 变更说明

### CONTACT (联系方式相关)
- **Detailed Types**: 咨询方式, 联系方式, 投诉渠道, 服务窗口
- **Common Fields**: 联系电话, 办公地址, 服务时间, 在线渠道

### RESULTS (结果相关)
- **Detailed Types**: 考试成绩, 录取结果, 排名结果, 审核结果
- **Common Fields**: 查询方式, 公布时间, 结果说明, 申诉渠道

### GENERAL_INFO (一般信息)
- **Detailed Types**: 基本信息, 常见问题, 注意事项, 温馨提示
- **Common Fields**: 内容说明, 适用范围, 更新信息, 相关链接

## Output Format

The `QuestionTypeExtractionResult` contains:

- `question`: Original question text
- `answer`: Original answer text
- `type`: High-level category (from predefined list or new type)
- `detailed_type`: Specific subtype in Chinese
- `confidence`: Confidence score (0-1)
- `rationale`: Explanation for the classification

The `FieldInfo` objects contain:
- `name`: Field name in Chinese
- `value`: Actual value extracted from the answer

## Integration with LLM

To use with a language model client:

```python
# Initialize with LLM client
extractor = QuestionTypeExtractor(llm_client=your_llm_client)

# The extractor will automatically use LLM for complex cases
result = extractor.extract_question_type(question, answer)
```