import argparse
import pandas as pd
from agents.qatype_extractor import QuestionTypeExtractor
from openai import OpenAI
from dataclasses import asdict
import os
api_key_r1 = 'sk-hmqokjrhfszsquludqhbdzftggjriimfelvjjqwzccxnqxmn'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="FAQ Filepath")
    parser.add_argument("--output", help="Output Filepath")
    
    args = parser.parse_args()
    filepath = args.file
    output_filepath = args.output
    
    # Check if the output file exists
    if not os.path.exists(output_filepath):
        # Load file
        df = pd.read_csv(filepath)
        df["exam_type"] = df["部分"].astype(str) + "_" + df["考试类型"].astype(str)
        df.drop(columns=["部分", "考试类型"], inplace=True)
        df = df.rename(columns={"问题": "question", "答案": "answer", "问题类型": "section"})
        
        llm_client = OpenAI(
            api_key=api_key_r1,  
            base_url="https://api.siliconflow.cn/v1"
        )
        
        # Initialize the extractor (without LLM client for now)
        extractor = QuestionTypeExtractor(llm_client=llm_client)
        
        # Convert DataFrame to list of dictionaries
        qa_pairs = df.to_dict(orient="records")
        
        # Extract question types
        results = extractor.batch_extract(qa_pairs)
        
        # Save results to CSV
        results_df = pd.DataFrame([asdict(obj) for obj in results])
        results_df.to_csv(output_filepath, index=False)
        
    # Merge type
    results_df['type'] = results_df['type'].str.strip()
    type_list = results_df['type'].unique()
    type_mapping = extractor.aggregate_and_deduplicate_specifictype(type_list)
    # Replace the type with the mapping
    results_df['type'] = results_df['type'].map(type_mapping)
    
    # Regenerate the types that occur only a few times
    type_counts = results_df['type'].value_counts()
    valid_types = type_counts[type_counts >= 30].index
    few_times_types = type_counts[type_counts < 30].index
    for index, row in results_df.iterrows():
        if row['type'] in few_times_types:
            original_row = df[df['问题'] == row['question']]
            ans = extractor._extract_with_llm_given_types(row['question'], row['answer'], original_row['部分'], original_row['考试类型'], valid_types)
            results_df.loc[index, 'type'] = ans.type
            results_df.loc[index, 'detailed_type'] = ans.detailed_type
            results_df.loc[index, 'confidence'] = ans.confidence
            results_df.loc[index, 'rationale'] = ans.rationale
    results_df.to_csv(output_filepath, index=False)