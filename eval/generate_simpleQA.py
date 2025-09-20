from agents.simpleQA_generator import SimpleQAGenerator
from agents.type_classifier import TypeClassifier
from openai import OpenAI
import json
import pandas as pd
from tqdm import tqdm
import argparse
api_key_r1 = 'sk-hmqokjrhfszsquludqhbdzftggjriimfelvjjqwzccxnqxmn'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qatype_filepath", help="QAPair Type Filepath", default="../data/eval/extracted_qatype_v0.csv")
    parser.add_argument("--input", help="Chunk Filepath", default="../data/v1_chunks.json")
    parser.add_argument("--output", help="Output Filepath", default="../data/eval/simpleQA_v0.json")
    parser.add_argument("--start_index", help="Start Index", default=0)
    parser.add_argument("--end_index", help="End Index", default=None)
    
    args = parser.parse_args()
    qatype_filepath = args.qatype_filepath
    filepath = args.input
    output_filepath = args.output
    start_index = args.start_index
    end_index = args.end_index
    llm_client = OpenAI(
        api_key=api_key_r1,
        base_url="https://api.siliconflow.cn/v1"
    )

    # load possible question types
    with open(qatype_filepath, 'r') as f:
        extracted_qatype = pd.read_csv(f)
    
    predefined_types_dict = dict()
    for index, row in extracted_qatype.iterrows():
        if pd.isna(row['type']):
            continue
        if row['type'] not in predefined_types_dict:
            predefined_types_dict[row['type']] = set()
        else:
            predefined_types_dict[row['type']].add(row['detailed_type'])
    
    type_classifier = TypeClassifier(llm_client=llm_client, predefined_types_dict=predefined_types_dict)
    generator = SimpleQAGenerator(llm_client, predefined_types_dict)
    
    # load chunk
    with open(filepath, 'r') as f:
        chunks = json.load(f)
    
    qa_list = []
    try:
        for index, chunk in tqdm(enumerate(chunks[start_index:end_index]), total=len(chunks[start_index:end_index])):
            type_list = type_classifier.classify_type(chunk)
            
            if type_list is None:
                continue
            else:
                for type_dict in type_list:
                    generated_qapairs = generator.generate_simple_QA(chunk, type_dict['question_type'], type_dict['detailed_types'])
                    if generated_qapairs:
                        for qapair in generated_qapairs:
                            qapair['source_chunk_metadata'] = chunk['metadata']
                            qa_list.append(qapair)
            
    except Exception as e:
        print(f"Error at chunk {index}")
        print(e)
        
    # TODO: deduplicate
    # For qapair from the same year, same doc_category, calculate the similarity
    # 对 query 用 SimHash/向量相似做 0.9 阈值合并，只保留 N 种问法。
    
    
    # Rewrite output filepath
    output_filepath = output_filepath.replace('.json', f'_{start_index}_{end_index}.json')
    with open(output_filepath, 'w') as f:
        json.dump(qa_list, f, ensure_ascii=False, indent=4)
        