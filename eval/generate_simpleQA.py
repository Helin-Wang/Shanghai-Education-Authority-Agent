from agents.simpleQA_generator import SimpleQAGenerator
from agents.type_classifier import TypeClassifier
from openai import OpenAI
import json
import pandas as pd
from tqdm import tqdm

api_key_r1 = 'sk-hmqokjrhfszsquludqhbdzftggjriimfelvjjqwzccxnqxmn'

if __name__ == "__main__":
    llm_client = OpenAI(
        api_key=api_key_r1,
        base_url="https://api.siliconflow.cn/v1"
    )

    # load possible question types
    with open('../data/eval/extracted_qatype_v0.csv', 'r') as f:
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
    with open('../data/v1_chunks.json', 'r') as f:
        chunks = json.load(f)
    
    qa_list = []
    try:
        for index, chunk in tqdm(enumerate(chunks), total=len(chunks)):
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
            break
    except Exception as e:
        print(f"Error at chunk {index}")
        print(e)
        
    # TODO: deduplicate
    
    
    
    with open('../data/eval/simpleQA_v0.json', 'w') as f:
        json.dump(qa_list, f, ensure_ascii=False, indent=4)
        