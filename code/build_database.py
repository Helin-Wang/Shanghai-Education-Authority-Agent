import json
from tqdm import tqdm
from doc.utils import parse_markdown
from doc.chunk import section_to_dict
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default="../data/v0_content.json")
    parser.add_argument("--output_filepath", type=str, default="../data/v1_tree_structure.json")
    args = parser.parse_args()
    
    filepath = args.filepath
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    processed_data = []
    for index, item in tqdm(enumerate(data), total=len(data)):
        raw_markdown = item['markdown']
        doc_metadata = {
            "title": item['title'],
            "link": item['link'],
            "year": item['year'],
            "category": item['category'],
        }
        root = parse_markdown(raw_markdown, doc_metadata)
        section_dict = section_to_dict(root)
        processed_data.append(section_dict)
        
    with open(args.output_filepath, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)