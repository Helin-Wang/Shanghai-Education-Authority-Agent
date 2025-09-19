import json
from tqdm import tqdm
from doc.utils import parse_markdown
from doc.chunk import section_to_dict, all_chunks_in_tree
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default="../data/v0_content.json")
    parser.add_argument("--output_tree_structure_filepath", type=str, default="../data/v1_tree_structure.json")
    parser.add_argument("--output_chunks_filepath", type=str, default="../data/v1_chunks.json")
    args = parser.parse_args()
    
    filepath = args.filepath
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Save as tree structure
    processed_tree_structure = []
    processed_chunks = []
    for index, item in tqdm(enumerate(data), total=len(data)):
        raw_markdown = item['markdown']
        doc_metadata = {
            "doc_id": index,
            "title": item['title'],
            "link": item['link'],
            "year": item['year'],
            "category": item['category'],
        }
        root = parse_markdown(raw_markdown, doc_metadata)
        section_dict = section_to_dict(root)
        processed_tree_structure.append(section_dict)
        processed_chunks.extend(all_chunks_in_tree(root))
        
    with open(args.output_tree_structure_filepath, "w", encoding="utf-8") as f:
        json.dump(processed_tree_structure, f, ensure_ascii=False, indent=4)
    
    # Save as chunks
    with open(args.output_chunks_filepath, "w", encoding="utf-8") as f:
        json.dump(processed_chunks, f, ensure_ascii=False, indent=4)