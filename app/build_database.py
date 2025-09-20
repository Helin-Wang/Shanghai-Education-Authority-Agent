import json
from tqdm import tqdm
from doc.utils import parse_markdown
import argparse
from doc.chunk import chunk_to_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default="../data/v0_content.json")
    parser.add_argument("--output_chunks_filepath", type=str, default="../data/v1_chunks.json")
    args = parser.parse_args()
    
    filepath = args.filepath
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    
    processed_chunks = []
    for index, item in tqdm(enumerate(data), total=len(data)):
        raw_markdown = item['markdown']
        doc_metadata = {
            "doc_id": item['doc_id'],
            "title": item['title'],
            "link": item['link'],
            "year": item['year'],
            "category": item['category'],
        }
        chunks = parse_markdown(raw_markdown, doc_metadata)
        chunks_dict = [chunk_to_dict(chunk) for chunk in chunks]
        processed_chunks.extend(chunks_dict)
        
    # Save as chunks
    with open(args.output_chunks_filepath, "w", encoding="utf-8") as f:
        json.dump(processed_chunks, f, ensure_ascii=False, indent=4)