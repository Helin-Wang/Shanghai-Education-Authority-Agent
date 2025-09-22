import json
from tqdm import tqdm
from doc.utils import parse_markdown
import argparse
from doc.chunk import chunk_to_dict
from database.init import init_table
import sqlite3
from database.utils import insert_chunk, insert_document, insert_chunk_metadata, insert_chunk_embedding
from models.M3eEmbedding import M3eEmbeddings
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default="../data/v0_content.json")
    parser.add_argument("--output_chunks_filepath", type=str, default="../data/v1_chunks.json")
    args = parser.parse_args()
    
    filepath = args.filepath
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    db_path = '../data/shanghai_education_authority_agent.db'
    if os.path.exists(db_path):
        os.remove(db_path)
    init_table()
    
    conn = sqlite3.connect(db_path)
    embedding_model = M3eEmbeddings()
    
    # Build Document Table
    for item in data:
        insert_document(conn, item['doc_id'], item['title'], item['link'], item['year'], ";".join(item['category']), item['published_date'], item['crawl_time'], item['markdown'])
    
    # Split into chunks
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
        for chunk in chunks_dict:
            chunk['doc_id'] = item['doc_id']
            chunk['category'] = ";".join(item['category'])
            chunk['year'] = item['year']
        processed_chunks.extend(chunks_dict)
        
    # Save as chunks
    with open(args.output_chunks_filepath, "w", encoding="utf-8") as f:
        json.dump(processed_chunks, f, ensure_ascii=False, indent=4)
        
    # Build Chunk-related Tables
    for chunk in processed_chunks:
        chunk['embedding'] = embedding_model.embed_text(chunk['text'])
        insert_chunk(conn, chunk['chunk_id'], chunk['text'])
        insert_chunk_metadata(conn, chunk['chunk_id'], chunk['doc_id'], chunk['category'], chunk['year'])
        insert_chunk_embedding(conn, chunk['chunk_id'], chunk['embedding'])
    conn.close()