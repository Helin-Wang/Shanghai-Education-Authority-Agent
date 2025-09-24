import json
from tqdm import tqdm
from doc.utils import parse_markdown
import argparse
from doc.chunk import chunk_to_dict
from database.init import init_table, create_view
import sqlite3
from database.utils import insert_chunk, insert_document, insert_chunk_metadata
from models.M3eEmbedding import M3eEmbeddings
import os
from langchain.schema import Document
from database.index import FAISSIndex, FTS5Index
from database.index import ChromaIndex

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
    
    print("Initializing Database...")
    init_table(db_path)
    embedding_model = M3eEmbeddings()
    
    # Build Document Table
    print("Building Document Table...")
    conn = sqlite3.connect(db_path)
    for item in data:
        insert_document(conn, item['doc_id'], item['title'], item['link'], item['year'], ";".join(item['category']), item['published_date'], item['crawl_time'], item['markdown'])
    
    # Split into chunks
    print("Building Chunk-related Table")
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
        
    # Build Chunk-related Tables
    for chunk in processed_chunks:
        insert_chunk(conn, chunk['metadata']['chunk_index'], chunk['text'])
        insert_chunk_metadata(conn, chunk['metadata']['chunk_index'], chunk['metadata']['doc_id'], ";".join(chunk['metadata']['category']), chunk['metadata']['year'])
        
    create_view(db_path)
    
    with open("../data/v1_chunks.json", "r", encoding="utf-8") as f:
        processed_chunks = json.load(f)
    
    # Build Langchain Documents, use ';'.join(chunk['metadata']['category']) as category
    langchain_documents = [Document(page_content=chunk['text'], metadata=chunk['metadata']) for chunk in processed_chunks]
    for i in range(len(langchain_documents)):
        langchain_documents[i].metadata['category'] = ';'.join(langchain_documents[i].metadata['category'])
    
    # Build Chroma Index
    ChromaIndex(langchain_documents, "../data/chromadb_index")
    
    # Build FTS5 Index
    FTS5Index(conn)
    conn.close()