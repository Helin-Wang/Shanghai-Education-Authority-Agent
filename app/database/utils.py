import hashlib
import sqlite3

def sha1_hex(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def insert_document(doc_id: str, title: str, link: str, year: str, category: str, published_date: str, crawl_time: str, markdown: str):
    conn = sqlite3.connect('../data/shanghai_education_authority_agent.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO documents (doc_id, title, link, year, category, published_date, crawl_time, markdown)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (doc_id, title, link, year, category, published_date, crawl_time, markdown))
    conn.commit()
    conn.close()
    
def insert_chunk(chunk_id: str, text: str):
    conn = sqlite3.connect('../data/shanghai_education_authority_agent.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO chunks (chunk_id, text)
        VALUES (?, ?)
    ''', (chunk_id, text))
    conn.commit()
    conn.close()

def insert_chunk_metadata(chunk_id: str, doc_id: str, category: str, year: str):
    conn = sqlite3.connect('../data/shanghai_education_authority_agent.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO chunk_metadata (chunk_id, doc_id, category, year)
        VALUES (?, ?, ?, ?)
    ''', (chunk_id, doc_id, category, year))
    conn.commit()
    conn.close()

def insert_chunk_embedding(chunk_id: str, embedding: bytes):
    conn = sqlite3.connect('../data/shanghai_education_authority_agent.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO chunk_embedding (chunk_id, embedding)
        VALUES (?, ?)
    ''', (chunk_id, embedding))
    conn.commit()
    conn.close()