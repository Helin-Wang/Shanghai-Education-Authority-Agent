from calendar import c
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from typing import List
from models.M3eEmbedding import M3eEmbeddings
import jieba
from langchain_community.vectorstores import Chroma

def FTS5Index(conn):
    cursor = conn.cursor()
    
    # Create chunks_fts table
    cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
        USING fts5(
            chunk_tokens,
            doc_title UNINDEXED,
            category UNINDEXED,
            year UNINDEXED,
            content='chunks',
            content_rowid='id',
            tokenize="unicode61" -- 这里不做复杂分词，直接按空格切
        );
    ''')
    
    # Insert data into chunks_fts (only if table is empty)
    cursor.execute('''
        INSERT INTO chunks_fts(rowid, chunk_tokens, doc_title, category, year)
        SELECT id, chunk_tokens, doc_title, category, year FROM chunks
        WHERE NOT EXISTS (SELECT 1 FROM chunks_fts LIMIT 1);
    ''')
    
    # Create trigger to update chunks_fts when chunks is inserted
    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, chunk_tokens, doc_title, category, year)
            VALUES (new.id, new.chunk_tokens, new.doc_title, new.category, new.year);
        END;
    ''')
    
    # Create trigger to update chunks_fts when chunks is deleted
    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid) VALUES('delete', old.id);
        END;
    ''')
    
    # Create trigger to update chunks_fts when chunks is updated
    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid) VALUES('delete', old.id);
            INSERT INTO chunks_fts(rowid, chunk_tokens, doc_title, category, year)
            VALUES (new.id, new.chunk_tokens, new.doc_title, new.category, new.year);
        END;
    ''')
    
    conn.commit()

def bm25_search(conn, query: str):
    # Tokenize query
    query_seg = " ".join(jieba.cut(query))

    cursor = conn.cursor()
    cursor.execute('''
        SELECT
        c.text,
        c.doc_id,
        c.doc_title,
        c.doc_link,
        c.category,
        c.year,
        bm25(chunks_fts) AS score
        FROM chunks_fts
        JOIN chunks c ON c.id = chunks_fts.rowid
        WHERE chunks_fts MATCH ?
        ORDER BY score
        LIMIT 20;
    ''', (query_seg,))
    rows = cursor.fetchall()
    
    # convert to documents with proper metadata structure
    docs = []
    for row in rows:
        doc = Document(
            page_content=row[0],  # text
            metadata={
                'doc_id': row[1],
                'doc_title': row[2], 
                'doc_link': row[3],
                'category': row[4],
                'year': row[5]
            }
        )
        docs.append([doc, row[6]])
    return docs


def FAISSIndex(docs: List[Document], db_path: str):
    embedding_model = M3eEmbeddings()
    
    # Build FAISS index
    faiss_db = FAISS.from_documents(docs, embedding=embedding_model)
    
    # Save FAISS index
    faiss_db.save_local(db_path)
    