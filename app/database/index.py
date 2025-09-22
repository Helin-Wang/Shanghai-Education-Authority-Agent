from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from typing import List
from models.M3eEmbedding import M3eEmbeddings
import jieba

def FTS5Index(conn):
    cursor = conn.cursor()
    cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_id UNINDEXED,
            seg, 
            prefix='2,3,4',
            tokenize="unicode61"
        )
    ''')
    
    # Retrieve data
    cursor.execute('''
        SELECT chunk_id, text FROM chunks;
    ''')
    rows = cursor.fetchall()
    
    # Tokenize and insert data
    for row in rows:
        chunk_id, text = row
        seg = " ".join(jieba.cut(text))
        cursor.execute('''
            INSERT INTO chunks_fts (chunk_id, seg)
            VALUES (?, ?);
        ''', (chunk_id, seg))
    
    # Trigger
    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks
        BEGIN
            INSERT INTO chunks_fts (chunk_id, seg)
            VALUES (NEW.chunk_id, NEW.seg);
        END;
    ''')
    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS trg_chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, seg)
            VALUES('delete', (SELECT rowid FROM chunks_fts WHERE chunk_id=old.chunk_id),
                    old.chunk_id, old.seg);
        END;
    ''')
    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS trg_chunks_au AFTER UPDATE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, seg)
        VALUES('delete', (SELECT rowid FROM chunks_fts WHERE chunk_id=old.chunk_id),
                old.chunk_id, old.seg);
        INSERT INTO chunks_fts(chunk_id, seg)
        VALUES (new.chunk_id, new.seg);
        END;
    ''')
    conn.commit()


def FAISSIndex(docs: List[Document], db_path: str):
    embedding_model = M3eEmbeddings()
    
    # Build FAISS index
    faiss_db = FAISS.from_documents(docs, embedding=embedding_model)
    
    # Save FAISS index
    faiss_db.save_local(db_path)
    