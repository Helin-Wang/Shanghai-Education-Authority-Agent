import sqlite3

def init_table(conn):
    cursor = conn.cursor()
    # -- documents
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT,
            link TEXT,
            year TEXT,
            category TEXT,
            published_date TEXT,
            crawl_time TEXT,
            markdown TEXT
        )
    ''')
        
    # --chunks  
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            text TEXT
        )
    ''')

    # --chunk metadata
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunk_metadata (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT,
            category TEXT,
            year TEXT
        )
    ''')

    # --chunk embedding
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunk_embedding (
            chunk_id TEXT PRIMARY KEY,
            embedding BLOB
        )
    ''')
    
    conn.commit()

