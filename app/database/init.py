import sqlite3

def init_table(db_path):
    conn = sqlite3.connect(db_path)
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
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT UNIQUE,
            chunk_tokens TEXT,
            text TEXT,
            doc_id TEXT,
            doc_title TEXT,
            doc_link TEXT,
            category TEXT,
            year TEXT
        )
    ''')

    conn.commit()
    conn.close()
