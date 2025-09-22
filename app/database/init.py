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

    conn.commit()
    conn.close()


def create_view(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE VIEW IF NOT EXISTS v_chunks_join AS
        SELECT
        c.chunk_id,
        c.text                         AS page_content,
        cm.doc_id,
        d.title                        AS doc_title,
        d.link                         AS doc_link,
        cm.category,
        cm.year,
        d.published_date,
        d.crawl_time
        FROM chunks c
        JOIN chunk_metadata cm ON cm.chunk_id = c.chunk_id
        JOIN documents d       ON d.doc_id    = cm.doc_id;      
    ''')
    conn.commit()