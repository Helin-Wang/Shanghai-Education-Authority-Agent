import hashlib
import sqlite3
import jieba

def sha1_hex(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def insert_document(conn, doc_id: str, title: str, link: str, year: str, category: str, published_date: str, crawl_time: str, markdown: str):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO documents (doc_id, title, link, year, category, published_date, crawl_time, markdown)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (doc_id, title, link, year, category, published_date, crawl_time, markdown))
    conn.commit()
    
def insert_chunk(conn, chunk_id: str, text: str, doc_id: str, doc_title: str, doc_link: str, category: str, year: str):
    cursor = conn.cursor()
    chunk_tokens = " ".join(jieba.cut(text))
    cursor.execute('''
        INSERT INTO chunks (chunk_id, chunk_tokens, text, doc_id, doc_title, doc_link, category, year)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (chunk_id, chunk_tokens, text, doc_id, doc_title, doc_link, category, year))
    conn.commit()


def chroma_filter(years, categories):
    # if only one filter is given, use filter instead of where
    chroma_filter = {}
    if years:
        if len(years) == 1:
            chroma_filter["year"] = years[0]  # ChromaDB uses simple equality
        else:
            chroma_filter["year"] = {"$in": years}  # Keep $in for multiple values
    if categories:
        if len(categories) == 1:
            chroma_filter["category"] = categories[0]  # ChromaDB uses simple equality
        else:
            chroma_filter["category"] = {"$in": categories}  # Keep $in for multiple values
    return chroma_filter

def chroma_search(chroma_db, query, k, chroma_filter):
    if len(chroma_filter) == 1:
        chroma_docs_with_scores = chroma_db.similarity_search_with_score(query, k=k,
                                                                        filter=chroma_filter)
    elif len(chroma_filter) == 0:
        chroma_docs_with_scores = chroma_db.similarity_search_with_score(query, k=k)
    else:
        filter_list = [{key: value} for key, value in chroma_filter.items()]
        chroma_docs_with_scores = chroma_db.similarity_search_with_score(query, k=k,
                                                                        filter={"$and": filter_list})
    return chroma_docs_with_scores