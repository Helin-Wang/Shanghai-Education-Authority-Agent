from doc.chunk import Chunk
import re
import hashlib
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from langchain.docstore.document import Document
from markdown_it import MarkdownIt

# GFM 风格的“分隔行”判定：---、:---、---:、:---:，多列用 | 分隔；首尾 | 可选
_SEP_RE = re.compile(
    r"""^\s*\|?\s*
        (?:\:?\-{3,}\:?\s*\|)+      # 至少一段  ---  并且后面跟一个 |
        \s*\:?\-{3,}\:?\s*          # 最后一段 ---
        \|?\s*$                     # 可选结尾 |
    """,
    re.VERBOSE
)
# “像一行表格”的判定：含 |，不是全空白；首尾 | 可选（宽松）
_ROW_HAS_PIPE_RE = re.compile(r"\|")

def detect_tables_in_text(text: str) -> List[Tuple[int, int]]:
    """
    Detect table boundaries in markdown text by character index .
    
    Args:
        text: The text to analyze
        
    Returns:
        List of (start_char_index, end_char_index) tuples for each table found
    """
    lines = text.split("\n")
    idx_of_line = [0]
    for i, ln in enumerate(lines[:-1]):
        idx_of_line.append(idx_of_line[-1] + len(ln) + 1)  # +1 是换行符


    in_code = False
    boundaries: List[Tuple[int, int]] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        if in_code:
            i += 1
            continue

        # 尝试匹配：表头行 + 分隔行
        # 1) 当前行“像表头”（含 |）
        if _ROW_HAS_PIPE_RE.search(line) and line.strip():
            # 2) 下一行存在并且是分隔行
            if i + 1 < n and _SEP_RE.match(lines[i + 1] or ""):
                # 确认是一个表格块，从 i（表头）开始
                start_line = i
                j = i + 2  # 从第一行数据行开始向后合并
                while j < n:
                    # 连续的数据行：宽松判定“含 | 且非空”
                    if _ROW_HAS_PIPE_RE.search(lines[j]) and lines[j].strip():
                        j += 1
                    else:
                        break
                end_line = j  # j 指向表格后的第一行

                # 计算字符区间
                start_char = idx_of_line[start_line]
                end_char = idx_of_line[end_line] if end_line < n else len(text)
                boundaries.append((start_char, end_char))

                # 跳过已消费的表格块
                i = end_line
                continue

        i += 1

    return boundaries

def normalize_headings(headings):
    if len(headings) == 0:
        return []
    for level in range(1, 4):
        # Check if the level is present in the headings
        if any(heading[0] == level for heading in headings):
            continue
        # If the level is not present, upgrade all the levels above it
        gap = min([heading[0] - level if heading[0] > level else 4 for heading in headings])
        for i, heading in enumerate(headings):
            if heading[0] > level:
                headings[i] = (heading[0] - gap, heading[1], heading[2], heading[3])
    return headings

def extract_headings_from_mdtext(md_text: str) -> List[Tuple[int, str, int, int]]:
    md = MarkdownIt()
    tokens = md.parse(md_text)
    lines = md_text.splitlines(keepends=True)
    
    headings: List[Tuple[int, str, int]] = []  # (level, title, start_line)
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.type == 'heading_open':
            level = int(token.tag[1])
            title = ''
            if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                title = tokens[i + 1].content.strip()
            start_line = token.map[0] if token.map else 0
            headings.append((level, title, start_line))
            # skip to heading_close
            i += 1
            while i < len(tokens) and tokens[i].type != 'heading_close':
                i += 1
        i += 1
    
    # Add a sentinel "end" to compute section boundaries
    doc_line_count = len(lines)
    for idx, (_, _, start) in enumerate(headings):
        next_start = doc_line_count
        # search next heading with same level
        for i in range(idx + 1, len(headings)):
            if headings[i][0] <= headings[idx][0]:
                next_start = headings[i][2]
                break
        #next_start = headings[idx + 1][2] if idx + 1 < len(headings) else doc_line_count
        headings[idx] = headings[idx] + (next_start,)
    
    # Ensure the level starts from 1 and decreases gradually
    headings = normalize_headings(headings)
    
    return headings

def extract_targeted_headings(headings: List[Tuple[int, str, int]], 
                              target_level: int) -> List[Tuple[int, str, int]]:
    return [heading for heading in headings if heading[0] == target_level]


def split_into_chunks(md_text: str, 
                      headings: List[Tuple[int, str, int, int]],
                      current_heading_text: str = "",
                      start_line: int = 0,
                      end_line: int = None,
                      current_level: int = 0,
                      min_chunk_size: int = 100,
                      max_chunk_size: int = 400
                    ) -> List[Chunk]:
    
    # Calculate the length of the current text
    current_text_lines = md_text.splitlines(keepends=True)[start_line:end_line]
    current_text = ''.join(current_text_lines)
    content_length = len(current_text)
    
    if content_length < max_chunk_size:
        return [Chunk(
            text=current_text,
            metadata={
                "level": current_level
            }
        )]
    
    # if there is no higher level heading, we need to split the chunk
    target_level = current_level + 1
    extracted_headings = extract_targeted_headings(headings, target_level)    
    if not extracted_headings:
        # NEVER split the table and complete sentence into different chunks
        
        table_boundaries = detect_tables_in_text(current_text)
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(current_text):
            # Determine the end position for this chunk
            end = start + max_chunk_size
            
            if end >= len(current_text):
                # Last chunk
                chunk_text = current_text[start:].strip()
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={
                        "level": current_level
                    }
                ))
                break
            
            # Check if we're about to split a table
            table_to_split = None
            for table_start, table_end in table_boundaries:
                if start < table_start < end < table_end:
                    # We would split this table - extend chunk to include entire table
                    table_to_split = (table_start, table_end)
                    break
            if table_to_split:
                # Extend chunk to include the entire table
                table_start, table_end = table_to_split
                chunk_text = current_text[start:table_end].strip()
                end = table_end
            else:
                # Normal chunking logic
                chunk_text = current_text[start:end]
                
                # Find sentence boundaries
                sentence_endings = ['.', '。', '!', '！', '?', '？', '\n\n']
                best_break = end
                
                # Look for sentence boundaries in the last part of the chunk
                search_start = max(start + 100, end - 100)
                for i in range(end - 1, search_start - 1, -1):
                    if current_text[i] in sentence_endings:
                        best_break = i + 1
                        break
                
                # If no sentence boundary found, look for other natural breaks
                if best_break == end:
                    # Look for paragraph breaks, commas, or other punctuation
                    natural_breaks = [',', '，', '\n', ';', '；', ':', '：']
                    for i in range(end - 1, search_start - 1, -1):
                        if current_text[i] in natural_breaks:
                            best_break = i + 1
                            break
                
                chunk_text = current_text[start:best_break].strip()
                end = best_break
                
            chunks.append(Chunk(
                text=chunk_text,
                metadata={
                    "level": current_level
                }
            ))
            start = end
            chunk_index += 1
            
        return chunks
                        
    else:
        sub_chunks = []
        # Before first heading
        if current_level == 0:
            sub_chunks.append(Chunk(
                text=''.join(current_text_lines[start_line:extracted_headings[0][2]]),
                metadata={
                    "level": current_level
                }
            ))
        
        # Filter headings to only include those within the current range
        relevant_headings = []
        for heading in extracted_headings:
            heading_start = heading[2]
            if heading_start >= start_line and (end_line is None or heading_start < end_line) and heading[0] == target_level:
                relevant_headings.append(heading)
        
        # Process headings in order, ensuring no overlap
        for i, heading in enumerate(relevant_headings):
            sub_start_line = heading[2]
            # Determine the end line for this heading
            if i + 1 < len(relevant_headings):
                sub_end_line = relevant_headings[i + 1][2]
            else:
                sub_end_line = end_line
            
            sub_chunks.extend(split_into_chunks(md_text, headings, heading[1], sub_start_line, sub_end_line, target_level))
        return sub_chunks

def merge_chunks(chunks: List[Chunk], max_chunk_size: int = 400) -> List[Chunk]:
    merged_chunks = []
    current_level = -1
    current_chunk = None
    for chunk in chunks:
        if chunk.metadata["level"] != current_level:
            if current_chunk:
                merged_chunks.append(current_chunk)
            current_level = chunk.metadata["level"]
            current_chunk = chunk
        else:
            if current_chunk and len(current_chunk.text) + len(chunk.text) <= max_chunk_size:
                current_chunk.text += '\n' + chunk.text
            else:
                if current_chunk:
                    merged_chunks.append(current_chunk)
                current_chunk = chunk
    if current_chunk:
        merged_chunks.append(current_chunk)
    return merged_chunks

def postprocess_chunks(chunks: List["Chunk"], doc_metadata: Dict[str, Any], overlap_size: int = 100):
    postprocessed_chunks = []
    
    for index, chunk in enumerate(chunks):
        postprocessed_chunk = Chunk(
            text="",
            metadata={}
        )

        # Chunk index
        postprocessed_chunk.metadata["chunk_index"] = index

        # Doc info
        postprocessed_chunk.metadata.update({
            "doc_id": doc_metadata.get("doc_id"),
            "doc_title": doc_metadata.get("title"),
            "doc_link": doc_metadata.get("link"),
            "category": doc_metadata.get("category"),
            "year": doc_metadata.get("year"),
        })

        # Start with current chunk text
        text_parts = [chunk.text]

        # Prefix overlap (previous chunk tail)
        if index > 0:
            prefix = chunks[index - 1].text[-overlap_size:]
            text_parts.insert(0, prefix)  # put before current text

        # Suffix overlap (next chunk head)
        if index < len(chunks) - 1:
            suffix = chunks[index + 1].text[:overlap_size]
            text_parts.append(suffix)

        # Join with line breaks
        postprocessed_chunk.text = "\n".join(text_parts)

        postprocessed_chunks.append(postprocessed_chunk)

    return postprocessed_chunks

def parse_markdown(md_text: str, doc_metadata: Dict[str, Any]) -> List[Chunk]:
    headings = extract_headings_from_mdtext(md_text)
    chunks = split_into_chunks(md_text, headings)
    merged_chunks = merge_chunks(chunks)
    merged_chunks = postprocess_chunks(merged_chunks, doc_metadata, overlap_size=50)
    return merged_chunks


"""
--------------------------------
--------------------------------
"""


def slugify(title: str) -> str:
    """GitHub-ish slugify (simple and deterministic)."""
    s = title.strip().lower()
    s = re.sub(r'[^\w\- ]+', '', s)      # remove punctuation except - and space
    s = re.sub(r'\s+', '-', s)           # spaces -> hyphens
    s = re.sub(r'-{2,}', '-', s)         # collapse ---
    return s.strip('-')

def stable_id(*parts: str, limit: int = 16) -> str:
    h = hashlib.sha1('||'.join(parts).encode('utf-8')).hexdigest()
    return h[:limit]


def convert_chunks_to_documents(chunks: List[Chunk], 
                               base_metadata: Dict[str, Any] = None) -> List[Document]:
    """
    Convert chunks to LangChain Document objects for RAG systems.
    
    Args:
        chunks: List of Chunk objects
        base_metadata: Base metadata to add to all documents
        
    Returns:
        List of Document objects
    """
    documents = []
    for chunk in chunks:
        # Combine base metadata with chunk metadata
        metadata = base_metadata.copy() if base_metadata else {}
        metadata.update(chunk.metadata)
        
        # Add chunk-specific metadata
        metadata.update({
            "chunk_id": f"{chunk.section_id}_chunk_{chunk.chunk_index}",
            "chunk_start": chunk.start_char,
            "chunk_end": chunk.end_char,
            "chunk_size": len(chunk.text)
        })
        
        documents.append(Document(
            page_content=chunk.text,
            metadata=metadata
        ))
    
    return documents




