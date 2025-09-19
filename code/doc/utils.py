from doc.chunk import Section, Chunk
import re
import hashlib
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from langchain.docstore.document import Document

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
        # # 处理围栏代码块开合（奇偶次切换）
        # if _is_code_fence(line):
        #     in_code = not in_code
        #     i += 1
        #     continue
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

def split_into_chunk(text: str, 
                                max_chunk_size: int = 500, 
                                min_chunk_size: int = 100,
                                overlap_size: int = 50,
                                section_id: str = "",
                                preserve_sentences: bool = True) -> List[Chunk]:
    """
    Chunk markdown text with tables as atomic units.
    
    Args:
        text: The text to chunk
        max_chunk_size: Maximum characters per chunk
        min_chunk_size: Minimum characters per chunk
        overlap_size: Number of characters to overlap between chunks
        section_id: ID of the parent section
        preserve_sentences: Whether to preserve sentence boundaries
        
    Returns:
        List of Chunk objects
    """
    # if len(text) <= max_chunk_size:
    #     return [Chunk(
    #         text=text.strip(),
    #         start_char=0,
    #         end_char=len(text),
    #         chunk_index=0,
    #         section_id=section_id,
    #         metadata={"chunk_type": "single", "length": len(text), "contains_table": False}
    #     )]
    
    # Detect tables in the text
    table_boundaries = detect_tables_in_text(text)
    
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(text):
        # Determine the end position for this chunk
        end = start + max_chunk_size
        
        if end >= len(text):
            # Last chunk
            chunk_text = text[start:].strip()
            if len(chunk_text) >= min_chunk_size:
                # Check if this chunk contains a table
                # TODO: modify table format
                # contains_table = any(start <= table_start < len(text) and start <= table_end <= len(text) 
                #                    for table_start, table_end in table_boundaries)
                
                chunks.append(Chunk(
                    text=chunk_text,
                    start_char=start,
                    end_char=len(text),
                    chunk_index=chunk_index,
                    section_id=section_id,
                    metadata={"chunk_type": "final", "length": len(chunk_text)}
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
            chunk_text = text[start:table_end].strip()
            end = table_end
        else:
            # Normal chunking logic
            chunk_text = text[start:end]
            
            if preserve_sentences:
                # Find sentence boundaries
                sentence_endings = ['.', '。', '!', '！', '?', '？', '\n\n']
                best_break = end
                
                # Look for sentence endings in the last part of the chunk
                search_start = max(start + min_chunk_size, end - 100)
                for i in range(end - 1, search_start - 1, -1):
                    if text[i] in sentence_endings:
                        best_break = i + 1
                        break
                
                # If no sentence boundary found, look for other natural breaks
                if best_break == end:
                    # Look for paragraph breaks, commas, or other punctuation
                    natural_breaks = [',', '，', '\n', ';', '；', ':', '：']
                    for i in range(end - 1, search_start - 1, -1):
                        if text[i] in natural_breaks:
                            best_break = i + 1
                            break
                
                chunk_text = text[start:best_break].strip()
                end = best_break
            else:
                chunk_text = chunk_text.strip()
            
            # Check if this chunk contains a table
            # TODO: modify table format
            # contains_table = any(start <= table_start < end and start <= table_end <= end 
            #                    for table_start, table_end in table_boundaries)
        
        # Ensure minimum chunk size
        if len(chunk_text) >= min_chunk_size:
            chunks.append(Chunk(
                text=chunk_text,
                start_char=start,
                end_char=end,
                chunk_index=chunk_index,
                section_id=section_id,
                metadata={"chunk_type": "normal", "length": len(chunk_text)}
            ))
            chunk_index += 1
        
        # Move start position with overlap
        start = max(start + 1, end - overlap_size)
    
    return chunks

def enrich_chunk_metadata(chunk: Chunk, section: Section, 
                         doc_metadata: Dict[str, Any] = None) -> Chunk:
    """
    Enrich chunk metadata with additional context for better RAG performance.
    
    Args:
        chunk: The chunk to enrich
        section: The parent section
        doc_metadata: Document-level metadata
        
    Returns:
        Enriched chunk
    """
    enriched_metadata = chunk.metadata.copy()
    
    # Add section context
    enriched_metadata.update({
        "section_title": section.title,
        "section_breadcrumb": " > ".join(section.breadcrumb),
        "section_anchor": section.anchor,
        "chunk_position": f"{chunk.chunk_index + 1}/{len(section.chunks)}" if section.chunks else "1/1"
    })
    
    # Add document metadata if available
    if doc_metadata:
        enriched_metadata.update({
            "doc_title": doc_metadata.get("title", ""),
            "doc_year": doc_metadata.get("year", ""),
            "doc_category": doc_metadata.get("category", ""),
            "doc_link": doc_metadata.get("link", "")
        })
    
    # TODO: Add content analysis
    # enriched_metadata.update({
    #     "word_count": len(chunk.text.split()),
    #     "char_count": len(chunk.text),
    #     "has_numbers": bool(re.search(r'\d', chunk.text)),
    #     "has_dates": bool(re.search(r'\d{4}年|\d{1,2}月|\d{1,2}日', chunk.text)),
    #     "has_urls": bool(re.search(r'http[s]?://', chunk.text)),
    #     "sentence_count": len(re.split(r'[。！？]', chunk.text)) - 1
    # })
    
    chunk.metadata = enriched_metadata
    return chunk


def parse_markdown(md_text: str, 
                   doc_metadata: Dict[str, Any] = None,
                   max_chunk_size: int = 500,
                   min_chunk_size: int = 100,
                   overlap_size: int = 50,
                chunk_sections: bool = True) -> Section:
    """
    Build an enhanced section tree with smart chunking that preserves tables as atomic units.
    
    Args:
        md_text: The markdown text to build the section tree from
        doc_metadata: Document-level metadata to enrich chunks
        max_chunk_size: Maximum characters per chunk
        min_chunk_size: Minimum characters per chunk
        overlap_size: Number of characters to overlap between chunks
        chunk_sections: Whether to chunk sections that are too long
        
    Returns:
        Returns an enhanced root section whose children are top-level sections with chunks
    """
    from markdown_it import MarkdownIt
    
    # Parse the markdown text into tokens
    md = MarkdownIt()
    tokens = md.parse(md_text)
    lines = md_text.splitlines(keepends=True)
    
    # Collect all headings and their start lines
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
    bounds: List[Tuple[int, int]] = []  # (start_line, end_line) per heading
    for idx, (_, _, start) in enumerate(headings):
        next_start = headings[idx + 1][2] if idx + 1 < len(headings) else doc_line_count
        bounds.append((start, next_start))
    
    # Build tree with a stack by heading level
    root = Section(
        level=0, title="", anchor="", start_line=0, end_line=headings[0][2] if headings else doc_line_count,
        chunks=[], breadcrumb=[], id="root", children=[], metadata={}
    )
    stack: List[Section] = [root]
    
    # Add chunk for text from beginning to the first heading
    chunks = split_into_chunk(
        ''.join(lines[0:root.end_line]),
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size,
        overlap_size=overlap_size,
        section_id=root.id,
        preserve_sentences=True
    )
    for chunk in chunks:
        chunk = enrich_chunk_metadata(chunk, root, doc_metadata)
        root.chunks.append(chunk)
    root.metadata = {
        "total_chunks": len(chunks),
        "total_length": len(''.join(lines[0:root.end_line])),
    }

    # Add chunks for each section
    for (level, title, start_line), (s, e) in zip(headings, bounds):
        # body starts after the heading line; end before next heading
        body_start = start_line + 1
        body_end = e
        body_text = ''.join(lines[body_start:body_end])
        
        # Preprocess the text for RAG
        processed_text = body_text
        
        # ascend/descend to the right parent
        while stack and level <= stack[-1].level:
            stack.pop()
        parent = stack[-1] if stack else root
        
        breadcrumb = parent.breadcrumb + [title]
        anchor = slugify(title)
        section_id = stable_id(*breadcrumb)
        
        # Create chunks if the section is long enough and chunking is enabled
        chunks = []
        chunks = split_into_chunk(
            processed_text,
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
            overlap_size=overlap_size,
            section_id=section_id,
            preserve_sentences=True
        )
       
        # Create section metadata
        section_metadata = {
            "total_chunks": len(chunks),
            "total_length": len(processed_text),
            # "has_content": len(processed_text.strip()) > 0,
            # "chunking_applied": chunk_sections and len(processed_text) > max_chunk_size,
            # "contains_tables": len(detect_tables_in_text(processed_text)) > 0,
            # "table_count": len(detect_tables_in_text(processed_text))
        }
        
        sec = Section(
            level=level,
            title=title,
            anchor=anchor,
            start_line=body_start,
            end_line=body_end,
            chunks=[],
            breadcrumb=breadcrumb,
            id=section_id,
            children=[],
            metadata=section_metadata
        )
        
        # Enrich chunk metadata
        for chunk in chunks:
            # Update chunk section references
            chunk.section_id = section_id
            chunk = enrich_chunk_metadata(chunk, sec, doc_metadata)
            sec.chunks.append(chunk)
        
        parent.children.append(sec)
        stack.append(sec)
    
    return root


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



