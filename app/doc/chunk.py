import re
import hashlib
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field, asdict

@dataclass
class Chunk:
    """Represents a text chunk optimized for RAG"""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

# def section_to_dict(root: Section) -> Dict[str, Any]:
#     d = asdict(root)
#     d['children'] = [section_to_dict(child) for child in root.children]
#     d['chunks'] = [chunk_to_dict(chunk) for chunk in root.chunks]
#     return d

def chunk_to_dict(chunk: Chunk) -> Dict[str, Any]:
    d = asdict(chunk)
    return d

# def all_chunks_in_tree(root: Section) -> List[Chunk]:
#     chunks = []
#     # chunks in current section
#     chunks.extend([chunk_to_dict(chunk) for chunk in root.chunks])
#     # chunks in children sections
#     for child in root.children:
#         chunks.extend(all_chunks_in_tree(child))
#     return chunks