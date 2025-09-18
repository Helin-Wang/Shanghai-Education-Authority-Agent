import re
import hashlib
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field

@dataclass
class Chunk:
    """Represents a text chunk optimized for RAG"""
    text: str
    start_char: int
    end_char: int
    chunk_index: int
    section_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Section:
    """Section with smart chunking for RAG"""
    level: int
    title: str
    anchor: str
    start_line: int
    end_line: int
    text: str
    breadcrumb: List[str]
    id: str
    chunks: List[Chunk] = field(default_factory=list)
    children: List["Section"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
