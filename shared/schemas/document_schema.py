"""
Document Schema Definitions for AgenticRAG
"""

from typing import Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DocumentMetadata:
    """Metadata for a document"""
    id: str
    file_path: str
    file_name: str
    folder_path: str
    file_size: int
    created_at: datetime
    updated_at: datetime
    total_chunks: int

@dataclass
class ChunkMetadata:
    """Metadata for a chunk"""
    id: str
    document_id: str
    chunk_number: int
    chunk_text: str
    token_count: int
    overlap_tokens: int
    previous_chunk_id: Optional[str]
    next_chunk_id: Optional[str]
    created_at: datetime