"""
Embedding Schema Definitions for AgenticRAG
"""

from typing import List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid

@dataclass
class EmbeddingJob:
    """A job for embedding generation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    chunk_id: str = ""
    text: str = ""
    model: str = "granite-embedding:latest"
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class EmbeddingResult:
    """Result from embedding generation"""
    job_id: str
    chunk_id: str
    embedding: List[float]
    model: str
    processing_time: float
    success: bool = True
    error: Optional[str] = None