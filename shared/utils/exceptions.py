"""
Custom exceptions for the AgenticRAG system.
"""

class AgenticRAGError(Exception):
    """Base exception for AgenticRAG system"""
    pass

class DatabaseError(AgenticRAGError):
    """Database-related errors"""
    pass

class EmbeddingError(AgenticRAGError):
    """Embedding-related errors"""
    pass