"""
Serialization utilities for the AgenticRAG system.
"""

from datetime import datetime
from typing import Any

def serialize_datetime(obj: Any) -> Any:
    """Serialize datetime objects to ISO format strings"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: serialize_datetime(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_datetime(item) for item in obj]
    else:
        return obj

def deserialize_datetime(obj: Any) -> Any:
    """Deserialize ISO format strings back to datetime objects"""
    if isinstance(obj, str):
        try:
            return datetime.fromisoformat(obj)
        except:
            return obj
    elif isinstance(obj, dict):
        return {k: deserialize_datetime(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deserialize_datetime(item) for item in obj]
    else:
        return obj