"""
Validation utilities for the AgenticRAG system.
"""

from pathlib import Path
from typing import Dict, Any, List
import os
import re

def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate configuration - delegated to config.validator"""
    from shared.config.validator import validate_config as _validate_config
    return _validate_config(config)

def validate_path(path: str) -> bool:
    """
    Validate that a path exists and is accessible.
    
    Args:
        path: File or directory path to validate
        
    Returns:
        True if path is valid and accessible, False otherwise
    """
    try:
        path_obj = Path(path)
        return path_obj.exists() and os.access(path, os.R_OK)
    except Exception:
        return False

def validate_file_path(path: str) -> bool:
    """
    Validate that a file path exists and is readable.
    
    Args:
        path: File path to validate
        
    Returns:
        True if file exists and is readable, False otherwise
    """
    try:
        path_obj = Path(path)
        return path_obj.is_file() and os.access(path, os.R_OK)
    except Exception:
        return False

def validate_directory_path(path: str) -> bool:
    """
    Validate that a directory path exists and is accessible.
    
    Args:
        path: Directory path to validate
        
    Returns:
        True if directory exists and is accessible, False otherwise
    """
    try:
        path_obj = Path(path)
        return path_obj.is_dir() and os.access(path, os.R_OK)
    except Exception:
        return False

def validate_url(url: str) -> bool:
    """Validate a URL format"""
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(url) is not None

def validate_port(port: int) -> bool:
    """Validate a port number"""
    return 1 <= port <= 65535