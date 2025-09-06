"""
Comprehensive validation utilities for the AgenticRAG system.

Provides robust validation for paths, URLs, configurations, and data types
with detailed error handling and comprehensive edge case coverage.
"""

from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple
import os
import re
import logging
from urllib.parse import urlparse
import ipaddress

logger = logging.getLogger(__name__)

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

def validate_url(url: str) -> Tuple[bool, Optional[str]]:
    """
    Comprehensive URL validation with detailed error reporting.
    
    Args:
        url: URL string to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not url or not isinstance(url, str):
        return False, "URL must be a non-empty string"
    
    try:
        parsed = urlparse(url.strip())
        
        # Check scheme
        if not parsed.scheme or parsed.scheme not in ['http', 'https']:
            return False, "URL must use http or https scheme"
        
        # Check hostname/netloc
        if not parsed.netloc:
            return False, "URL must contain a valid hostname"
        
        # Validate hostname part
        hostname = parsed.hostname
        if hostname:
            # Check if it's an IP address
            try:
                ipaddress.ip_address(hostname)
            except ValueError:
                # Not an IP, validate as hostname
                if not _is_valid_hostname(hostname):
                    return False, f"Invalid hostname: {hostname}"
        
        # Validate port if present
        if parsed.port is not None:
            is_valid_port, port_error = validate_port(parsed.port)
            if not is_valid_port:
                return False, f"Invalid port: {port_error}"
        
        return True, None
        
    except Exception as e:
        return False, f"URL parsing error: {str(e)}"

def _is_valid_hostname(hostname: str) -> bool:
    """Validate hostname according to RFC standards"""
    if len(hostname) > 253:
        return False
    
    # Check for localhost
    if hostname.lower() == 'localhost':
        return True
    
    # Validate each label
    labels = hostname.split('.')
    for label in labels:
        if not label or len(label) > 63:
            return False
        if not re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$', label):
            return False
    
    return True

def validate_port(port: Union[int, str]) -> Tuple[bool, Optional[str]]:
    """
    Validate a port number with detailed error reporting.
    
    Args:
        port: Port number (int or string)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        port_num = int(port)
        if port_num < 1:
            return False, "Port must be greater than 0"
        if port_num > 65535:
            return False, "Port must be less than 65536"
        return True, None
    except (ValueError, TypeError):
        return False, f"Port must be a valid integer, got {type(port).__name__}"

def validate_email(email: str) -> Tuple[bool, Optional[str]]:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not email or not isinstance(email, str):
        return False, "Email must be a non-empty string"
    
    email = email.strip()
    
    # Basic RFC 5322 compliant regex
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(pattern, email):
        return False, "Invalid email format"
    
    # Additional checks
    local, domain = email.rsplit('@', 1)
    
    if len(local) > 64:
        return False, "Email local part too long (max 64 characters)"
    
    if len(domain) > 253:
        return False, "Email domain too long (max 253 characters)"
    
    return True, None

def validate_json_structure(data: Any, required_fields: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate JSON data structure contains required fields.
    
    Args:
        data: Data structure to validate
        required_fields: List of required field names
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(data, dict):
        return False, "Data must be a dictionary"
    
    missing_fields = []
    for field in required_fields:
        if field not in data:
            missing_fields.append(field)
    
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    return True, None