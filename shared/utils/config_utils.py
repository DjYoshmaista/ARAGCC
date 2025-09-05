"""
Configuration utilities for the AgenticRAG system
Handles loading and parsing of system configuration files
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional

def load_system_config() -> Dict[str, Any]:
    """
    Load system configuration from YAML file
    
    Returns:
        Dict with configuration data
    """
    config_path = Path(__file__).parent.parent.parent / "shared" / "configs" / "system.yaml"
    
    if not config_path.exists():
        # Fallback to default configuration
        return get_default_config()
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading system config from {config_path}: {e}")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """
    Get default system configuration
    
    Returns:
        Dict with default configuration values
    """
    return {
        "system": {
            "name": "Multi-Agent Orchestration System",
            "version": "1.0.0",
            "environment": "development"
        },
        "ingestion": {
            "multi_instance_embedding": False,  # Temporarily disabled to fix event loop issues
            "embedding_instances": 1,           # Use single instance for stability
            "max_concurrent_instances": 1,     # Use single instance for stability  
            "max_embedding_workers": 8,        # Keep high worker count for throughput
            "embedding_batch_size": 64
        },
        "models": {
            "embedding_model": "granite-embedding:latest",
            "embedding_dimensions": 384
        }
    }

def get_ingestion_config() -> Dict[str, Any]:
    """
    Get ingestion-specific configuration
    
    Returns:
        Dict with ingestion configuration values
    """
    config = load_system_config()
    return config.get("ingestion", {})

def get_model_config() -> Dict[str, Any]:
    """
    Get model-specific configuration
    
    Returns:
        Dict with model configuration values
    """
    config = load_system_config()
    return config.get("models", {})

def should_use_multi_instance_embedding() -> bool:
    """
    Check if multi-instance embedding processing should be used
    
    Returns:
        bool: True if multi-instance processing should be enabled
    """
    ingestion_config = get_ingestion_config()
    return ingestion_config.get("multi_instance_embedding", True)

def get_embedding_instance_count() -> int:
    """
    Get the number of embedding model instances to create
    
    Returns:
        int: Number of instances to create
    """
    ingestion_config = get_ingestion_config()
    return ingestion_config.get("embedding_instances", 8)

def get_max_concurrent_instances() -> int:
    """
    Get the maximum number of concurrent instances allowed
    
    Returns:
        int: Maximum concurrent instances
    """
    ingestion_config = get_ingestion_config()
    return ingestion_config.get("max_concurrent_instances", 8)