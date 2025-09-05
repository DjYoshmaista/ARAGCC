"""
Configuration validator for the AgenticRAG system.
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration and return list of errors.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Validate services section
    if 'services' not in config:
        errors.append("Missing 'services' section in config")
    else:
        services = config['services']
        
        # Validate orchestrator service
        if 'orchestrator' not in services:
            errors.append("Missing 'orchestrator' service config")
        else:
            orch = services['orchestrator']
            if 'host' not in orch or 'port' not in orch:
                errors.append("Orchestrator service missing 'host' or 'port'")
        
        # Validate model gateway service  
        if 'model_gateway' not in services:
            errors.append("Missing 'model_gateway' service config")
        else:
            mg = services['model_gateway']
            if 'host' not in mg or 'port' not in mg:
                errors.append("Model gateway service missing 'host' or 'port'")
    
    # Validate databases section
    if 'databases' not in config:
        errors.append("Missing 'databases' section in config")
    else:
        databases = config['databases']
        
        # Validate PostgreSQL config
        if 'postgresql' not in databases:
            errors.append("Missing 'postgresql' database config")
        else:
            pg = databases['postgresql']
            required_pg_fields = ['host', 'port', 'database', 'username', 'password']
            for field in required_pg_fields:
                if field not in pg:
                    errors.append(f"PostgreSQL config missing required field: {field}")
        
        # Validate Qdrant config
        if 'qdrant' not in databases:
            errors.append("Missing 'qdrant' database config")
        else:
            qdrant = databases['qdrant']
            if 'host' not in qdrant or 'port' not in qdrant:
                errors.append("Qdrant config missing 'host' or 'port'")
    
    # Validate models section
    if 'models' not in config:
        errors.append("Missing 'models' section in config")
    else:
        models = config['models']
        if 'embedding_model' not in models:
            errors.append("Missing 'embedding_model' in models config")
    
    if errors:
        logger.warning(f"Configuration validation found {len(errors)} errors")
        for error in errors:
            logger.warning(f"  - {error}")
    else:
        logger.info("Configuration validation passed")
    
    return errors