"""
Configuration manager for the AgenticRAG system.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'settings', 'system.yaml')
        
        self.config_path = Path(config_path)
        self._config: Optional[Dict[str, Any]] = None
        self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return self._config
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            self._config = self._get_default_config()
            return self._config
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        if self._config is None:
            self.load_config()
        return self._config or {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)"""
        config = self.get_config()
        keys = key.split('.')
        
        current = config
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'services': {
                'orchestrator': {'host': 'localhost', 'port': 8001},
                'model_gateway': {'host': 'localhost', 'port': 8070, 'ollama_url': 'http://localhost:11434'},
                'vector_engine': {'host': 'localhost', 'port': 8080}
            },
            'databases': {
                'postgresql': {
                    'host': 'localhost', 'port': 5432, 'database': 'agentic_system',
                    'username': 'postgres', 'password': 'password'
                },
                'qdrant': {'host': 'localhost', 'port': 6333, 'collection_name': 'embeddings'}
            },
            'models': {
                'embedding_model': 'granite-embedding:latest',
                'embedding_dimensions': 384
            },
            'logging': {'level': 'INFO'}
        }

# Global singleton
_config_manager = None

def get_config() -> Dict[str, Any]:
    """Get global configuration"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.get_config()