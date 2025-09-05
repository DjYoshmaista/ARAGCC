"""
Configuration Manager for the AgenticRAG CLI
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, asdict

@dataclass
class CLIConfig:
    """CLI configuration structure"""
    # Service endpoints
    orchestrator_url: str = "http://localhost:8001"
    model_gateway_url: str = "http://localhost:8070"
    vector_engine_url: str = "http://localhost:8080"
    
    # Database settings
    postgresql_host: str = "localhost"
    postgresql_port: int = 5432
    postgresql_db: str = "agentic_system"
    postgresql_user: str = "postgres"
    postgresql_password: str = "password"
    
    # Interface settings
    verbose: bool = False
    color_output: bool = True

class ConfigManager:
    """Manages CLI configuration"""
    
    def __init__(self):
        self.config_dir = Path.home() / ".config" / "agentic-rag"
        self.config_file = self.config_dir / "config.yaml"
        self.config = CLIConfig()
        self._ensure_directories()
        self._load_config()
    
    def _ensure_directories(self):
        """Ensure necessary directories exist"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self):
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Update config with loaded data
                for key, value in config_data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
    
    def _save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(asdict(self.config), f, default_flow_style=False)
        except Exception as e:
            print(f"Error: Could not save config file: {e}")
    
    def get_config(self) -> CLIConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self._save_config()