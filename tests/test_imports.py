#!/usr/bin/env python3
"""
Test script to verify modular structure
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test that all modules can be imported"""
    try:
        # Test CLI imports
        from cli import command_parser, config_manager, api_client
        print("✅ CLI modules imported successfully")
        
        # Test core imports
        from core.orchestrator import app
        print("✅ Core orchestrator module imported successfully")
        
        from core.ingestion import controller, text_processor, progress_tracker
        print("✅ Core ingestion modules imported successfully")
        
        from core.embedding import queue, load_balancer, instance_manager
        print("✅ Core embedding modules imported successfully")
        
        from core.database import manager
        print("✅ Core database module imported successfully")
        
        from core.models import gateway
        print("✅ Core models module imported successfully")
        
        # Test shared imports
        from shared.schemas import task_schema, document_schema, embedding_schema
        print("✅ Shared schemas imported successfully")
        
        from shared.utils import logger, rate_limiter, config_utils, validators
        print("✅ Shared utils imported successfully")
        
        print("🎉 All modules imported successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)