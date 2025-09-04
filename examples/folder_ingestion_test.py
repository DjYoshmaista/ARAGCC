#!/usr/bin/env python3
"""
Test script for folder ingestion functionality
"""

import asyncio
import sys
import os

# Add the orchestrator directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services', 'orchestrator'))

from ingestion_controller import IngestionController
import yaml

async def test_folder_ingestion():
    """Test the folder ingestion functionality"""
    
    # Load system configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'shared', 'configs', 'system.yaml')
    with open(config_path, 'r') as f:
        system_config = yaml.safe_load(f)
    
    # Create ingestion configuration
    ingestion_config = {
        "postgresql": system_config["databases"]["postgresql"],
        "qdrant": system_config["databases"]["qdrant"],
        "model_gateway_url": f"http://{system_config['services']['model_gateway']['host']}:{system_config['services']['model_gateway']['port']}",
        "chunk_size": 1000,
        "overlap_size": 50,
        "max_concurrent_files": 2,
        "max_concurrent_chunks": 5,
        "embedding_dimensions": 640
    }
    
    # Create and initialize ingestion controller
    controller = IngestionController(ingestion_config)
    
    try:
        await controller.initialize()
        print("Ingestion controller initialized successfully")
        
        # Test with a sample directory
        test_paths = ["txt"]  # Adjust this path as needed
        stats = await controller.ingest_paths(test_paths, recursive=True)
        
        print("\nIngestion completed with stats:")
        print(f"Files ingested: {stats.get('files_ingested', 0)}")
        print(f"Chunks created: {stats.get('chunks_created', 0)}")
        print(f"Errors: {stats.get('errors', 0)}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
    finally:
        controller.db_manager.close_connections()

if __name__ == "__main__":
    asyncio.run(test_folder_ingestion())