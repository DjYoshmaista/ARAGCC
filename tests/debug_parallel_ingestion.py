#!/usr/bin/env python3
"""
Debug script to test the parallel ingestion controller directly
"""

import sys
import os
import logging
import yaml
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'services', 'orchestrator'))

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load system configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), 'shared', 'configs', 'system.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_parallel_ingestion():
    """Test the parallel ingestion controller directly"""
    try:
        logger.info("Testing parallel ingestion controller...")
        
        # Load system configuration
        system_config = load_config()
        logger.info("System configuration loaded successfully")
        
        # Create ingestion configuration exactly like in app.py
        ingestion_config = {
            "postgresql": system_config["databases"]["postgresql"],
            "qdrant": system_config["databases"]["qdrant"],
            "model_gateway_url": f"http://{system_config['services']['model_gateway']['host']}:{system_config['services']['model_gateway']['port']}",
            "chunk_size": 1000,  # Smaller for testing
            "overlap_size": 200,
            "max_concurrent_files": 2,
            "max_concurrent_chunks": 5,
            "embedding_dimensions": 1024,
            "max_embedding_workers": 2
        }
        
        logger.info(f"Ingestion config created")
        
        # Import the parallel ingestion controller
        from parallel_ingestion_controller import ParallelIngestionController
        from embedding_queue import embedding_queue
        
        logger.info("Modules imported successfully")
        
        # Check initial embedding queue state
        logger.info(f"Initial embedding queue Ollama clients: {len(embedding_queue.ollama_clients)}")
        logger.info(f"Initial embedding queue clients working: {any(embedding_queue.ollama_clients)}")
        
        # Create controller
        controller = ParallelIngestionController(ingestion_config)
        logger.info("Parallel ingestion controller created")
        
        # Check embedding queue state after controller creation
        logger.info(f"After controller creation - Ollama clients: {len(embedding_queue.ollama_clients)}")
        logger.info(f"After controller creation - Clients working: {any(embedding_queue.ollama_clients)}")
        
        # Test with a small test file
        test_file_path = "/home/yosh/repos/AgenticRAGQCode/sample-document.txt"
        
        # Create a small test file if it doesn't exist
        if not os.path.exists(test_file_path):
            with open(test_file_path, 'w') as f:
                f.write("This is a test document for embedding processing.\n" * 10)
            logger.info(f"Created test file: {test_file_path}")
        
        # Process the test file
        logger.info("Starting to process test file...")
        controller.start_workers()
        logger.info("Workers started")
        
        # Check embedding queue state after starting workers
        logger.info(f"After starting workers - Ollama clients: {len(embedding_queue.ollama_clients)}")
        logger.info(f"After starting workers - Clients working: {any(embedding_queue.ollama_clients)}")
        
        # Wait a bit to see if processing starts
        time.sleep(5)
        
        # Get stats
        stats = controller.get_stats()
        logger.info(f"Controller stats: {stats}")
        
        # Get embedding queue stats
        embedding_stats = embedding_queue.get_stats()
        logger.info(f"Embedding queue stats: {embedding_stats}")
        
        # Stop workers
        controller.stop_workers()
        logger.info("Stopped workers")
        
    except Exception as e:
        logger.error(f"Error testing parallel ingestion: {e}", exc_info=True)
        return False
        
    return True

if __name__ == "__main__":
    logger.info("Starting parallel ingestion debug test...")
    success = test_parallel_ingestion()
    if success:
        logger.info("Debug test completed successfully")
    else:
        logger.error("Debug test failed")
        sys.exit(1)