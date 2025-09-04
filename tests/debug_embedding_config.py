#!/usr/bin/env python3
"""
Debug script to test the embedding queue exactly as used by the parallel ingestion controller
"""

import sys
import os
import logging
import yaml
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

def test_embedding_queue_with_config():
    """Test the embedding queue with the same configuration used by the ingestion controller"""
    try:
        logger.info("Testing embedding queue with configuration...")
        
        # Load system configuration
        system_config = load_config()
        logger.info("System configuration loaded successfully")
        
        # Create ingestion configuration exactly like in app.py
        ingestion_config = {
            "postgresql": system_config["databases"]["postgresql"],
            "qdrant": system_config["databases"]["qdrant"],
            "model_gateway_url": f"http://{system_config['services']['model_gateway']['host']}:{system_config['services']['model_gateway']['port']}",
            "chunk_size": 5000,
            "overlap_size": 32,
            "max_concurrent_files": 5,
            "max_concurrent_chunks": 10,
            "embedding_dimensions": 1024  # For dengcao/Qwen3-Embedding-0.6B:Q8_0
        }
        
        logger.info(f"Ingestion config created")
        
        # Import the embedding queue
        from embedding_queue import embedding_queue, EmbeddingJob
        
        logger.info("Embedding queue imported successfully")
        
        # Check embedding queue state
        logger.info(f"Ollama clients available: {len(embedding_queue.ollama_clients)}")
        logger.info(f"Ollama clients working: {any(embedding_queue.ollama_clients)}")
        
        # Check if any clients are None
        for i, client in enumerate(embedding_queue.ollama_clients):
            logger.info(f"Ollama client {i}: {client is not None}")
        
        # Test creating a embedding job exactly like the controller would
        model = system_config.get('models', {}).get('embedding_model', 'dengcao/Qwen3-Embedding-0.6B:Q8_0')
        logger.info(f"Using embedding model: {model}")
        
        job = EmbeddingJob(
            id="test-job-1",
            chunk_id="test-chunk-1",
            text="This is a test sentence for embedding.",
            model=model
        )
        
        logger.info("Created test embedding job with configured model")
        
        # Add job to queue
        embedding_queue.add_job(job)
        logger.info("Added job to queue")
        
        # Start workers like the controller would
        embedding_queue.start_workers(num_workers=2)
        logger.info("Started workers")
        
        # Try to get result
        import time
        time.sleep(2)  # Give worker time to process
        
        result = embedding_queue.get_result(timeout=5.0)
        if result:
            logger.info(f"Got result: success={result.success}, embedding length={len(result.embedding) if result.embedding else 0}")
            if result.success:
                logger.info("Embedding generation successful!")
            else:
                logger.error(f"Embedding generation failed: {result.error}")
        else:
            logger.warning("No result received within timeout")
            
        # Get stats
        stats = embedding_queue.get_stats()
        logger.info(f"Queue stats: {stats}")
        
        # Stop workers
        embedding_queue.stop_workers()
        logger.info("Stopped workers")
        
    except Exception as e:
        logger.error(f"Error testing embedding queue with config: {e}", exc_info=True)
        return False
        
    return True

if __name__ == "__main__":
    logger.info("Starting embedding queue debug test with configuration...")
    success = test_embedding_queue_with_config()
    if success:
        logger.info("Debug test with configuration completed successfully")
    else:
        logger.error("Debug test with configuration failed")
        sys.exit(1)