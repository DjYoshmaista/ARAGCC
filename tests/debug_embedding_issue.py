#!/usr/bin/env python3
"""
Debug script to test the embedding queue and identify the issue
"""

import sys
import os
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_embedding_queue():
    """Test the embedding queue directly"""
    try:
        logger.info("Testing embedding queue initialization...")
        
        # Import the embedding queue
        from services.orchestrator.embedding_queue import embedding_queue, EmbeddingJob
        
        logger.info("Embedding queue imported successfully")
        logger.info(f"Ollama clients available: {len(embedding_queue.ollama_clients)}")
        logger.info(f"Ollama clients working: {any(embedding_queue.ollama_clients)}")
        
        # Check if any clients are None
        for i, client in enumerate(embedding_queue.ollama_clients):
            logger.info(f"Ollama client {i}: {client is not None}")
        
        # Test creating a simple embedding job
        job = EmbeddingJob(
            id="test-job-1",
            chunk_id="test-chunk-1",
            text="This is a test sentence for embedding.",
            model="dengcao/Qwen3-Embedding-0.6B:Q8_0"
        )
        
        logger.info("Created test embedding job")
        
        # Add job to queue
        embedding_queue.add_job(job)
        logger.info("Added job to queue")
        
        # Start workers
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
        logger.error(f"Error testing embedding queue: {e}", exc_info=True)
        return False
        
    return True

if __name__ == "__main__":
    logger.info("Starting embedding queue debug test...")
    success = test_embedding_queue()
    if success:
        logger.info("Debug test completed successfully")
    else:
        logger.error("Debug test failed")
        sys.exit(1)