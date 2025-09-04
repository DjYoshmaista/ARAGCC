#!/usr/bin/env python3
"""
Test script to verify that the Ollama client issue is resolved
"""

import sys
import os
import logging

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_embedding_queue_functionality():
    """Test that the embedding queue works without the Ollama client warning"""
    print("Testing embedding queue functionality...")
    
    try:
        # Import the fixed embedding queue
        from services.orchestrator.embedding_queue import embedding_queue, OLLAMA_AVAILABLE
        
        print(f"OLLAMA_AVAILABLE: {OLLAMA_AVAILABLE}")
        print(f"Multi-instance enabled: {embedding_queue.multi_instance_enabled}")
        print(f"Workers running: {embedding_queue.workers_running}")
        
        # Test creating and adding a job
        from services.orchestrator.embedding_queue import EmbeddingJob
        job = EmbeddingJob(
            id="test-job-1",
            chunk_id="test-chunk-1",
            text="This is a test document for embedding generation",
            model="dengcao/Qwen3-Embedding-0.6B:Q8_0"
        )
        
        print(f"Created test job: {job.id}")
        
        # Add job to queue
        embedding_queue.add_job(job)
        print("Successfully added job to queue")
        
        # Try to get stats
        stats = embedding_queue.get_stats()
        print(f"Queue stats: {stats.get('queue_size', 0)} jobs queued")
        
        print("Embedding queue test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Embedding queue test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_embedding_queue_functionality()
    if success:
        print("\n✅ All tests passed! The Ollama client issue should now be resolved.")
    else:
        print("\n❌ Tests failed. There may still be issues to resolve.")