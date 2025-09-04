#!/usr/bin/env python3
"""
Comprehensive test of embedding queue initialization and Ollama availability
"""

import sys
import os
import logging

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging to match the system
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_embedding_queue_initialization():
    """Test embedding queue initialization exactly as in the running system"""
    print("Testing embedding queue initialization...")
    
    try:
        # Import exactly as in embedding_queue.py
        try:
            import ollama
            OLLAMA_AVAILABLE = True
            print(f"Ollama library imported successfully")
            print(f"Ollama version: {ollama.__version__ if hasattr(ollama, '__version__') else 'Unknown'}")
        except ImportError:
            OLLAMA_AVAILABLE = False
            print(f"Ollama import failed")
        
        print(f"OLLAMA_AVAILABLE: {OLLAMA_AVAILABLE}")
        
        # Now test the actual embedding queue singleton
        from services.orchestrator.embedding_queue import embedding_queue, OLLAMA_AVAILABLE as QUEUE_OLLAMA_AVAILABLE
        
        print(f"Embedding queue OLLAMA_AVAILABLE: {QUEUE_OLLAMA_AVAILABLE}")
        print(f"Multi-instance enabled: {embedding_queue.multi_instance_enabled}")
        print(f"Max workers: {embedding_queue.max_workers}")
        
        # Test config utilities
        from services.orchestrator.config_utils import should_use_multi_instance_embedding, get_embedding_instance_count, get_max_concurrent_instances
        
        print(f"Config multi-instance: {should_use_multi_instance_embedding()}")
        print(f"Config instance count: {get_embedding_instance_count()}")
        print(f"Config max concurrent: {get_max_concurrent_instances()}")
        
        # Test a mock job
        from services.orchestrator.embedding_queue import EmbeddingJob
        job = EmbeddingJob(
            id="test-job",
            chunk_id="test-chunk",
            text="This is a test",
            model="dengcao/Qwen3-Embedding-0.6B:Q8_0"
        )
        
        print(f"Created test job: {job}")
        
        # Try to add job to queue
        try:
            embedding_queue.add_job(job)
            print("Successfully added job to queue")
        except Exception as e:
            print(f"Failed to add job to queue: {e}")
            import traceback
            traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"Embedding queue test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_embedding_queue_initialization()