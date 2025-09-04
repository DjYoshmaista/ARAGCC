#!/usr/bin/env python3
"""
Comprehensive debug script to simulate the actual workflow and identify issues
"""

import sys
import os
import time
import threading
from pathlib import Path

# Add the orchestrator directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'services', 'orchestrator'))

# Import the components
from embedding_queue import embedding_queue, EmbeddingJob, EmbeddingResult
from parallel_ingestion_controller import ParallelIngestionController

def test_workflow():
    print("=== Testing Full Workflow ===")
    
    # Load system configuration
    config_path = os.path.join(os.path.dirname(__file__), 'shared', 'configs', 'system.yaml')
    import yaml
    with open(config_path, 'r') as f:
        system_config = yaml.safe_load(f)
    
    # Create ingestion configuration
    ingestion_config = {
        "postgresql": system_config["databases"]["postgresql"],
        "qdrant": system_config["databases"]["qdrant"],
        "embedding_model": system_config["models"].get("embedding_model", "dengcao/Qwen3-Embedding-0.6B:Q8_0"),
        "embedding_dimensions": system_config["models"].get("embedding_dimensions", 1024),
        "max_cpu_workers": 2,
        "max_db_workers": 2,
        "max_concurrent_files": 10,
        "chunk_batch_size": 50,
        "checkpoint_interval": 10,
        "max_embedding_workers": 2,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "qdrant_collection": "document_embeddings"
    }
    
    print("Creating ParallelIngestionController...")
    controller = ParallelIngestionController(ingestion_config)
    
    print("Initializing database connections...")
    try:
        controller.initialize_database(
            system_config["databases"]["postgresql"],
            system_config["databases"]["qdrant"]
        )
        print("✅ Database connections initialized")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return
    
    print("Starting workers...")
    try:
        controller.start_workers()
        print("✅ Workers started")
    except Exception as e:
        print(f"❌ Worker start failed: {e}")
        return
    
    # Create a test embedding job
    print("Creating test embedding job...")
    test_job = EmbeddingJob(
        id="workflow_test_1",
        chunk_id="workflow_chunk_1",
        text="This is a test sentence for workflow testing.",
        model=system_config["models"].get("embedding_model", "dengcao/Qwen3-Embedding-0.6B:Q8_0")
    )
    
    # Add job to embedding queue
    print("Adding job to embedding queue...")
    embedding_queue.add_job(test_job)
    
    # Wait for result
    print("Waiting for result...")
    start_time = time.time()
    result = None
    while time.time() - start_time < 10:  # 10 second timeout
        result = embedding_queue.get_result(timeout=1.0)
        if result:
            break
    
    if result:
        print(f"✅ Result received!")
        print(f"  - Success: {result.success}")
        print(f"  - Processing time: {result.processing_time:.2f}s")
        print(f"  - Embedding dimensions: {len(result.embedding)}")
        if result.success:
            print(f"  - First 5 values: {result.embedding[:5]}")
        else:
            print(f"  - Error: {result.error}")
    else:
        print("❌ No result received within timeout")
        # Check embedding queue stats
        stats = embedding_queue.get_stats()
        print(f"  - Queue size: {stats.get('queue_size', 0)}")
        print(f"  - Result queue size: {stats.get('result_queue_size', 0)}")
        print(f"  - Total queued: {stats.get('total_queued', 0)}")
        print(f"  - Total processed: {stats.get('total_processed', 0)}")
        print(f"  - Total failed: {stats.get('total_failed', 0)}")
    
    # Stop workers
    print("Stopping workers...")
    controller.stop_workers()
    print("✅ Workers stopped")
    
    return result is not None and result.success

if __name__ == "__main__":
    success = test_workflow()
    if success:
        print("\n🎉 Workflow test completed successfully!")
    else:
        print("\n💥 Workflow test failed!")