#!/usr/bin/env python3
"""Restart ingestion with single-instance configuration"""

import asyncio
import logging
import sys
import os
import requests
import time

# Add the services directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services', 'orchestrator'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_ingestion_status():
    """Check current ingestion status"""
    try:
        # Find the currently running task (assuming it's the last one)
        response = requests.get("http://localhost:8001/tasks", timeout=10)
        if response.status_code == 200:
            tasks = response.json()
            if tasks:
                # Get the most recent task
                latest_task = tasks[-1]
                task_id = latest_task.get('id')
                
                logger.info(f"Found active task: {task_id}")
                
                # Get task status
                status_response = requests.get(f"http://localhost:8001/tasks/{task_id}", timeout=10)
                if status_response.status_code == 200:
                    task_info = status_response.json()
                    logger.info(f"Task status: {task_info.get('status', 'unknown')}")
                    logger.info(f"Task description: {task_info.get('description', 'N/A')}")
                    return task_id
                
        logger.warning("No active tasks found")
        return None
        
    except Exception as e:
        logger.error(f"Failed to check ingestion status: {e}")
        return None

def get_task_progress(task_id):
    """Get progress for a specific task"""
    try:
        response = requests.get(f"http://localhost:8001/tasks/{task_id}/progress", timeout=10)
        if response.status_code == 200:
            progress = response.json()
            logger.info(f"Progress: {progress}")
            return progress
        else:
            logger.warning(f"Failed to get progress: HTTP {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Failed to get task progress: {e}")
        return None

async def test_single_instance_embedding():
    """Test embedding generation with single instance configuration"""
    try:
        from config_utils import should_use_multi_instance_embedding, get_embedding_instance_count
        from embedding_load_balancer import get_load_balancer
        
        logger.info(f"Multi-instance embedding: {should_use_multi_instance_embedding()}")
        logger.info(f"Embedding instance count: {get_embedding_instance_count()}")
        
        # Reset and test load balancer
        load_balancer = get_load_balancer()
        await load_balancer.initialize_connections()
        
        # Test embedding generation
        test_text = "Testing single instance embedding configuration"
        test_model = "granite-embedding"
        
        embedding, success, error_msg = await load_balancer.generate_embedding(test_text, test_model)
        
        if success and embedding:
            logger.info(f"✅ Single-instance embedding test successful! ({len(embedding)} dims)")
            return True
        else:
            logger.error(f"❌ Single-instance embedding test failed: {error_msg}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to test single-instance embedding: {e}")
        return False

def main():
    logger.info("🚀 Checking ingestion status and testing single-instance configuration...")
    
    # Check current ingestion status
    task_id = check_ingestion_status()
    
    if task_id:
        # Get current progress
        progress = get_task_progress(task_id)
        if progress:
            files_processed = progress.get('files_processed', 0)
            total_files = progress.get('total_files', 0)
            embeddings_completed = progress.get('embeddings_completed', 0)
            
            logger.info(f"📊 Current Progress:")
            logger.info(f"  - Files processed: {files_processed}/{total_files}")
            logger.info(f"  - Embeddings completed: {embeddings_completed}")
            
            if files_processed > 0:
                percentage = (files_processed / total_files) * 100 if total_files > 0 else 0
                logger.info(f"  - Progress: {percentage:.2f}%")
    
    # Test single-instance configuration
    logger.info("\n🧪 Testing single-instance embedding configuration...")
    success = asyncio.run(test_single_instance_embedding())
    
    if success:
        logger.info("\n✅ Single-instance configuration is working correctly!")
        logger.info("💡 The ingestion should continue with better stability.")
        logger.info("🔄 Monitor the logs for 'Event loop is closed' errors - they should be gone.")
    else:
        logger.error("\n❌ Single-instance configuration test failed.")
    
    logger.info(f"\n📝 Configuration changes made:")
    logger.info(f"  - multi_instance_embedding: False")
    logger.info(f"  - embedding_instances: 1") 
    logger.info(f"  - max_concurrent_instances: 1")
    logger.info(f"  - max_embedding_workers: 8 (for parallel processing)")

if __name__ == "__main__":
    main()