#!/usr/bin/env python3
"""
Force Multi-Instance Test Script

This script forces the multi-instance Ollama system to start more instances
and then tests the load balancing with actual embedding generation.
"""

import asyncio
import sys
import os
import time
import logging

# Add the orchestrator directory to the path
sys.path.append('/home/yosh/repos/AgenticRAGQCode/services/orchestrator')

from ollama_instance_manager import get_instance_manager
from embedding_load_balancer import get_load_balancer, LoadBalancingStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("force_multi_instance_test")

async def main():
    """Force start multiple Ollama instances and test them"""
    
    # Get the instance manager
    instance_manager = get_instance_manager()
    
    # Force start 8 instances
    logger.info("Forcing start of 8 Ollama instances...")
    success = await instance_manager.start_instances(8)
    
    if not success:
        logger.error("Failed to start instances")
        return
    
    # Wait a bit for instances to stabilize
    logger.info("Waiting for instances to stabilize...")
    await asyncio.sleep(10)
    
    # Get stats
    stats = instance_manager.get_stats()
    logger.info(f"Instance Manager Stats:")
    logger.info(f"  Total instances: {stats['total_instances']}")
    logger.info(f"  Running instances: {stats['running_instances']}")
    logger.info(f"  Loaded instances: {stats['loaded_instances']}")
    
    for instance in stats['instances']:
        logger.info(f"  Instance {instance['port']}: {instance['status']} - "
                   f"Model loaded: {instance['model_loaded']}")
    
    # Test load balancer
    logger.info("Testing load balancer...")
    load_balancer = get_load_balancer(LoadBalancingStrategy.LEAST_CONNECTIONS)
    
    # Initialize connections
    await load_balancer.initialize_connections()
    
    # Test embedding generation
    test_texts = [
        "This is a test document for multi-instance Ollama.",
        "Artificial intelligence and machine learning are transforming technology.",
        "Vector databases enable efficient semantic search capabilities.",
        "Load balancing ensures optimal resource utilization across instances.",
        "GPU memory management is crucial for multi-model inference."
    ]
    
    logger.info(f"Generating embeddings for {len(test_texts)} test texts...")
    
    start_time = time.time()
    tasks = []
    for i, text in enumerate(test_texts):
        task = asyncio.create_task(load_balancer.generate_embedding(text, "nomic-embed-text"))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = time.time()
    
    # Analyze results
    successful = 0
    failed = 0
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Text {i} failed: {result}")
            failed += 1
        else:
            embedding, success, error_msg = result
            if success:
                successful += 1
                logger.info(f"Text {i} generated embedding with {len(embedding) if embedding else 0} dimensions")
            else:
                logger.error(f"Text {i} failed: {error_msg}")
                failed += 1
    
    logger.info(f"Results: {successful} successful, {failed} failed")
    logger.info(f"Total time: {end_time - start_time:.2f} seconds")
    
    # Get load balancer stats
    lb_stats = load_balancer.get_stats()
    logger.info(f"Load Balancer Stats:")
    logger.info(f"  Strategy: {lb_stats['strategy']}")
    logger.info(f"  Total requests: {lb_stats['total_requests']}")
    logger.info(f"  Success rate: {lb_stats['success_rate']:.2%}")
    logger.info(f"  Avg response time: {lb_stats['avg_response_time']:.3f}s")
    
    # Instance stats
    for port, inst_stats in lb_stats['instance_stats'].items():
        logger.info(f"  Instance {port}: {inst_stats['active_connections']} connections, "
                   f"state: {inst_stats['circuit_breaker_state']}")
    
    # Cleanup
    await load_balancer.cleanup()

if __name__ == "__main__":
    asyncio.run(main())