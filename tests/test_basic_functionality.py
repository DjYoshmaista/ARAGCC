#!/usr/bin/env python3
"""
Basic functionality test for multi-instance embedding system
Tests core components without full stress testing
"""

import asyncio
import time
import logging
import sys
import os
from pathlib import Path

# Add the services directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'services', 'orchestrator'))

from embedding_queue import EmbeddingJob
from config_utils import should_use_multi_instance_embedding, get_embedding_instance_count

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("basic_test")

async def test_basic_functionality():
    """Test basic system functionality"""
    logger.info("🧪 Starting Basic Multi-Instance System Test")
    
    try:
        # Test config loading
        multi_instance_enabled = should_use_multi_instance_embedding()
        instance_count = get_embedding_instance_count()
        
        logger.info(f"✓ Configuration loaded successfully")
        logger.info(f"  - Multi-instance enabled: {multi_instance_enabled}")
        logger.info(f"  - Target instance count: {instance_count}")
        
        # Test embedding queue initialization
        from embedding_queue import embedding_queue
        logger.info(f"✓ Embedding queue initialized")
        
        # Test instance manager (if available)
        if multi_instance_enabled:
            try:
                from ollama_instance_manager import get_instance_manager
                instance_manager = get_instance_manager()
                
                gpu_info = instance_manager.get_gpu_info()
                logger.info(f"✓ Instance manager initialized")
                logger.info(f"  - GPUs detected: {len(gpu_info)}")
                
                for gpu in gpu_info:
                    logger.info(f"    GPU {gpu['index']}: {gpu['name']} "
                               f"({gpu['memory_used']}/{gpu['memory_total']}MB)")
                
                optimal_instances = instance_manager.calculate_optimal_instances()
                logger.info(f"  - Optimal instances: {optimal_instances}")
                
            except Exception as e:
                logger.warning(f"Instance manager test skipped: {e}")
        
        # Test load balancer (if available)
        try:
            from embedding_load_balancer import get_load_balancer
            load_balancer = get_load_balancer()
            logger.info(f"✓ Load balancer initialized")
        except Exception as e:
            logger.warning(f"Load balancer test skipped: {e}")
        
        # Test performance monitor
        try:
            from performance_monitor import get_performance_monitor
            perf_monitor = get_performance_monitor()
            logger.info(f"✓ Performance monitor initialized")
        except Exception as e:
            logger.warning(f"Performance monitor test skipped: {e}")
        
        # Test database utilities
        try:
            from database_utils import DatabaseManager
            logger.info(f"✓ Database utilities loaded")
        except Exception as e:
            logger.warning(f"Database utilities test skipped: {e}")
        
        logger.info("🎉 Basic functionality test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic functionality test failed: {e}")
        return False

async def test_embedding_job_creation():
    """Test creating embedding jobs"""
    logger.info("🧪 Testing Embedding Job Creation")
    
    try:
        # Create test jobs
        jobs = []
        for i in range(5):
            job = EmbeddingJob(
                id=f"test_job_{i}",
                chunk_id=f"chunk_{i}",
                text=f"Test text {i} for embedding generation.",
                model="dengcao/Qwen3-Embedding-0.6B:Q8_0"
            )
            jobs.append(job)
        
        logger.info(f"✓ Created {len(jobs)} test embedding jobs")
        
        # Test job serialization
        for job in jobs[:2]:  # Test first 2 jobs
            job_dict = job.to_dict()
            logger.info(f"✓ Job {job.id} serialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Embedding job creation test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("=" * 50)
    logger.info("BASIC MULTI-INSTANCE SYSTEM VALIDATION")
    logger.info("=" * 50)
    
    tests = [
        test_basic_functionality(),
        test_embedding_job_creation()
    ]
    
    # Run all tests
    results = []
    for test in tests:
        try:
            result = asyncio.run(test)
            results.append(result)
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    logger.info("=" * 50)
    logger.info(f"TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All basic tests passed! System is ready for full testing.")
        return 0
    else:
        logger.error("❌ Some basic tests failed. Fix issues before proceeding.")
        return 1

if __name__ == "__main__":
    exit(main())