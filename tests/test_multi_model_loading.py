#!/usr/bin/env python3
"""
Test script to manually start Ollama instances and load models in each
"""

import asyncio
import subprocess
import time
import logging
import httpx
from pathlib import Path
import sys
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_loading_test")

MODEL = "dengcao/Qwen3-Embedding-0.6B:Q8_0"

async def test_instance(port: int):
    """Test if an instance has the model loaded"""
    try:
        async with httpx.AsyncClient() as client:
            # Test model loading with a small embedding request
            # Try /api/embed first, fallback to /api/embeddings
            response = None
            try:
                response = await client.post(
                    f"http://localhost:{port}/api/embed",
                    json={
                        "model": MODEL,
                        "prompt": "test"
                    },
                    timeout=30.0
                )
            except Exception:
                response = await client.post(
                    f"http://localhost:{port}/api/embeddings",
                    json={
                        "model": MODEL,
                        "prompt": "test"
                    },
                    timeout=30.0
                )
            if response.status_code == 200:
                logger.info(f"✅ Port {port}: Model loaded and working")
                return True
            else:
                logger.error(f"❌ Port {port}: HTTP {response.status_code}")
                return False
    except Exception as e:
        logger.error(f"❌ Port {port}: {e}")
        return False

async def preload_model(port: int):
    """Preload model in an instance"""
    try:
        async with httpx.AsyncClient() as client:
            # Send embedding request to trigger model loading
            logger.info(f"Loading model in instance {port}...")
            # Try /api/embed first, fallback to /api/embeddings
            response = None
            try:
                response = await client.post(
                    f"http://localhost:{port}/api/embed",
                    json={
                        "model": MODEL,
                        "prompt": "preload test"
                    },
                    timeout=120.0  # Model loading can take time
                )
            except Exception:
                response = await client.post(
                    f"http://localhost:{port}/api/embeddings",
                    json={
                        "model": MODEL,
                        "prompt": "preload test"
                    },
                    timeout=120.0  # Model loading can take time
                )
            
            if response.status_code == 200:
                logger.info(f"✅ Port {port}: Model preloaded successfully")
                return True
            else:
                logger.error(f"❌ Port {port}: Preload failed with HTTP {response.status_code}")
                return False
                
    except Exception as e:
        logger.error(f"❌ Port {port}: Preload failed: {e}")
        return False

def start_ollama_instance(port: int):
    """Start a single Ollama instance"""
    env = os.environ.copy()
    env['OLLAMA_HOST'] = f'0.0.0.0:{port}'
    
    logger.info(f"Starting Ollama on port {port}")
    process = subprocess.Popen(
        ['ollama', 'serve'],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )
    
    return process

async def wait_for_instance_ready(port: int, timeout: int = 60):
    """Wait for instance to be ready"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://localhost:{port}/api/tags", timeout=5.0)
                if response.status_code == 200:
                    return True
        except Exception:
            pass
        await asyncio.sleep(2)
    
    return False

async def main():
    """Main test function"""
    logger.info("🧪 Testing Multi-Instance Model Loading")
    
    # Test current instances first
    logger.info("\n📊 Testing currently running instances...")
    running_instances = []
    for port in range(11434, 11443):  # Check ports 11434-11442
        if await test_instance(port):
            running_instances.append(port)
    
    logger.info(f"Found {len(running_instances)} working instances: {running_instances}")
    
    if not running_instances:
        logger.info("\n🚀 Starting new instances...")
        # Start multiple instances
        processes = []
        ports = [11434, 11435, 11436, 11437]  # Start 4 instances for testing
        
        for port in ports:
            if port != 11434:  # Don't start main instance if it's already running
                process = start_ollama_instance(port)
                processes.append((port, process))
        
        # Wait for instances to be ready
        logger.info("⏳ Waiting for instances to be ready...")
        await asyncio.sleep(10)  # Give instances time to start
        
        ready_instances = []
        for port in ports:
            if await wait_for_instance_ready(port):
                ready_instances.append(port)
                logger.info(f"✅ Instance {port} is ready")
            else:
                logger.error(f"❌ Instance {port} failed to start")
        
        running_instances = ready_instances
    
    if not running_instances:
        logger.error("❌ No instances available for testing")
        return
    
    # Preload models in all instances
    logger.info(f"\n🔄 Preloading models in {len(running_instances)} instances...")
    
    # Check GPU memory before
    logger.info("\n📊 GPU Memory Before Model Loading:")
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(result.stdout)
    
    # Preload models concurrently
    tasks = [preload_model(port) for port in running_instances]
    results = await asyncio.gather(*tasks)
    
    successful_loads = sum(results)
    logger.info(f"\n✅ Successfully preloaded models in {successful_loads}/{len(running_instances)} instances")
    
    # Check GPU memory after
    logger.info("\n📊 GPU Memory After Model Loading:")
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(result.stdout)
    
    # Test concurrent embedding generation
    logger.info(f"\n⚡ Testing concurrent embedding generation...")
    embed_tasks = []
    for i, port in enumerate(running_instances):
        embed_tasks.append(test_instance(port))
    
    embed_results = await asyncio.gather(*embed_tasks)
    working_instances = sum(embed_results)
    logger.info(f"✅ {working_instances}/{len(running_instances)} instances successfully generated embeddings")

if __name__ == "__main__":
    asyncio.run(main())