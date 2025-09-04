#!/usr/bin/env python3
"""
Start multiple Ollama instances and preload models in each
"""

import asyncio
import subprocess
import time
import logging
import httpx
import os
import signal
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multi_instance_starter")

MODEL = "dengcao/Qwen3-Embedding-0.6B:Q8_0"
PORTS = [11434, 11435, 11436, 11437, 11438, 11439, 11440, 11441, 11442]
processes = []

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal, stopping all instances...")
    for port, process in processes:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            logger.info(f"Stopped instance on port {port}")
        except:
            pass
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def start_ollama_instance(port: int):
    """Start a single Ollama instance"""
    if port == 11434:
        # Skip main instance if already running
        try:
            result = subprocess.run(['pgrep', '-f', 'ollama serve'], capture_output=True)
            if result.returncode == 0:
                logger.info(f"Port {port}: Main instance already running, skipping")
                return None
        except:
            pass
    
    env = os.environ.copy()
    env['OLLAMA_HOST'] = f'0.0.0.0:{port}'
    
    logger.info(f"Starting Ollama on port {port}")
    process = subprocess.Popen(
        ['ollama', 'serve'],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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

async def preload_model(port: int):
    """Preload model in an instance"""
    try:
        async with httpx.AsyncClient() as client:
            logger.info(f"Loading model in instance {port}...")
            # Try /api/embed first, fallback to /api/embeddings
            response = None
            try:
                response = await client.post(
                    f"http://localhost:{port}/api/embed",
                    json={
                        "model": MODEL,
                        "prompt": "preload"
                    },
                    timeout=120.0
                )
            except Exception:
                response = await client.post(
                    f"http://localhost:{port}/api/embeddings",
                    json={
                        "model": MODEL,
                        "prompt": "preload"
                    },
                    timeout=120.0
                )
            
            if response.status_code == 200:
                logger.info(f"✅ Port {port}: Model loaded successfully")
                return True
            else:
                logger.error(f"❌ Port {port}: Model load failed with HTTP {response.status_code}")
                return False
                
    except Exception as e:
        logger.error(f"❌ Port {port}: Model load failed: {e}")
        return False

async def main():
    """Start multiple instances and preload models"""
    global processes
    
    logger.info(f"🚀 Starting {len(PORTS)} Ollama instances...")
    
    # Start all instances
    for port in PORTS:
        process = start_ollama_instance(port)
        if process:
            processes.append((port, process))
        else:
            processes.append((port, None))  # Already running
    
    # Wait for instances to be ready
    logger.info("⏳ Waiting for instances to be ready...")
    await asyncio.sleep(10)
    
    ready_ports = []
    for port in PORTS:
        if await wait_for_instance_ready(port, 30):
            ready_ports.append(port)
            logger.info(f"✅ Instance {port} is ready")
        else:
            logger.error(f"❌ Instance {port} failed to start")
    
    logger.info(f"📊 {len(ready_ports)} instances ready: {ready_ports}")
    
    if not ready_ports:
        logger.error("No instances available")
        return
    
    # Check GPU memory before loading
    logger.info("\n📊 GPU Memory Before Model Loading:")
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(result.stdout)
    
    # Preload models in all instances
    logger.info(f"\n🔄 Preloading models in {len(ready_ports)} instances...")
    
    # Load models concurrently but with slight delays to avoid overwhelming system
    tasks = []
    for i, port in enumerate(ready_ports):
        # Add small delay between starts
        task = asyncio.create_task(preload_after_delay(port, i * 2))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    successful_loads = sum(results)
    logger.info(f"\n✅ Successfully loaded models in {successful_loads}/{len(ready_ports)} instances")
    
    # Check GPU memory after loading
    logger.info("\n📊 GPU Memory After Model Loading:")
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(result.stdout)
    
    # Keep instances running
    logger.info(f"\n🔄 Keeping {successful_loads} instances running with models loaded...")
    logger.info("Press Ctrl+C to stop all instances")
    
    try:
        # Monitor instances periodically
        while True:
            await asyncio.sleep(30)
            
            # Check if processes are still running
            running_count = 0
            for port, process in processes:
                if process and process.poll() is None:
                    running_count += 1
                elif not process:  # Main instance
                    running_count += 1
            
            logger.info(f"Status: {running_count} instances still running")
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested...")

async def preload_after_delay(port: int, delay: float):
    """Preload model after a delay"""
    await asyncio.sleep(delay)
    return await preload_model(port)

if __name__ == "__main__":
    asyncio.run(main())