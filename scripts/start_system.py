#!/usr/bin/env python3
"""
System startup script for the refactored AgenticRAG system.
"""

import subprocess
import time
import sys
import os
import logging
from pathlib import Path

# Add to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("system_startup")

def start_service(service_name: str, script_path: str, port: int):
    """Start a service and return the process"""
    try:
        logger.info(f"Starting {service_name} on port {port}...")
        
        # Use python -m to run as module
        cmd = [
            sys.executable, script_path,
            "--host", "0.0.0.0",
            "--port", str(port)
        ]
        
        process = subprocess.Popen(
            cmd,
            cwd=os.path.dirname(__file__) + "/.."
        )
        
        # Give process time to start
        time.sleep(2)
        
        if process.poll() is None:
            logger.info(f"✅ {service_name} started successfully (PID: {process.pid})")
            return process
        else:
            logger.error(f"❌ {service_name} failed to start (exit code: {process.returncode})")
            logger.error(f"Check logs/{service_name.lower().replace(' ', '-')}-app.log for details")
            return None
            
    except Exception as e:
        logger.error(f"❌ Failed to start {service_name}: {e}")
        return None

def main():
    """Start all system services"""
    logger.info("🚀 Starting AgenticRAG System (Refactored Architecture)")
    
    # Load configuration
    config = get_config()
    
    processes = []
    
    # Start Model Gateway
    model_gateway_process = start_service(
        "Model Gateway",
        "services/model_gateway_service/app.py",
        config['services']['model_gateway']['port']
    )
    if model_gateway_process:
        processes.append(("Model Gateway", model_gateway_process))
    
    # Wait a bit for model gateway to be ready
    time.sleep(3)
    
    # Start Orchestrator
    orchestrator_process = start_service(
        "Orchestrator", 
        "services/orchestrator_service/app.py",
        config['services']['orchestrator']['port']
    )
    if orchestrator_process:
        processes.append(("Orchestrator", orchestrator_process))
    
    if not processes:
        logger.error("❌ No services started successfully")
        return 1
    
    logger.info(f"✅ Started {len(processes)} services successfully")
    logger.info("System is ready!")
    logger.info("")
    logger.info("Available endpoints:")
    logger.info(f"  - Orchestrator: http://localhost:{config['services']['orchestrator']['port']}")
    logger.info(f"  - Model Gateway: http://localhost:{config['services']['model_gateway']['port']}")
    logger.info("")
    logger.info("Use the CLI: python interfaces/cli/app.py --help")
    logger.info("Press Ctrl+C to stop all services")
    
    try:
        # Wait for keyboard interrupt
        while True:
            time.sleep(1)
            
            # Check if any process has died
            for name, process in processes:
                if process.poll() is not None:
                    logger.error(f"❌ {name} process has died")
    
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down system...")
        
        # Terminate all processes
        for name, process in processes:
            logger.info(f"Stopping {name}...")
            process.terminate()
            
        # Wait for processes to terminate
        for name, process in processes:
            try:
                process.wait(timeout=5)
                logger.info(f"✅ {name} stopped")
            except subprocess.TimeoutExpired:
                logger.warning(f"⚠️ Force killing {name}")
                process.kill()
        
        logger.info("✅ System shutdown complete")
        return 0

if __name__ == "__main__":
    sys.exit(main())