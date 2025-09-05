"""
Ollama Instance Manager for Multi-Instance Parallel Embedding Generation

This module manages multiple Ollama instances to maximize GPU utilization and
embedding generation throughput. It handles:
- Spawning multiple Ollama processes on different ports
- GPU memory monitoring and optimal instance scaling
- Model preloading and health monitoring
- Load balancing across instances
"""

import os
import subprocess
import time
import threading
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import httpx
import asyncio
from datetime import datetime, timezone

@dataclass
class OllamaInstance:
    """Represents a single Ollama instance"""
    port: int
    process: Optional[subprocess.Popen] = None
    pid: Optional[int] = None
    status: str = "initializing"  # initializing, running, failed, stopped
    model_loaded: bool = False
    load_balancer_weight: float = 1.0
    requests_handled: int = 0
    last_response_time: float = 0.0
    gpu_memory_used: Optional[int] = None
    base_url: str = None
    
    def __post_init__(self):
        if self.base_url is None:
            self.base_url = f"http://localhost:{self.port}"

class InstanceStatus(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    FAILED = "failed"
    STOPPED = "stopped"

class OllamaInstanceManager:
    """
    Manages multiple Ollama instances for maximum parallel embedding generation
    
    Features:
    - Auto-spawns optimal number of instances based on GPU memory
    - Preloads embedding models in each instance
    - Health monitoring and automatic failover
    - Load balancing with performance-based weights
    """
    
    def __init__(self, base_port: int = 11434, max_instances: int = 12):
        self.logger = logging.getLogger("ollama_instance_manager")
        self.base_port = base_port
        self.max_instances = max_instances
        self.instances: List[OllamaInstance] = []
        self.lock = threading.Lock()
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Configuration
        self.embedding_model = "granite-embedding:latest"
        self.model_size_mb = 62  # Approximate size for planning
        self.startup_timeout = 60  # seconds
        self.health_check_interval = 30  # seconds
        
        # Performance tracking
        self.total_requests = 0
        self.total_response_time = 0.0
        self.failed_requests = 0
        
        # GPU monitoring
        self.gpu_info = []
        
        self.logger.info("Ollama Instance Manager initialized")
    
    def get_gpu_info(self) -> List[Dict[str, Any]]:
        """Get current GPU memory information"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0:
                return []
            
            gpu_info = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 5:
                        gpu_info.append({
                            'index': int(parts[0]),
                            'name': parts[1],
                            'memory_used': int(parts[2]),
                            'memory_total': int(parts[3]),
                            'utilization': int(parts[4]),
                            'memory_free': int(parts[3]) - int(parts[2])
                        })
            
            return gpu_info
            
        except Exception as e:
            self.logger.warning(f"Could not get GPU info: {e}")
            return []
    
    def calculate_optimal_instances(self) -> int:
        """Calculate optimal number of instances based on available GPU memory"""
        self.gpu_info = self.get_gpu_info()
        
        if not self.gpu_info:
            # Fallback if GPU monitoring is unavailable
            self.logger.warning("GPU monitoring unavailable, using conservative instance count")
            return min(4, self.max_instances)
        
        total_free_memory = sum(gpu['memory_free'] for gpu in self.gpu_info)
        
        # Account for base system memory usage and safety margin
        usable_memory = total_free_memory * 0.8  # 20% safety margin
        
        # Calculate how many instances we can fit
        # Each instance needs model memory + working memory
        memory_per_instance = self.model_size_mb + 200  # 200MB working memory
        theoretical_instances = int(usable_memory / memory_per_instance)
        
        # Apply practical limits
        optimal_instances = min(theoretical_instances, self.max_instances)
        optimal_instances = max(optimal_instances, 2)  # At least 2 instances
        
        self.logger.info(f"GPU Memory Analysis:")
        for gpu in self.gpu_info:
            self.logger.info(f"  GPU {gpu['index']}: {gpu['memory_used']}/{gpu['memory_total']}MB "
                           f"({gpu['utilization']}% util)")
        self.logger.info(f"Total free memory: {total_free_memory}MB")
        self.logger.info(f"Optimal instances: {optimal_instances}")
        
        return optimal_instances
    
    async def start_instances(self, num_instances: Optional[int] = None) -> bool:
        """Start the optimal number of Ollama instances"""
        if num_instances is None:
            num_instances = self.calculate_optimal_instances()
        
        self.logger.info(f"Starting {num_instances} Ollama instances...")
        
        success_count = 0
        tasks = []
        
        # Start all instances concurrently
        for i in range(num_instances):
            port = self.base_port + i
            task = asyncio.create_task(self._start_single_instance(port))
            tasks.append(task)
        
        # Wait for all instances to start
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to start instance {i}: {result}")
            elif result:
                success_count += 1
        
        if success_count == 0:
            self.logger.error("Failed to start any Ollama instances")
            return False
        
        self.logger.info(f"Successfully started {success_count}/{num_instances} instances")
        
        # Start monitoring
        self._start_monitoring()
        
        # Preload models in all instances
        await self._preload_models()
        
        return success_count > 0
    
    async def _start_single_instance(self, port: int) -> bool:
        """Start a single Ollama instance on specified port"""
        try:
            # Check if port is available
            if self._is_port_in_use(port):
                # Try to discover if this is an existing Ollama instance
                if await self._discover_existing_instance(port):
                    self.logger.info(f"Port {port} already in use, discovered existing Ollama instance")
                    return True
                else:
                    self.logger.warning(f"Port {port} is already in use by non-Ollama service, skipping")
                    return False
            
            # Set environment for this instance
            env = os.environ.copy()
            env['OLLAMA_HOST'] = f'0.0.0.0:{port}'
            
            # Start Ollama process
            self.logger.info(f"Starting Ollama instance on port {port}")
            process = subprocess.Popen(
                ['ollama', 'serve'],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group for clean shutdown
            )
            
            # Create instance object
            instance = OllamaInstance(
                port=port,
                process=process,
                pid=process.pid,
                status="initializing"
            )
            
            with self.lock:
                self.instances.append(instance)
            
            # Wait for instance to be ready
            if await self._wait_for_instance_ready(instance):
                instance.status = "running"
                self.logger.info(f"Ollama instance started successfully on port {port}")
                return True
            else:
                instance.status = "failed"
                self.logger.error(f"Ollama instance failed to start on port {port}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start Ollama instance on port {port}: {e}")
            return False
    
    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use"""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            return sock.connect_ex(('localhost', port)) == 0
    
    async def _discover_existing_instance(self, port: int) -> bool:
        """Discover and register an existing Ollama instance on the given port"""
        try:
            # Try to connect to the instance to verify it's Ollama
            async with httpx.AsyncClient() as client:
                # Test both IPv4 and IPv6 addresses
                for host in ["localhost", "127.0.0.1", "[::1]"]:
                    test_url = f"http://{host}:{port}"
                    try:
                        response = await client.get(f"{test_url}/api/tags", timeout=5.0)
                        if response.status_code == 200:
                            # Successfully connected - this is an Ollama instance
                            self.logger.info(f"Discovered existing Ollama instance on {test_url}")
                            
                            # Create instance object for existing instance
                            instance = OllamaInstance(port=port, status="running")
                            instance.base_url = test_url
                            
                            with self.lock:
                                # Check if we already have this instance registered
                                existing = next((inst for inst in self.instances if inst.port == port), None)
                                if not existing:
                                    self.instances.append(instance)
                                    self.logger.info(f"Registered existing instance on port {port}")
                            
                            return True
                    except Exception:
                        continue
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error discovering instance on port {port}: {e}")
            return False
    
    async def _wait_for_instance_ready(self, instance: OllamaInstance, timeout: int = 60) -> bool:
        """Wait for an Ollama instance to be ready to accept requests"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{instance.base_url}/api/tags", timeout=5.0)
                    if response.status_code == 200:
                        return True
            except Exception:
                pass
            
            await asyncio.sleep(2)
        
        return False
    
    async def _preload_models(self):
        """Preload the embedding model in all instances"""
        self.logger.info(f"Preloading model '{self.embedding_model}' in all instances...")
        
        tasks = []
        for instance in self.instances:
            if instance.status == "running":
                task = asyncio.create_task(self._preload_model_in_instance(instance))
                tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for r in results if r is True)
            self.logger.info(f"Model preloaded in {success_count}/{len(tasks)} instances")
    
    async def _preload_model_in_instance(self, instance: OllamaInstance) -> bool:
        """Preload model in a specific instance"""
        try:
            async with httpx.AsyncClient() as client:
                # Send empty prompt to trigger model loading
                response = await client.post(
                    f"{instance.base_url}/api/generate",
                    json={
                        "model": self.embedding_model,
                        "prompt": "",
                        "stream": False
                    },
                    timeout=120.0  # Model loading can take time
                )
                
                if response.status_code == 200:
                    instance.model_loaded = True
                    self.logger.info(f"Model preloaded in instance on port {instance.port}")
                    return True
                    
        except Exception as e:
            self.logger.error(f"Failed to preload model in instance {instance.port}: {e}")
        
        return False
    
    def get_next_instance(self) -> Optional[OllamaInstance]:
        """Get the next available instance using load balancing"""
        with self.lock:
            available_instances = [i for i in self.instances 
                                 if i.status == "running" and i.model_loaded]
            
            if not available_instances:
                return None
            
            # Simple round-robin for now, can be enhanced with weighted load balancing
            # Find instance with lowest request count
            best_instance = min(available_instances, key=lambda x: x.requests_handled)
            best_instance.requests_handled += 1
            
            return best_instance
    
    def _start_monitoring(self):
        """Start background monitoring of instances"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_instances, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Instance monitoring started")
    
    def _monitor_instances(self):
        """Background thread to monitor instance health"""
        while self.monitoring_active:
            try:
                asyncio.run(self._check_instance_health())
                time.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Error in instance monitoring: {e}")
    
    async def _check_instance_health(self):
        """Check health of all instances"""
        tasks = []
        for instance in self.instances:
            if instance.status == "running":
                task = asyncio.create_task(self._check_single_instance_health(instance))
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_single_instance_health(self, instance: OllamaInstance):
        """Check health of a single instance"""
        try:
            start_time = time.time()
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{instance.base_url}/api/tags", timeout=10.0)
                
                response_time = time.time() - start_time
                instance.last_response_time = response_time
                
                if response.status_code != 200:
                    self.logger.warning(f"Instance {instance.port} returned status {response.status_code}")
                    instance.status = "failed"
                
        except Exception as e:
            self.logger.error(f"Health check failed for instance {instance.port}: {e}")
            instance.status = "failed"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about all instances"""
        with self.lock:
            running_instances = [i for i in self.instances if i.status == "running"]
            loaded_instances = [i for i in self.instances if i.model_loaded]
            
            stats = {
                'total_instances': len(self.instances),
                'running_instances': len(running_instances),
                'loaded_instances': len(loaded_instances),
                'total_requests': self.total_requests,
                'failed_requests': self.failed_requests,
                'instances': []
            }
            
            for instance in self.instances:
                instance_stats = {
                    'port': instance.port,
                    'status': instance.status,
                    'model_loaded': instance.model_loaded,
                    'requests_handled': instance.requests_handled,
                    'last_response_time': instance.last_response_time,
                    'base_url': instance.base_url
                }
                stats['instances'].append(instance_stats)
            
            # Add GPU info if available
            if self.gpu_info:
                stats['gpu_info'] = self.gpu_info
            
            return stats
    
    def stop_all_instances(self):
        """Stop all Ollama instances"""
        self.logger.info("Stopping all Ollama instances...")
        
        # Stop monitoring
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        # Stop all instances
        with self.lock:
            for instance in self.instances:
                if instance.process and instance.status == "running":
                    try:
                        # Terminate process group to kill all child processes
                        os.killpg(os.getpgid(instance.process.pid), subprocess.signal.SIGTERM)
                        instance.process.wait(timeout=10)
                        instance.status = "stopped"
                        self.logger.info(f"Stopped instance on port {instance.port}")
                    except Exception as e:
                        self.logger.error(f"Failed to stop instance {instance.port}: {e}")
                        try:
                            os.killpg(os.getpgid(instance.process.pid), subprocess.signal.SIGKILL)
                        except:
                            pass
        
        self.logger.info("All instances stopped")
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.stop_all_instances()
        except:
            pass

# Global instance manager
instance_manager = None

def get_instance_manager() -> OllamaInstanceManager:
    """Get the global instance manager (singleton pattern)"""
    global instance_manager
    if instance_manager is None:
        instance_manager = OllamaInstanceManager()
    return instance_manager