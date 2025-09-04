"""
Load Balancer for Multi-Instance Ollama Embedding Requests

This module provides intelligent load balancing across multiple Ollama instances
to maximize embedding generation throughput and GPU utilization.
"""

import asyncio
import time
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import httpx
from threading import Lock
import random

from ollama_instance_manager import OllamaInstanceManager, OllamaInstance, get_instance_manager

class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RESPONSE_TIME = "weighted_response_time"
    RANDOM = "random"

@dataclass
class LoadBalancerStats:
    """Statistics for load balancer performance"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    requests_per_second: float = 0.0
    active_connections: int = 0
    
    def update_response_time(self, response_time: float):
        """Update average response time"""
        if self.successful_requests == 0:
            self.avg_response_time = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.avg_response_time = (alpha * response_time + 
                                    (1 - alpha) * self.avg_response_time)

class EmbeddingLoadBalancer:
    """
    Intelligent load balancer for embedding requests across multiple Ollama instances
    
    Features:
    - Multiple load balancing strategies
    - Health monitoring and failover
    - Performance-based routing
    - Connection pooling and reuse
    - Circuit breaker pattern for failed instances
    """
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS):
        self.logger = logging.getLogger("embedding_load_balancer")
        self.strategy = strategy
        self.instance_manager: OllamaInstanceManager = get_instance_manager()
        
        # Load balancing state
        self.current_instance_index = 0
        self.instance_weights: Dict[int, float] = {}  # port -> weight
        self.instance_connections: Dict[int, int] = {}  # port -> active connections
        self.instance_circuit_breakers: Dict[int, Dict] = {}  # port -> circuit breaker state
        
        # Connection pool
        self.client_pool: Dict[int, httpx.AsyncClient] = {}
        self.pool_lock = Lock()
        
        # Statistics
        self.stats = LoadBalancerStats()
        self.stats_lock = Lock()
        
        # Circuit breaker configuration
        self.circuit_breaker_threshold = 5  # failures before opening circuit
        self.circuit_breaker_timeout = 30  # seconds before retrying
        self.circuit_breaker_recovery_requests = 3  # successful requests to close circuit
        
        self.logger.info(f"Embedding Load Balancer initialized with strategy: {strategy}")
    
    async def initialize_connections(self):
        """Initialize connection pool for all instances"""
        # First, try to discover additional running Ollama instances
        await self._discover_additional_instances()
        
        instances = self.instance_manager.instances
        
        with self.pool_lock:
            for instance in instances:
                if instance.status == "running" and instance.port not in self.client_pool:
                    # Create persistent HTTP client for each instance
                    client = httpx.AsyncClient(
                        base_url=instance.base_url,
                        timeout=httpx.Timeout(60.0),
                        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
                    )
                    self.client_pool[instance.port] = client
                    self.instance_connections[instance.port] = 0
                    self.instance_weights[instance.port] = 1.0
                    self._initialize_circuit_breaker(instance.port)
                    
                    self.logger.info(f"Initialized connection pool for instance {instance.port}")
    
    async def _discover_additional_instances(self):
        """Discover additional Ollama instances that may be running"""
        # Try common Ollama ports
        potential_ports = [11434, 11435, 11436, 11437, 11438, 11439, 11440, 11441, 11442]
        
        discovered_count = 0
        for port in potential_ports:
            # Skip if we already know about this instance
            if any(instance.port == port for instance in self.instance_manager.instances):
                continue
            
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"http://localhost:{port}/api/tags", timeout=2.0)
                    if response.status_code == 200:
                        # Found a running Ollama instance
                        from ollama_instance_manager import OllamaInstance
                        instance = OllamaInstance(
                            port=port,
                            status="running",
                            base_url=f"http://localhost:{port}"
                        )
                        
                        # Add to instance manager
                        with self.instance_manager.lock:
                            self.instance_manager.instances.append(instance)
                        
                        self.logger.info(f"Discovered additional Ollama instance on port {port}")
                        discovered_count += 1
            except Exception:
                # Port not available or not Ollama, skip silently
                pass
        
        if discovered_count > 0:
            self.logger.info(f"Discovered {discovered_count} additional Ollama instances")
    
    def _initialize_circuit_breaker(self, port: int):
        """Initialize circuit breaker state for an instance"""
        self.instance_circuit_breakers[port] = {
            'state': 'closed',  # closed, open, half_open
            'failures': 0,
            'last_failure_time': 0,
            'success_count': 0
        }
    
    def _update_circuit_breaker(self, port: int, success: bool):
        """Update circuit breaker state based on request result"""
        breaker = self.instance_circuit_breakers.get(port)
        if not breaker:
            return
        
        if success:
            breaker['failures'] = max(0, breaker['failures'] - 1)
            if breaker['state'] == 'half_open':
                breaker['success_count'] += 1
                if breaker['success_count'] >= self.circuit_breaker_recovery_requests:
                    breaker['state'] = 'closed'
                    breaker['success_count'] = 0
                    self.logger.info(f"Circuit breaker closed for instance {port}")
        else:
            breaker['failures'] += 1
            breaker['last_failure_time'] = time.time()
            
            if (breaker['state'] == 'closed' and 
                breaker['failures'] >= self.circuit_breaker_threshold):
                breaker['state'] = 'open'
                self.logger.warning(f"Circuit breaker opened for instance {port}")
            elif breaker['state'] == 'half_open':
                breaker['state'] = 'open'
                breaker['success_count'] = 0
    
    def _is_instance_available(self, port: int) -> bool:
        """Check if instance is available based on circuit breaker state"""
        breaker = self.instance_circuit_breakers.get(port)
        if not breaker:
            return True
        
        if breaker['state'] == 'closed':
            return True
        elif breaker['state'] == 'open':
            # Check if timeout has passed
            if time.time() - breaker['last_failure_time'] > self.circuit_breaker_timeout:
                breaker['state'] = 'half_open'
                breaker['success_count'] = 0
                self.logger.info(f"Circuit breaker half-opened for instance {port}")
                return True
            return False
        elif breaker['state'] == 'half_open':
            return True
        
        return False
    
    def _select_instance(self) -> Optional[OllamaInstance]:
        """Select the best instance based on load balancing strategy"""
        available_instances = [
            instance for instance in self.instance_manager.instances
            if (instance.status == "running" and 
                self._is_instance_available(instance.port))
        ]
        
        if not available_instances:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(available_instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(available_instances)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_RESPONSE_TIME:
            return self._weighted_response_time_select(available_instances)
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(available_instances)
        
        return available_instances[0]  # Fallback
    
    def _round_robin_select(self, instances: List[OllamaInstance]) -> OllamaInstance:
        """Round-robin instance selection"""
        instance = instances[self.current_instance_index % len(instances)]
        self.current_instance_index += 1
        return instance
    
    def _least_connections_select(self, instances: List[OllamaInstance]) -> OllamaInstance:
        """Select instance with least active connections"""
        return min(instances, 
                  key=lambda x: self.instance_connections.get(x.port, 0))
    
    def _weighted_response_time_select(self, instances: List[OllamaInstance]) -> OllamaInstance:
        """Select instance based on weighted response time"""
        # Lower response time = higher weight
        weights = []
        for instance in instances:
            response_time = instance.last_response_time or 0.1
            weight = 1.0 / max(response_time, 0.01)  # Avoid division by zero
            weights.append(weight)
        
        # Weighted random selection
        total_weight = sum(weights)
        r = random.uniform(0, total_weight)
        cumsum = 0
        
        for instance, weight in zip(instances, weights):
            cumsum += weight
            if r <= cumsum:
                return instance
        
        return instances[-1]  # Fallback
    
    async def generate_embedding(self, text: str, model: str) -> Tuple[Optional[List[float]], bool, str]:
        """
        Generate embedding using load-balanced instance selection
        
        Returns:
            Tuple of (embedding, success, error_message)
        """
        start_time = time.time()
        
        with self.stats_lock:
            self.stats.total_requests += 1
        
        # Select best instance
        instance = self._select_instance()
        if not instance:
            error_msg = "No available instances for embedding generation"
            self.logger.error(error_msg)
            with self.stats_lock:
                self.stats.failed_requests += 1
            return None, False, error_msg
        
        # Track connection
        with self.pool_lock:
            self.instance_connections[instance.port] = \
                self.instance_connections.get(instance.port, 0) + 1
        
        try:
            # Get HTTP client for this instance
            client = self.client_pool.get(instance.port)
            if not client:
                error_msg = f"No HTTP client available for instance {instance.port}"
                self.logger.error(error_msg)
                return None, False, error_msg
            
            # Use /api/embeddings endpoint directly (more reliable than /api/embed)
            embedding = None
            embeddings_error = None
            
            try:
                self.logger.debug(f"Attempting /api/embeddings on instance {instance.port} with model {model}")
                response = await client.post(
                    "/api/embeddings",
                    json={
                        "model": model,
                        "prompt": text,
                        "options": {
                            "num_thread": 4,
                            "num_gpu": 1
                        }
                    },
                    timeout=30.0
                )
                
                if response.status_code == 500:
                    # Handle 500 errors specifically
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", f"HTTP 500 from /api/embeddings on port {instance.port}")
                        if "out of memory" in error_msg.lower() or "cudaMalloc failed" in error_msg:
                            self.logger.error(f"GPU memory issue on instance {instance.port}: {error_msg}")
                        else:
                            self.logger.error(f"HTTP 500 error on instance {instance.port}: {error_msg}")
                        raise Exception(f"HTTP 500: {error_msg}")
                    except json.JSONDecodeError:
                        raise Exception(f"HTTP 500 Internal Server Error on instance {instance.port}")
                
                response.raise_for_status()
                data = response.json()
                
                # Handle /api/embeddings response structure
                embeddings_list = data.get("embedding")
                if embeddings_list:
                    embedding = embeddings_list
                else:
                    # Some versions return "embeddings" as array
                    embeddings_array = data.get("embeddings")
                    if embeddings_array and isinstance(embeddings_array, list) and len(embeddings_array) > 0:
                        embedding = embeddings_array[0]
                    else:
                        raise Exception("No embedding found in /api/embeddings response")
                
                self.logger.info(f"✅ Successfully used /api/embeddings on instance {instance.port} ({len(embedding)} dims)")
                
            except Exception as e:
                embeddings_error = e
                # Check if this is an event loop closed error
                if "Event loop is closed" in str(e):
                    self.logger.error(f"❌ Event loop closed on instance {instance.port} - HTTP client issue")
                    # Try to recreate the HTTP client for this instance
                    try:
                        with self.pool_lock:
                            if instance.port in self.client_pool:
                                old_client = self.client_pool[instance.port]
                                try:
                                    await old_client.aclose()
                                except:
                                    pass  # Ignore errors closing old client
                                
                                # Create new HTTP client
                                new_client = httpx.AsyncClient(
                                    base_url=instance.base_url,
                                    timeout=httpx.Timeout(60.0),
                                    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
                                )
                                self.client_pool[instance.port] = new_client
                                self.logger.info(f"🔄 Recreated HTTP client for instance {instance.port}")
                    except Exception as recreate_error:
                        self.logger.error(f"Failed to recreate HTTP client for instance {instance.port}: {recreate_error}")
                else:
                    self.logger.error(f"❌ /api/embeddings failed on instance {instance.port}: {e}")
                raise Exception(f"Embedding generation failed on instance {instance.port}: {e}")
            
            # Validate embedding
            if not embedding or not isinstance(embedding, list):
                error_msg = f"Invalid embedding format: {type(embedding)}"
                self.logger.error(error_msg)
                return None, False, error_msg
            
            # Update statistics and circuit breaker
            response_time = time.time() - start_time
            instance.last_response_time = response_time
            self._update_circuit_breaker(instance.port, True)
            
            with self.stats_lock:
                self.stats.successful_requests += 1
                self.stats.update_response_time(response_time)
            
            self.logger.debug(f"Generated embedding using instance {instance.port} "
                            f"({len(embedding)} dims, {response_time:.3f}s)")
            
            return embedding, True, ""
            
        except Exception as e:
            error_msg = f"Embedding generation failed on instance {instance.port}: {str(e)}"
            self.logger.error(error_msg)
            
            # Update circuit breaker
            self._update_circuit_breaker(instance.port, False)
            
            with self.stats_lock:
                self.stats.failed_requests += 1
            
            return None, False, error_msg
            
        finally:
            # Release connection
            with self.pool_lock:
                self.instance_connections[instance.port] = \
                    max(0, self.instance_connections.get(instance.port, 0) - 1)
    
    async def batch_generate_embeddings(self, requests: List[Tuple[str, str]]) -> List[Tuple[Optional[List[float]], bool, str]]:
        """
        Generate multiple embeddings concurrently using all available instances
        
        Args:
            requests: List of (text, model) tuples
            
        Returns:
            List of (embedding, success, error_message) tuples
        """
        if not requests:
            return []
        
        self.logger.info(f"Generating {len(requests)} embeddings using load balancer")
        
        # Create tasks for concurrent processing
        tasks = []
        for text, model in requests:
            task = asyncio.create_task(self.generate_embedding(text, model))
            tasks.append(task)
        
        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append((None, False, str(result)))
            else:
                processed_results.append(result)
        
        successful = sum(1 for _, success, _ in processed_results if success)
        self.logger.info(f"Batch embedding completed: {successful}/{len(requests)} successful")
        
        return processed_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive load balancer statistics"""
        with self.stats_lock:
            stats = {
                'strategy': self.strategy.value,
                'total_requests': self.stats.total_requests,
                'successful_requests': self.stats.successful_requests,
                'failed_requests': self.stats.failed_requests,
                'success_rate': (self.stats.successful_requests / max(self.stats.total_requests, 1)),
                'avg_response_time': self.stats.avg_response_time,
                'active_connections': sum(self.instance_connections.values()),
                'instance_stats': {}
            }
        
        # Add per-instance statistics
        for port, connections in self.instance_connections.items():
            breaker = self.instance_circuit_breakers.get(port, {})
            instance_weight = self.instance_weights.get(port, 1.0)
            
            stats['instance_stats'][port] = {
                'active_connections': connections,
                'weight': instance_weight,
                'circuit_breaker_state': breaker.get('state', 'unknown'),
                'circuit_breaker_failures': breaker.get('failures', 0)
            }
        
        return stats
    
    async def cleanup(self):
        """Cleanup resources and close connections"""
        self.logger.info("Cleaning up load balancer resources...")
        
        with self.pool_lock:
            for client in self.client_pool.values():
                try:
                    await client.aclose()
                except Exception as e:
                    self.logger.error(f"Error closing HTTP client: {e}")
            
            self.client_pool.clear()
            self.instance_connections.clear()
        
        self.logger.info("Load balancer cleanup completed")

# Global load balancer instance
_load_balancer = None

def get_load_balancer(strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS) -> EmbeddingLoadBalancer:
    """Get the global load balancer instance (singleton pattern)"""
    global _load_balancer
    if _load_balancer is None:
        _load_balancer = EmbeddingLoadBalancer(strategy)
    return _load_balancer
