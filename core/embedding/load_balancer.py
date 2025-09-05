"""
Load balancer for distributing embedding requests across multiple instances.
"""

import random
import threading
import time
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_LOADED = "least_loaded"

@dataclass
class InstanceMetrics:
    """Metrics for a model instance"""
    url: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_used: Optional[float] = None
    current_load: int = 0
    weight: float = 1.0
    is_healthy: bool = True
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

class EmbeddingLoadBalancer:
    """Load balancer for distributing embedding generation across multiple instances"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.instances: List[str] = []
        self.metrics: Dict[str, InstanceMetrics] = {}
        self.current_index = 0
        self.lock = threading.Lock()
        
    def add_instance(self, url: str, weight: float = 1.0):
        """Add a new instance to the load balancer"""
        with self.lock:
            if url not in self.instances:
                self.instances.append(url)
                self.metrics[url] = InstanceMetrics(url=url, weight=weight)
                logger.info(f"Added instance to load balancer: {url}")
    
    def get_next_instance(self) -> Optional[str]:
        """Get the next instance based on the load balancing strategy"""
        with self.lock:
            if not self.instances:
                return None
            
            healthy_instances = [url for url in self.instances if self.metrics[url].is_healthy]
            
            if not healthy_instances:
                healthy_instances = self.instances
            
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                instance = healthy_instances[self.current_index % len(healthy_instances)]
                self.current_index += 1
                return instance
            elif self.strategy == LoadBalancingStrategy.RANDOM:
                return random.choice(healthy_instances)
            elif self.strategy == LoadBalancingStrategy.LEAST_LOADED:
                return min(healthy_instances, key=lambda url: self.metrics[url].current_load)
            else:
                return healthy_instances[0]
    
    def record_request(self, url: str, success: bool, response_time: float):
        """Record metrics for a request"""
        with self.lock:
            if url in self.metrics:
                metrics = self.metrics[url]
                metrics.total_requests += 1
                metrics.last_used = time.time()
                
                if success:
                    metrics.successful_requests += 1
                else:
                    metrics.failed_requests += 1
                
                # Update average response time
                alpha = 0.1  # Smoothing factor
                if metrics.avg_response_time == 0:
                    metrics.avg_response_time = response_time
                else:
                    metrics.avg_response_time = (
                        alpha * response_time + 
                        (1 - alpha) * metrics.avg_response_time
                    )
                
                # Update health status based on success rate
                if metrics.total_requests >= 10:
                    failure_rate = 1.0 - metrics.success_rate
                    metrics.is_healthy = failure_rate <= 0.5
    
    def get_instance_count(self) -> int:
        """Get total number of instances"""
        return len(self.instances)

# Global singleton
_load_balancer = None
_lb_lock = threading.Lock()

def get_load_balancer(strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN) -> EmbeddingLoadBalancer:
    """Get or create singleton load balancer"""
    global _load_balancer
    
    if _load_balancer is None:
        with _lb_lock:
            if _load_balancer is None:
                _load_balancer = EmbeddingLoadBalancer(strategy)
    
    return _load_balancer