"""
Performance monitoring utilities.
"""

import time
import threading
from typing import Dict, Any
from dataclasses import dataclass

@dataclass 
class PerformanceMonitor:
    """Simple performance monitoring"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {}
        self.lock = threading.Lock()
    
    def record_metric(self, name: str, value: float):
        """Record a performance metric"""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        with self.lock:
            return {
                'elapsed_time': time.time() - self.start_time,
                'metrics': self.metrics.copy()
            }