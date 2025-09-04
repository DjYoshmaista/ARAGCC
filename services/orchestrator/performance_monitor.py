"""
Comprehensive Performance Monitor for Multi-Instance Embedding Generation

This module provides real-time performance monitoring, GPU utilization tracking,
bottleneck detection, and automated alerting for the embedding generation system.
"""

import time
import threading
import logging
import json
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from collections import deque
import statistics

@dataclass
class GPUMetrics:
    """GPU performance metrics"""
    index: int
    name: str
    utilization_percent: int
    memory_used_mb: int
    memory_total_mb: int
    memory_free_mb: int
    temperature: Optional[int] = None
    power_usage: Optional[int] = None

@dataclass
class SystemMetrics:
    """Overall system performance metrics"""
    timestamp: datetime
    
    # GPU Metrics
    gpu_metrics: List[GPUMetrics]
    total_gpu_memory_used_mb: int
    total_gpu_memory_total_mb: int
    avg_gpu_utilization: float
    
    # Ollama Instances
    ollama_instances_running: int
    ollama_instances_healthy: int
    
    # Embedding Performance
    embedding_queue_size: int
    embedding_processing_rate: float  # embeddings/second
    embedding_success_rate: float
    avg_embedding_time_ms: float
    
    # Qdrant Performance
    qdrant_insertion_rate: float  # vectors/second
    qdrant_batch_efficiency: float
    
    # Overall Throughput
    chunks_processed_per_second: float
    files_processed_per_second: float
    
    # Resource Utilization
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float

@dataclass
class PerformanceAlert:
    """Performance alert definition"""
    timestamp: datetime
    alert_type: str  # "warning", "critical", "info"
    category: str  # "gpu", "embedding", "qdrant", "system"
    message: str
    metric_value: Optional[float] = None
    threshold_value: Optional[float] = None
    
class PerformanceMonitor:
    """
    Comprehensive performance monitor for the multi-instance embedding system
    
    Features:
    - Real-time GPU monitoring and memory tracking
    - Ollama instance health monitoring
    - Embedding throughput and bottleneck detection
    - Automated alerting and threshold monitoring
    - Performance trend analysis and reporting
    """
    
    def __init__(self, monitoring_interval: float = 5.0):
        self.logger = logging.getLogger("performance_monitor")
        self.monitoring_interval = monitoring_interval
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Metrics storage (keep last 1000 measurements for trend analysis)
        self.metrics_history: deque[SystemMetrics] = deque(maxlen=1000)
        self.alerts_history: deque[PerformanceAlert] = deque(maxlen=500)
        
        # Performance thresholds
        self.thresholds = {
            'gpu_utilization_low': 50.0,      # Alert if avg GPU util < 50%
            'gpu_memory_high': 95.0,          # Alert if GPU memory > 95%
            'embedding_rate_low': 5.0,        # Alert if < 5 embeddings/sec
            'queue_backlog_high': 10000,      # Alert if queue > 10k items
            'success_rate_low': 95.0,         # Alert if success rate < 95%
            'response_time_high': 2000.0,     # Alert if avg response > 2s
        }
        
        # Instance references (set by external components)
        self.instance_manager = None
        self.embedding_queue = None
        self.load_balancer = None
        
        # Performance tracking
        self.last_metrics_time = time.time()
        self.performance_baseline = None
        
        self.logger.info("Performance Monitor initialized")
    
    def set_components(self, instance_manager=None, embedding_queue=None, load_balancer=None):
        """Set references to monitored components"""
        self.instance_manager = instance_manager
        self.embedding_queue = embedding_queue
        self.load_balancer = load_balancer
        self.logger.info("Performance monitor components configured")
    
    def start_monitoring(self):
        """Start background performance monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                start_time = time.time()
                
                # Collect metrics
                metrics = self._collect_system_metrics()
                if metrics:
                    self.metrics_history.append(metrics)
                    
                    # Check for alerts
                    alerts = self._analyze_metrics_for_alerts(metrics)
                    for alert in alerts:
                        self.alerts_history.append(alert)
                        self._log_alert(alert)
                    
                    # Log performance summary periodically
                    if len(self.metrics_history) % 12 == 0:  # Every 12 intervals (1 minute at 5s intervals)
                        self._log_performance_summary(metrics)
                
                # Sleep for remaining interval time
                elapsed = time.time() - start_time
                sleep_time = max(0, self.monitoring_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> Optional[SystemMetrics]:
        """Collect comprehensive system metrics"""
        try:
            # GPU metrics
            gpu_metrics = self._get_gpu_metrics()
            
            # System resources
            cpu_usage, memory_usage = self._get_system_resources()
            disk_io = self._get_disk_io()
            
            # Component-specific metrics
            embedding_metrics = self._get_embedding_metrics()
            ollama_metrics = self._get_ollama_metrics()
            qdrant_metrics = self._get_qdrant_metrics()
            
            # Calculate derived metrics
            total_gpu_memory_used = sum(gpu.memory_used_mb for gpu in gpu_metrics)
            total_gpu_memory_total = sum(gpu.memory_total_mb for gpu in gpu_metrics)
            avg_gpu_utilization = statistics.mean([gpu.utilization_percent for gpu in gpu_metrics]) if gpu_metrics else 0
            
            return SystemMetrics(
                timestamp=datetime.now(timezone.utc),
                gpu_metrics=gpu_metrics,
                total_gpu_memory_used_mb=total_gpu_memory_used,
                total_gpu_memory_total_mb=total_gpu_memory_total,
                avg_gpu_utilization=avg_gpu_utilization,
                ollama_instances_running=ollama_metrics.get('running', 0),
                ollama_instances_healthy=ollama_metrics.get('healthy', 0),
                embedding_queue_size=embedding_metrics.get('queue_size', 0),
                embedding_processing_rate=embedding_metrics.get('processing_rate', 0.0),
                embedding_success_rate=embedding_metrics.get('success_rate', 0.0),
                avg_embedding_time_ms=embedding_metrics.get('avg_time_ms', 0.0),
                qdrant_insertion_rate=qdrant_metrics.get('insertion_rate', 0.0),
                qdrant_batch_efficiency=qdrant_metrics.get('batch_efficiency', 0.0),
                chunks_processed_per_second=embedding_metrics.get('processing_rate', 0.0),
                files_processed_per_second=0.0,  # Would need file processing metrics
                cpu_usage_percent=cpu_usage,
                memory_usage_percent=memory_usage,
                disk_io_read_mb=disk_io.get('read_mb', 0.0),
                disk_io_write_mb=disk_io.get('write_mb', 0.0)
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return None
    
    def _get_gpu_metrics(self) -> List[GPUMetrics]:
        """Get GPU performance metrics"""
        try:
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return []
            
            gpu_metrics = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 5:
                        gpu_metrics.append(GPUMetrics(
                            index=int(parts[0]),
                            name=parts[1],
                            utilization_percent=int(parts[2]),
                            memory_used_mb=int(parts[3]),
                            memory_total_mb=int(parts[4]),
                            memory_free_mb=int(parts[4]) - int(parts[3]),
                            temperature=int(parts[5]) if len(parts) > 5 and parts[5].replace('.', '').isdigit() else None,
                            power_usage=int(float(parts[6])) if len(parts) > 6 and parts[6].replace('.', '').isdigit() else None
                        ))
            
            return gpu_metrics
            
        except Exception as e:
            self.logger.debug(f"Could not get GPU metrics: {e}")
            return []
    
    def _get_system_resources(self) -> Tuple[float, float]:
        """Get CPU and memory usage"""
        try:
            import psutil
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            return cpu_usage, memory_usage
        except ImportError:
            # Fallback using system commands
            try:
                # CPU usage
                cpu_result = subprocess.run(['top', '-bn1'], capture_output=True, text=True, timeout=5)
                cpu_usage = 0.0
                for line in cpu_result.stdout.split('\n'):
                    if 'Cpu(s):' in line:
                        # Parse CPU usage from top output
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if 'us' in part:
                                cpu_usage = float(parts[i-1].replace('%', ''))
                                break
                
                # Memory usage
                mem_result = subprocess.run(['free', '-m'], capture_output=True, text=True, timeout=5)
                memory_usage = 0.0
                for line in mem_result.stdout.split('\n'):
                    if line.startswith('Mem:'):
                        parts = line.split()
                        total = float(parts[1])
                        used = float(parts[2])
                        memory_usage = (used / total) * 100
                        break
                
                return cpu_usage, memory_usage
                
            except Exception as e:
                self.logger.debug(f"Could not get system resources: {e}")
                return 0.0, 0.0
    
    def _get_disk_io(self) -> Dict[str, float]:
        """Get disk I/O metrics"""
        try:
            import psutil
            disk_io = psutil.disk_io_counters()
            if disk_io:
                return {
                    'read_mb': disk_io.read_bytes / (1024 * 1024),
                    'write_mb': disk_io.write_bytes / (1024 * 1024)
                }
        except:
            pass
        
        return {'read_mb': 0.0, 'write_mb': 0.0}
    
    def _get_embedding_metrics(self) -> Dict[str, Any]:
        """Get embedding queue performance metrics"""
        if not self.embedding_queue:
            return {}
        
        try:
            stats = self.embedding_queue.get_stats()
            
            # Calculate rates
            processing_rate = stats.get('processing_rate', 0.0)
            total_requests = stats.get('total_processed', 0) + stats.get('total_failed', 0)
            success_rate = (stats.get('total_processed', 0) / max(total_requests, 1)) * 100
            
            # Average processing time
            processing_times = stats.get('processing_times', [])
            avg_time_ms = (statistics.mean(processing_times) * 1000) if processing_times else 0.0
            
            return {
                'queue_size': stats.get('queue_size', 0),
                'processing_rate': processing_rate,
                'success_rate': success_rate,
                'avg_time_ms': avg_time_ms,
                'total_processed': stats.get('total_processed', 0),
                'total_failed': stats.get('total_failed', 0)
            }
            
        except Exception as e:
            self.logger.debug(f"Could not get embedding metrics: {e}")
            return {}
    
    def _get_ollama_metrics(self) -> Dict[str, Any]:
        """Get Ollama instance metrics"""
        if not self.instance_manager:
            return {}
        
        try:
            stats = self.instance_manager.get_stats()
            return {
                'running': stats.get('running_instances', 0),
                'healthy': stats.get('loaded_instances', 0),
                'total': stats.get('total_instances', 0)
            }
        except Exception as e:
            self.logger.debug(f"Could not get Ollama metrics: {e}")
            return {}
    
    def _get_qdrant_metrics(self) -> Dict[str, Any]:
        """Get Qdrant performance metrics"""
        # For now, return empty dict - could be enhanced with Qdrant API calls
        return {
            'insertion_rate': 0.0,
            'batch_efficiency': 0.0
        }
    
    def _analyze_metrics_for_alerts(self, metrics: SystemMetrics) -> List[PerformanceAlert]:
        """Analyze metrics and generate alerts"""
        alerts = []
        timestamp = datetime.now(timezone.utc)
        
        # GPU utilization alerts
        if metrics.avg_gpu_utilization < self.thresholds['gpu_utilization_low']:
            alerts.append(PerformanceAlert(
                timestamp=timestamp,
                alert_type="warning",
                category="gpu",
                message=f"Low GPU utilization: {metrics.avg_gpu_utilization:.1f}%",
                metric_value=metrics.avg_gpu_utilization,
                threshold_value=self.thresholds['gpu_utilization_low']
            ))
        
        # GPU memory alerts
        gpu_memory_usage = (metrics.total_gpu_memory_used_mb / max(metrics.total_gpu_memory_total_mb, 1)) * 100
        if gpu_memory_usage > self.thresholds['gpu_memory_high']:
            alerts.append(PerformanceAlert(
                timestamp=timestamp,
                alert_type="critical",
                category="gpu",
                message=f"High GPU memory usage: {gpu_memory_usage:.1f}%",
                metric_value=gpu_memory_usage,
                threshold_value=self.thresholds['gpu_memory_high']
            ))
        
        # Embedding performance alerts
        if metrics.embedding_processing_rate < self.thresholds['embedding_rate_low']:
            alerts.append(PerformanceAlert(
                timestamp=timestamp,
                alert_type="warning",
                category="embedding",
                message=f"Low embedding processing rate: {metrics.embedding_processing_rate:.1f}/sec",
                metric_value=metrics.embedding_processing_rate,
                threshold_value=self.thresholds['embedding_rate_low']
            ))
        
        # Queue backlog alerts
        if metrics.embedding_queue_size > self.thresholds['queue_backlog_high']:
            alerts.append(PerformanceAlert(
                timestamp=timestamp,
                alert_type="warning",
                category="embedding",
                message=f"High embedding queue backlog: {metrics.embedding_queue_size} items",
                metric_value=float(metrics.embedding_queue_size),
                threshold_value=self.thresholds['queue_backlog_high']
            ))
        
        # Success rate alerts
        if metrics.embedding_success_rate < self.thresholds['success_rate_low']:
            alerts.append(PerformanceAlert(
                timestamp=timestamp,
                alert_type="critical",
                category="embedding",
                message=f"Low embedding success rate: {metrics.embedding_success_rate:.1f}%",
                metric_value=metrics.embedding_success_rate,
                threshold_value=self.thresholds['success_rate_low']
            ))
        
        return alerts
    
    def _log_alert(self, alert: PerformanceAlert):
        """Log performance alert"""
        level = logging.WARNING if alert.alert_type == "warning" else logging.ERROR
        self.logger.log(level, f"PERFORMANCE ALERT [{alert.category.upper()}]: {alert.message}")
    
    def _log_performance_summary(self, metrics: SystemMetrics):
        """Log periodic performance summary"""
        self.logger.info(
            f"Performance Summary - "
            f"GPU: {metrics.avg_gpu_utilization:.1f}% util, "
            f"{metrics.total_gpu_memory_used_mb}/{metrics.total_gpu_memory_total_mb}MB mem | "
            f"Embedding: {metrics.embedding_processing_rate:.1f}/sec, "
            f"{metrics.embedding_queue_size} queued | "
            f"Ollama: {metrics.ollama_instances_healthy}/{metrics.ollama_instances_running} healthy"
        )
    
    def get_performance_report(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.metrics_history:
            return {"error": "No metrics available"}
        
        # Filter metrics by duration
        cutoff_time = datetime.now(timezone.utc).timestamp() - (duration_minutes * 60)
        recent_metrics = [m for m in self.metrics_history 
                         if m.timestamp.timestamp() > cutoff_time]
        
        if not recent_metrics:
            return {"error": "No recent metrics available"}
        
        # Calculate averages and trends
        avg_gpu_util = statistics.mean([m.avg_gpu_utilization for m in recent_metrics])
        avg_embedding_rate = statistics.mean([m.embedding_processing_rate for m in recent_metrics])
        avg_queue_size = statistics.mean([m.embedding_queue_size for m in recent_metrics])
        avg_success_rate = statistics.mean([m.embedding_success_rate for m in recent_metrics])
        
        # Get recent alerts
        recent_alerts = [a for a in self.alerts_history 
                        if a.timestamp.timestamp() > cutoff_time]
        
        return {
            "report_period_minutes": duration_minutes,
            "metrics_count": len(recent_metrics),
            "averages": {
                "gpu_utilization_percent": avg_gpu_util,
                "embedding_processing_rate": avg_embedding_rate,
                "queue_size": avg_queue_size,
                "success_rate_percent": avg_success_rate
            },
            "current_status": asdict(recent_metrics[-1]) if recent_metrics else None,
            "alerts_count": len(recent_alerts),
            "recent_alerts": [asdict(alert) for alert in recent_alerts[-10:]]  # Last 10 alerts
        }
    
    def save_metrics_to_file(self, filepath: str):
        """Save metrics history to JSON file"""
        try:
            metrics_data = [asdict(m) for m in self.metrics_history]
            alerts_data = [asdict(a) for a in self.alerts_history]
            
            # Convert datetime objects to ISO strings for JSON serialization
            for metric in metrics_data:
                metric['timestamp'] = metric['timestamp'].isoformat()
            
            for alert in alerts_data:
                alert['timestamp'] = alert['timestamp'].isoformat()
            
            data = {
                "metrics": metrics_data,
                "alerts": alerts_data,
                "export_time": datetime.now(timezone.utc).isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.logger.info(f"Performance metrics saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")

# Global performance monitor instance
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor (singleton pattern)"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor