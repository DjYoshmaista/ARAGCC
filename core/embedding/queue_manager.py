"""
Embedding queue manager for high-performance parallel embedding generation.
"""

import asyncio
import threading
import time
import pickle
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone
from queue import Queue, Empty
import logging

from .generator import EmbeddingGenerator

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingJob:
    """A job for embedding generation"""
    id: str
    chunk_id: str
    text: str
    model: str = "granite-embedding:latest"
    priority: int = 1
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'chunk_id': self.chunk_id,
            'text': self.text,
            'model': self.model,
            'priority': self.priority,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

@dataclass 
class EmbeddingResult:
    """Result from embedding generation"""
    job_id: str
    chunk_id: str
    embedding: List[float]
    model: str
    processing_time: float
    success: bool = True
    error: Optional[str] = None

class EmbeddingQueueManager:
    """
    Multi-threaded embedding queue manager for high-performance processing
    
    Features:
    - Parallel processing with configurable worker count
    - Job prioritization and queuing
    - Result handling and storage callbacks
    - Performance monitoring and statistics
    - Persistent state management
    """
    
    def __init__(self, 
                 model_gateway_url: str,
                 max_workers: int = 4,
                 max_queue_size: int = 10000,
                 state_file: str = "data/embedding_queue_state.pkl"):
        self.model_gateway_url = model_gateway_url
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.state_file = Path(state_file)
        
        # Queues
        self.input_queue = Queue(maxsize=max_queue_size)
        self.result_queue = Queue()
        
        # Worker management
        self.workers_running = False
        self.worker_threads = []
        self.result_thread = None
        
        # Statistics
        self.processing_stats = {
            'total_queued': 0,
            'total_processed': 0,
            'total_failed': 0,
            'workers_running': False,
            'num_workers': 0,
            'queue_size': 0,
            'result_queue_size': 0,
            'last_activity': None,
            'processing_rate': 0.0,
            'avg_processing_time': 0.0
        }
        
        # Performance tracking
        self.processing_times = []
        self.last_stats_update = time.time()
        
        # Locks
        self.stats_lock = threading.Lock()
        
        # Result callback
        self.result_callback = None
        
        # Load existing state
        self.load_state()
    
    def set_result_callback(self, callback):
        """Set callback function for handling embedding results"""
        self.result_callback = callback
    
    def start_workers(self):
        """Start worker threads for embedding generation"""
        if self.workers_running:
            logger.warning("Workers are already running")
            return
        
        self.workers_running = True
        
        # Start embedding worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._embedding_worker,
                name=f"EmbeddingWorker-{i+1}",
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
        
        # Start result processing thread
        self.result_thread = threading.Thread(
            target=self._result_worker,
            name="EmbeddingResultWorker",
            daemon=True
        )
        self.result_thread.start()
        
        with self.stats_lock:
            self.processing_stats['workers_running'] = True
            self.processing_stats['num_workers'] = self.max_workers
        
        logger.info(f"Started {self.max_workers} embedding workers")
    
    def stop_workers(self):
        """Stop worker threads"""
        self.workers_running = False
        
        # Add sentinel values to wake up workers
        for _ in range(self.max_workers):
            try:
                self.input_queue.put(None, timeout=1.0)
            except:
                pass
        
        # Wait for workers to finish
        for worker in self.worker_threads:
            worker.join(timeout=5.0)
        
        self.worker_threads.clear()
        
        # Stop result worker
        try:
            self.result_queue.put(None, timeout=1.0)
        except:
            pass
        
        if self.result_thread:
            self.result_thread.join(timeout=5.0)
        
        with self.stats_lock:
            self.processing_stats['workers_running'] = False
            self.processing_stats['num_workers'] = 0
        
        logger.info("Stopped embedding workers")
    
    def add_job(self, job: EmbeddingJob) -> bool:
        """Add embedding job to queue"""
        try:
            self.input_queue.put(job, timeout=1.0)
            
            with self.stats_lock:
                self.processing_stats['total_queued'] += 1
                self.processing_stats['queue_size'] = self.input_queue.qsize()
                self.processing_stats['last_activity'] = datetime.now(timezone.utc).isoformat()
            
            logger.debug(f"Added embedding job: {job.id}")
            return True
        except:
            logger.error(f"Failed to add embedding job - queue may be full")
            return False
    
    def add_jobs_batch(self, jobs: List[EmbeddingJob]) -> int:
        """Add multiple jobs to queue"""
        added_count = 0
        for job in jobs:
            if self.add_job(job):
                added_count += 1
            else:
                break
        
        logger.info(f"Added {added_count}/{len(jobs)} jobs to embedding queue")
        return added_count
    
    def _embedding_worker(self):
        """Worker thread for processing embedding jobs"""
        # Create embedding generator for this worker
        generator = EmbeddingGenerator(self.model_gateway_url)
        
        while self.workers_running:
            try:
                job = self.input_queue.get(timeout=1.0)
                
                if job is None:  # Sentinel value to stop worker
                    break
                
                start_time = time.time()
                
                # Generate embedding using asyncio
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    embedding = loop.run_until_complete(
                        generator.generate_embedding(job.text, job.model)
                    )
                    
                    loop.close()
                    
                    processing_time = time.time() - start_time
                    
                    if embedding:
                        result = EmbeddingResult(
                            job_id=job.id,
                            chunk_id=job.chunk_id,
                            embedding=embedding,
                            model=job.model,
                            processing_time=processing_time,
                            success=True
                        )
                        
                        # Track processing time
                        self.processing_times.append(processing_time)
                        if len(self.processing_times) > 100:
                            self.processing_times = self.processing_times[-100:]
                        
                        with self.stats_lock:
                            self.processing_stats['total_processed'] += 1
                            self.processing_stats['last_activity'] = datetime.now(timezone.utc).isoformat()
                            
                            if self.processing_times:
                                self.processing_stats['avg_processing_time'] = sum(self.processing_times) / len(self.processing_times)
                        
                        logger.debug(f"Generated embedding for job {job.id} in {processing_time:.3f}s")
                    else:
                        result = EmbeddingResult(
                            job_id=job.id,
                            chunk_id=job.chunk_id,
                            embedding=[],
                            model=job.model,
                            processing_time=processing_time,
                            success=False,
                            error="Failed to generate embedding"
                        )
                        
                        with self.stats_lock:
                            self.processing_stats['total_failed'] += 1
                        
                        logger.error(f"Failed to generate embedding for job {job.id}")
                    
                    # Add result to queue
                    self.result_queue.put(result)
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    result = EmbeddingResult(
                        job_id=job.id,
                        chunk_id=job.chunk_id,
                        embedding=[],
                        model=job.model,
                        processing_time=processing_time,
                        success=False,
                        error=str(e)
                    )
                    
                    self.result_queue.put(result)
                    
                    with self.stats_lock:
                        self.processing_stats['total_failed'] += 1
                    
                    logger.error(f"Embedding worker error for job {job.id}: {e}")
                
                # Update queue size
                with self.stats_lock:
                    self.processing_stats['queue_size'] = self.input_queue.qsize()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Embedding worker error: {e}")
        
        # Close generator
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(generator.close())
            loop.close()
        except:
            pass
        
        logger.debug("Embedding worker stopped")
    
    def _result_worker(self):
        """Worker thread for processing embedding results"""
        while self.workers_running:
            try:
                result = self.result_queue.get(timeout=1.0)
                
                if result is None:  # Sentinel value to stop worker
                    break
                
                # Call result callback if set
                if self.result_callback:
                    try:
                        self.result_callback(result)
                    except Exception as e:
                        logger.error(f"Result callback error: {e}")
                
                # Update stats
                with self.stats_lock:
                    self.processing_stats['result_queue_size'] = self.result_queue.qsize()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Result worker error: {e}")
        
        logger.debug("Result worker stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        with self.stats_lock:
            stats = self.processing_stats.copy()
            
            # Calculate processing rate
            current_time = time.time()
            time_diff = current_time - self.last_stats_update
            
            if time_diff >= 10.0 and stats['total_processed'] > 0:  # Update rate every 10 seconds
                processed_since_last = stats['total_processed']
                stats['processing_rate'] = processed_since_last / time_diff if time_diff > 0 else 0.0
                self.last_stats_update = current_time
            
            return stats
    
    def clear_stats(self):
        """Reset processing statistics"""
        with self.stats_lock:
            self.processing_stats.update({
                'total_queued': 0,
                'total_processed': 0,
                'total_failed': 0,
                'processing_rate': 0.0,
                'avg_processing_time': 0.0,
                'last_activity': None
            })
        
        self.processing_times.clear()
        logger.info("Cleared embedding processing statistics")
    
    def save_state(self):
        """Save queue state to file"""
        try:
            # Create directory if needed
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            
            state = {
                'stats': self.get_stats(),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            with open(self.state_file, 'wb') as f:
                pickle.dump(state, f)
            
            logger.debug(f"Saved embedding queue state to {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def load_state(self):
        """Load queue state from file"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'rb') as f:
                    state = pickle.load(f)
                
                # Restore statistics (but not running state)
                saved_stats = state.get('stats', {})
                with self.stats_lock:
                    self.processing_stats.update({
                        'total_queued': saved_stats.get('total_queued', 0),
                        'total_processed': saved_stats.get('total_processed', 0),
                        'total_failed': saved_stats.get('total_failed', 0),
                    })
                
                logger.info(f"Loaded embedding queue state from {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    
    def clear_state(self):
        """Clear saved state file"""
        try:
            if self.state_file.exists():
                self.state_file.unlink()
            self.clear_stats()
            logger.info("Cleared embedding queue state")
        except Exception as e:
            logger.error(f"Failed to clear state: {e}")

# Global singleton instance
_embedding_queue_manager = None
_queue_lock = threading.Lock()

def get_embedding_queue_manager(model_gateway_url: str = None, **kwargs) -> EmbeddingQueueManager:
    """Get or create singleton embedding queue manager"""
    global _embedding_queue_manager
    
    if _embedding_queue_manager is None:
        with _queue_lock:
            if _embedding_queue_manager is None:
                if model_gateway_url is None:
                    raise ValueError("model_gateway_url required for first initialization")
                _embedding_queue_manager = EmbeddingQueueManager(model_gateway_url, **kwargs)
    
    return _embedding_queue_manager