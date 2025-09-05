"""
Enhanced progress tracking system with dual progress bars and real-time updates.
"""

import asyncio
import threading
import time
import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue, Empty
import logging

logger = logging.getLogger(__name__)

@dataclass
class DatabaseActivity:
    """Tracks database activity for timeout detection"""
    last_postgres_activity: datetime = None
    last_vector_activity: datetime = None
    postgres_operations: int = 0
    vector_operations: int = 0
    
    def __post_init__(self):
        now = datetime.now(timezone.utc)
        if self.last_postgres_activity is None:
            self.last_postgres_activity = now
        if self.last_vector_activity is None:
            self.last_vector_activity = now

@dataclass
class ProgressUpdate:
    """Progress update message"""
    timestamp: datetime
    update_type: str  # 'postgres', 'vector', 'file', 'status', 'embedding'
    data: Dict[str, Any]
    task_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'update_type': self.update_type,
            'data': self.data,
            'task_id': self.task_id
        }

class ProgressStats:
    """Container for progress statistics"""
    
    def __init__(self):
        self.lock = threading.Lock()
        
        # File and chunk statistics
        self.total_files = 0
        self.processed_files = 0
        self.failed_files = 0
        self.total_chunks = 0
        self.processed_chunks = 0
        
        # Embedding statistics
        self.total_embeddings_needed = 0
        self.embeddings_generated = 0
        self.embeddings_stored = 0
        self.embedding_failures = 0
        
        # Database statistics
        self.documents_stored = 0
        self.chunks_stored = 0
        self.vectors_stored = 0
        
        # Performance metrics
        self.start_time = time.time()
        self.last_update = time.time()
        self.files_per_second = 0.0
        self.embeddings_per_second = 0.0
        
        # Status tracking
        self.is_active = True
        self.is_paused = False
        self.current_phase = "initializing"
        
    def update_file_progress(self, processed: int = None, failed: int = None, total: int = None):
        """Update file processing progress"""
        with self.lock:
            if total is not None:
                self.total_files = max(self.total_files, total)
            if processed is not None:
                self.processed_files = processed
            if failed is not None:
                self.failed_files = failed
            
            self._update_performance_metrics()
    
    def update_chunk_progress(self, processed: int = None, total: int = None):
        """Update chunk processing progress"""
        with self.lock:
            if total is not None:
                self.total_chunks = max(self.total_chunks, total)
            if processed is not None:
                self.processed_chunks = processed
                
    def update_embedding_progress(self, generated: int = None, stored: int = None, 
                                 failed: int = None, total_needed: int = None):
        """Update embedding processing progress"""
        with self.lock:
            if total_needed is not None:
                self.total_embeddings_needed = max(self.total_embeddings_needed, total_needed)
            if generated is not None:
                self.embeddings_generated = generated
            if stored is not None:
                self.embeddings_stored = stored
            if failed is not None:
                self.embedding_failures = failed
            
            self._update_performance_metrics()
    
    def update_database_progress(self, documents: int = None, chunks: int = None, vectors: int = None):
        """Update database storage progress"""
        with self.lock:
            if documents is not None:
                self.documents_stored = documents
            if chunks is not None:
                self.chunks_stored = chunks
            if vectors is not None:
                self.vectors_stored = vectors
    
    def _update_performance_metrics(self):
        """Update performance metrics (called with lock held)"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if elapsed > 0:
            self.files_per_second = self.processed_files / elapsed
            self.embeddings_per_second = self.embeddings_generated / elapsed
        
        self.last_update = current_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics as dictionary"""
        with self.lock:
            elapsed = time.time() - self.start_time
            return {
                # File progress
                'files': {
                    'total': self.total_files,
                    'processed': self.processed_files,
                    'failed': self.failed_files,
                    'remaining': max(0, self.total_files - self.processed_files - self.failed_files),
                    'progress_percent': (self.processed_files / max(1, self.total_files)) * 100
                },
                
                # Chunk progress  
                'chunks': {
                    'total': self.total_chunks,
                    'processed': self.processed_chunks,
                    'remaining': max(0, self.total_chunks - self.processed_chunks),
                    'progress_percent': (self.processed_chunks / max(1, self.total_chunks)) * 100
                },
                
                # Embedding progress
                'embeddings': {
                    'total_needed': self.total_embeddings_needed,
                    'generated': self.embeddings_generated,
                    'stored': self.embeddings_stored,
                    'failed': self.embedding_failures,
                    'remaining': max(0, self.total_embeddings_needed - self.embeddings_generated),
                    'progress_percent': (self.embeddings_generated / max(1, self.total_embeddings_needed)) * 100
                },
                
                # Database progress
                'database': {
                    'documents_stored': self.documents_stored,
                    'chunks_stored': self.chunks_stored,
                    'vectors_stored': self.vectors_stored
                },
                
                # Performance metrics
                'performance': {
                    'elapsed_time': elapsed,
                    'files_per_second': self.files_per_second,
                    'embeddings_per_second': self.embeddings_per_second,
                    'last_update': datetime.fromtimestamp(self.last_update, tz=timezone.utc).isoformat()
                },
                
                # Status
                'status': {
                    'is_active': self.is_active,
                    'is_paused': self.is_paused,
                    'current_phase': self.current_phase
                }
            }

class ProgressTracker:
    """
    Enhanced progress tracker with dual progress bars and real-time monitoring
    """
    
    def __init__(self, task_id: str, heartbeat_timeout: int = 300):
        self.task_id = task_id
        self.heartbeat_timeout = heartbeat_timeout
        
        # Progress statistics
        self.stats = ProgressStats()
        
        # Activity tracking
        self.database_activity = DatabaseActivity()
        
        # Update queues
        self.update_queue = Queue()
        self.callback_queue = Queue()
        
        # Threading
        self.active = False
        self.worker_thread = None
        self.callback_thread = None
        self.lock = threading.Lock()
        
        # Callbacks for progress updates
        self.progress_callbacks: List[Callable] = []
        
        # State file for persistence
        self.state_file = Path(f"data/progress_{task_id}.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Progress tracker initialized for task {task_id}")
    
    def add_progress_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for progress updates"""
        self.progress_callbacks.append(callback)
    
    def start(self):
        """Start the progress tracker"""
        with self.lock:
            if self.active:
                return
            
            self.active = True
            self.stats.is_active = True
            
            # Start worker threads
            self.worker_thread = threading.Thread(
                target=self._progress_worker,
                name=f"ProgressWorker-{self.task_id}",
                daemon=True
            )
            self.worker_thread.start()
            
            self.callback_thread = threading.Thread(
                target=self._callback_worker,
                name=f"CallbackWorker-{self.task_id}",
                daemon=True
            )
            self.callback_thread.start()
            
            logger.info(f"Progress tracker started for task {self.task_id}")
    
    def stop(self):
        """Stop the progress tracker"""
        with self.lock:
            if not self.active:
                return
            
            self.active = False
            self.stats.is_active = False
            
            # Add sentinel values to queues
            self.update_queue.put(None)
            self.callback_queue.put(None)
            
            logger.info(f"Progress tracker stopped for task {self.task_id}")
    
    def pause(self):
        """Pause progress tracking"""
        self.stats.is_paused = True
        logger.info(f"Progress tracking paused for task {self.task_id}")
    
    def resume(self):
        """Resume progress tracking"""
        self.stats.is_paused = False
        logger.info(f"Progress tracking resumed for task {self.task_id}")
    
    def record_file_activity(self, activity_type: str, data: Dict[str, Any]):
        """Record file processing activity"""
        update = ProgressUpdate(
            timestamp=datetime.now(timezone.utc),
            update_type='file',
            data={'activity_type': activity_type, **data},
            task_id=self.task_id
        )
        self.update_queue.put(update)
    
    def record_database_activity(self, db_type: str, operation: str, data: Dict[str, Any]):
        """Record database activity"""
        update = ProgressUpdate(
            timestamp=datetime.now(timezone.utc),
            update_type=db_type,  # 'postgres' or 'vector'
            data={'operation': operation, **data},
            task_id=self.task_id
        )
        self.update_queue.put(update)
    
    def record_embedding_activity(self, activity_type: str, data: Dict[str, Any]):
        """Record embedding processing activity"""
        update = ProgressUpdate(
            timestamp=datetime.now(timezone.utc),
            update_type='embedding',
            data={'activity_type': activity_type, **data},
            task_id=self.task_id
        )
        self.update_queue.put(update)
    
    def update_totals(self, total_files: int = None, total_chunks: int = None, 
                     total_embeddings: int = None):
        """Update total counts"""
        if total_files is not None:
            self.stats.update_file_progress(total=total_files)
        if total_chunks is not None:
            self.stats.update_chunk_progress(total=total_chunks)
        if total_embeddings is not None:
            self.stats.update_embedding_progress(total_needed=total_embeddings)
        
        # Trigger callback
        self.callback_queue.put(self.stats.get_stats())
    
    def _progress_worker(self):
        """Worker thread for processing progress updates"""
        logger.debug(f"Progress worker started for task {self.task_id}")
        
        while self.active:
            try:
                update = self.update_queue.get(timeout=1.0)
                
                if update is None:  # Sentinel value
                    break
                
                self._process_update(update)
                
                # Save state periodically
                if time.time() % 30 < 1:  # Every ~30 seconds
                    self._save_state()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in progress worker: {e}")
        
        logger.debug(f"Progress worker stopped for task {self.task_id}")
    
    def _callback_worker(self):
        """Worker thread for handling progress callbacks"""
        logger.debug(f"Callback worker started for task {self.task_id}")
        
        while self.active:
            try:
                stats = self.callback_queue.get(timeout=1.0)
                
                if stats is None:  # Sentinel value
                    break
                
                # Call all registered callbacks
                for callback in self.progress_callbacks:
                    try:
                        callback(stats)
                    except Exception as e:
                        logger.error(f"Error in progress callback: {e}")
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in callback worker: {e}")
        
        logger.debug(f"Callback worker stopped for task {self.task_id}")
    
    def _process_update(self, update: ProgressUpdate):
        """Process a progress update"""
        try:
            if update.update_type == 'file':
                self._process_file_update(update)
            elif update.update_type == 'postgres':
                self._process_postgres_update(update)
            elif update.update_type == 'vector':
                self._process_vector_update(update)
            elif update.update_type == 'embedding':
                self._process_embedding_update(update)
            
            # Update activity tracking
            now = datetime.now(timezone.utc)
            if update.update_type == 'postgres':
                self.database_activity.last_postgres_activity = now
                self.database_activity.postgres_operations += 1
            elif update.update_type in ['vector', 'embedding']:
                self.database_activity.last_vector_activity = now
                self.database_activity.vector_operations += 1
            
            # Trigger callback with current stats
            self.callback_queue.put(self.stats.get_stats())
            
        except Exception as e:
            logger.error(f"Error processing update: {e}")
    
    def _process_file_update(self, update: ProgressUpdate):
        """Process file-related updates"""
        data = update.data
        activity_type = data.get('activity_type')
        
        if activity_type == 'file_discovered':
            # Increment total files
            self.stats.update_file_progress(total=self.stats.total_files + 1)
        elif activity_type == 'file_processed':
            # Increment processed files
            self.stats.update_file_progress(processed=self.stats.processed_files + 1)
        elif activity_type == 'file_failed':
            # Increment failed files
            self.stats.update_file_progress(failed=self.stats.failed_files + 1)
        elif activity_type == 'chunks_created':
            # Add chunks from this file
            chunk_count = data.get('chunk_count', 0)
            self.stats.update_chunk_progress(total=self.stats.total_chunks + chunk_count)
    
    def _process_postgres_update(self, update: ProgressUpdate):
        """Process PostgreSQL-related updates"""
        data = update.data
        operation = data.get('operation')
        
        if operation == 'document_stored':
            self.stats.update_database_progress(documents=self.stats.documents_stored + 1)
        elif operation == 'chunk_stored':
            self.stats.update_database_progress(chunks=self.stats.chunks_stored + 1)
            self.stats.update_chunk_progress(processed=self.stats.processed_chunks + 1)
    
    def _process_vector_update(self, update: ProgressUpdate):
        """Process vector database updates"""
        data = update.data
        operation = data.get('operation')
        
        if operation == 'vector_stored':
            self.stats.update_database_progress(vectors=self.stats.vectors_stored + 1)
            self.stats.update_embedding_progress(stored=self.stats.embeddings_stored + 1)
    
    def _process_embedding_update(self, update: ProgressUpdate):
        """Process embedding-related updates"""
        data = update.data
        activity_type = data.get('activity_type')
        
        if activity_type == 'embedding_generated':
            self.stats.update_embedding_progress(generated=self.stats.embeddings_generated + 1)
        elif activity_type == 'embedding_failed':
            self.stats.update_embedding_progress(failed=self.stats.embedding_failures + 1)
        elif activity_type == 'embedding_queued':
            # Increment total embeddings needed
            self.stats.update_embedding_progress(
                total_needed=self.stats.total_embeddings_needed + 1
            )
    
    def _save_state(self):
        """Save current state to file"""
        try:
            # Convert database activity to JSON-serializable format
            db_activity_dict = asdict(self.database_activity)
            if db_activity_dict.get('last_postgres_activity'):
                db_activity_dict['last_postgres_activity'] = db_activity_dict['last_postgres_activity'].isoformat()
            if db_activity_dict.get('last_vector_activity'):
                db_activity_dict['last_vector_activity'] = db_activity_dict['last_vector_activity'].isoformat()
            
            state = {
                'task_id': self.task_id,
                'stats': self.stats.get_stats(),
                'database_activity': db_activity_dict,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save progress state: {e}")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current progress statistics"""
        return self.stats.get_stats()

# Global registry for progress trackers
_progress_trackers: Dict[str, ProgressTracker] = {}
_tracker_lock = threading.Lock()

def get_progress_tracker(task_id: str, heartbeat_timeout: int = 300) -> ProgressTracker:
    """Get or create progress tracker for task"""
    with _tracker_lock:
        if task_id not in _progress_trackers:
            _progress_trackers[task_id] = ProgressTracker(task_id, heartbeat_timeout)
        return _progress_trackers[task_id]

def cleanup_progress_tracker(task_id: str):
    """Clean up progress tracker for task"""
    with _tracker_lock:
        if task_id in _progress_trackers:
            tracker = _progress_trackers[task_id]
            tracker.stop()
            del _progress_trackers[task_id]
            logger.info(f"Cleaned up progress tracker for task {task_id}")