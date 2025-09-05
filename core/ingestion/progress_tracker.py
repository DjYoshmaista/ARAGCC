"""
Real-time progress tracking system with heartbeat monitoring and database activity detection
Provides dual progress tracking for PostgreSQL and vector database operations
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
    update_type: str  # 'postgres', 'vector', 'file', 'status'
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

class ProgressTracker:
    """
    Real-time progress tracker with heartbeat monitoring
    Tracks both PostgreSQL text processing and vector embedding operations
    """
    
    def __init__(self, task_id: str, heartbeat_timeout: int = 300):
        """
        Initialize progress tracker
        
        Args:
            task_id: Unique task identifier
            heartbeat_timeout: Timeout in seconds for database inactivity (default 5 minutes)
        """
        self.task_id = task_id
        self.heartbeat_timeout = heartbeat_timeout
        self.logger = logging.getLogger(f"progress_tracker_{task_id}")
        
        # Progress tracking
        self.db_activity = DatabaseActivity()
        self.progress_queue = Queue()
        self.subscribers: List[Callable] = []
        
        # State tracking
        self.postgres_stats = {
            'files_processed': 0,
            'chunks_created': 0,
            'documents_stored': 0,
            'current_file': None,
            'start_time': datetime.now(timezone.utc),
            'last_activity': datetime.now(timezone.utc)
        }
        
        self.vector_stats = {
            'embeddings_generated': 0,
            'embeddings_stored': 0,
            'queue_size': 0,
            'processing_rate': 0.0,
            'start_time': datetime.now(timezone.utc),
            'last_activity': datetime.now(timezone.utc)
        }
        
        # Control flags
        self.active = False
        self.paused = False
        self.heartbeat_thread = None
        self.message_thread = None
        
        self.logger.info(f"Progress tracker initialized for task {task_id}")
    
    def start(self):
        """Start the progress tracking system"""
        if self.active:
            self.logger.info("Progress tracking already active, not starting again")
            return
        
        self.active = True
        
        # Start heartbeat monitor
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_monitor,
            daemon=True,
            name=f"HeartbeatMonitor-{self.task_id}"
        )
        self.heartbeat_thread.start()
        
        # Start message processor
        self.message_thread = threading.Thread(
            target=self._message_processor,
            daemon=True,
            name=f"MessageProcessor-{self.task_id}"
        )
        self.message_thread.start()
        
        self.logger.info("Progress tracking started")
    
    def stop(self):
        """Stop the progress tracking system"""
        self.active = False
        
        # Signal stop
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=2.0)
        
        if self.message_thread and self.message_thread.is_alive():
            self.message_thread.join(timeout=2.0)
        
        self.logger.info("Progress tracking stopped")
    
    def record_postgres_activity(self, activity_type: str, data: Dict[str, Any]):
        """Record PostgreSQL database activity"""
        now = datetime.now(timezone.utc)
        self.db_activity.last_postgres_activity = now
        self.db_activity.postgres_operations += 1
        self.postgres_stats['last_activity'] = now
        
        # Update specific stats
        if activity_type == 'document_stored':
            self.postgres_stats['documents_stored'] += 1
        elif activity_type == 'chunk_created':
            self.postgres_stats['chunks_created'] += data.get('count', 1)
        elif activity_type == 'file_processed':
            self.postgres_stats['files_processed'] += 1
            self.postgres_stats['current_file'] = data.get('filename')
        
        # Send update
        update = ProgressUpdate(
            timestamp=now,
            update_type='postgres',
            data={
                'activity_type': activity_type,
                'stats': self.postgres_stats.copy(),
                **data
            },
            task_id=self.task_id
        )
        
        self.progress_queue.put(update)
        self.logger.debug(f"PostgreSQL activity recorded: {activity_type} with data: {data}")
    
    def record_vector_activity(self, activity_type: str, data: Dict[str, Any]):
        """Record vector database activity"""
        now = datetime.now(timezone.utc)
        self.db_activity.last_vector_activity = now
        self.db_activity.vector_operations += 1
        self.vector_stats['last_activity'] = now
        
        # Update specific stats
        if activity_type == 'embedding_generated':
            self.vector_stats['embeddings_generated'] += data.get('count', 1)
        elif activity_type == 'embedding_stored':
            self.vector_stats['embeddings_stored'] += data.get('count', 1)
        elif activity_type == 'queue_update':
            self.vector_stats['queue_size'] = data.get('queue_size', 0)
            self.vector_stats['processing_rate'] = data.get('processing_rate', 0.0)
        
        # Send update
        update = ProgressUpdate(
            timestamp=now,
            update_type='vector',
            data={
                'activity_type': activity_type,
                'stats': self.vector_stats.copy(),
                **data
            },
            task_id=self.task_id
        )
        
        self.progress_queue.put(update)
        self.logger.debug(f"Vector activity recorded: {activity_type}")
    
    def record_file_activity(self, activity_type: str, data: Dict[str, Any]):
        """Record file processing activity"""
        update = ProgressUpdate(
            timestamp=datetime.now(timezone.utc),
            update_type='file',
            data={
                'activity_type': activity_type,
                **data
            },
            task_id=self.task_id
        )
        
        self.progress_queue.put(update)
        self.logger.debug(f"File activity recorded: {activity_type}")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current progress statistics"""
        now = datetime.now(timezone.utc)
        
        postgres_elapsed = (now - self.postgres_stats['start_time']).total_seconds()
        vector_elapsed = (now - self.vector_stats['start_time']).total_seconds()
        
        return {
            'task_id': self.task_id,
            'active': self.active,
            'paused': self.paused,
            'postgres': {
                **self.postgres_stats,
                'elapsed_seconds': postgres_elapsed,
                'files_per_second': self.postgres_stats['files_processed'] / max(postgres_elapsed, 1)
            },
            'vector': {
                **self.vector_stats,
                'elapsed_seconds': vector_elapsed,
                'embeddings_per_second': self.vector_stats['embeddings_generated'] / max(vector_elapsed, 1)
            },
            'database_activity': {
                'postgres_operations': self.db_activity.postgres_operations,
                'vector_operations': self.db_activity.vector_operations,
                'last_postgres_activity': self.db_activity.last_postgres_activity.isoformat(),
                'last_vector_activity': self.db_activity.last_vector_activity.isoformat()
            }
        }
    
    def subscribe_to_updates(self, callback: Callable[[ProgressUpdate], None]):
        """Subscribe to progress updates"""
        self.subscribers.append(callback)
    
    def _heartbeat_monitor(self):
        """Monitor database activity and detect timeouts"""
        while self.active:
            try:
                now = datetime.now(timezone.utc)
                
                # Check PostgreSQL activity timeout
                postgres_inactive = (now - self.db_activity.last_postgres_activity).total_seconds()
                vector_inactive = (now - self.db_activity.last_vector_activity).total_seconds()
                
                timeout_detected = False
                timeout_info = {}
                
                if postgres_inactive > self.heartbeat_timeout:
                    timeout_info['postgres_inactive_seconds'] = postgres_inactive
                    timeout_detected = True
                
                if vector_inactive > self.heartbeat_timeout:
                    timeout_info['vector_inactive_seconds'] = vector_inactive
                    timeout_detected = True
                
                if timeout_detected and not self.paused:
                    self.logger.warning(f"Database inactivity detected: {timeout_info}")
                    
                    # Send timeout notification
                    update = ProgressUpdate(
                        timestamp=now,
                        update_type='timeout',
                        data={
                            'timeout_detected': True,
                            'timeout_info': timeout_info,
                            'heartbeat_timeout': self.heartbeat_timeout,
                            'current_stats': self.get_current_stats()
                        },
                        task_id=self.task_id
                    )
                    
                    self.progress_queue.put(update)
                    
                    # Pause until user decides
                    self.paused = True
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Heartbeat monitor error: {e}")
                time.sleep(5)
    
    def _message_processor(self):
        """Process and distribute progress messages"""
        while self.active:
            try:
                try:
                    update = self.progress_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # Send to all subscribers
                for callback in self.subscribers:
                    try:
                        callback(update)
                    except Exception as e:
                        self.logger.error(f"Subscriber callback error: {e}")
                
                self.progress_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Message processor error: {e}")
                time.sleep(1)
    
    def resume(self):
        """Resume processing after timeout"""
        self.paused = False
        # Update activity timestamps to reset timeout
        now = datetime.now(timezone.utc)
        self.db_activity.last_postgres_activity = now
        self.db_activity.last_vector_activity = now
        self.logger.info("Processing resumed")
    
    def pause(self):
        """Pause processing"""
        self.paused = True
        self.logger.info("Processing paused")

# Global progress tracker registry
_progress_trackers: Dict[str, ProgressTracker] = {}

def get_progress_tracker(task_id: str, heartbeat_timeout: int = 300) -> ProgressTracker:
    """Get or create a progress tracker for a task"""
    if task_id not in _progress_trackers:
        _progress_trackers[task_id] = ProgressTracker(task_id, heartbeat_timeout)
        logging.getLogger("progress_tracker").info(f"Created new progress tracker for task {task_id}")
    else:
        logging.getLogger("progress_tracker").info(f"Retrieved existing progress tracker for task {task_id}")
    return _progress_trackers[task_id]

def cleanup_progress_tracker(task_id: str):
    """Clean up a progress tracker"""
    if task_id in _progress_trackers:
        _progress_trackers[task_id].stop()
        del _progress_trackers[task_id]