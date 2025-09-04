"""
Concurrent Processing Manager for AgenticRAG System

This module enables concurrent processing of document metadata and embeddings,
allowing both operations to happen simultaneously for improved performance.
"""

import asyncio
import threading
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import uuid

# Local imports
from database_utils import DatabaseManager, DocumentMetadata, ChunkMetadata
from embedding_queue import embedding_queue, EmbeddingJob, EmbeddingResult
from text_chunker import TextChunker

@dataclass
class ConcurrentProcessingTask:
    """A task for concurrent document and embedding processing"""
    task_id: str
    document_metadata: DocumentMetadata
    chunk_metadata_list: List[ChunkMetadata]
    chunk_texts: Dict[str, str]  # chunk_id -> text
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

class ConcurrentProcessingManager:
    """
    Manages concurrent processing of document metadata and embeddings.
    
    This class enables:
    1. Concurrent processing of document metadata in PostgreSQL
    2. Concurrent generation of embeddings using GPU instances
    3. Proper linking of documents, chunks, and embeddings
    4. Efficient resource utilization through async processing
    """
    
    def __init__(self, db_manager: DatabaseManager, config: Dict[str, Any]):
        """
        Initialize the concurrent processing manager.
        
        Args:
            db_manager: Database manager for PostgreSQL operations
            config: System configuration dictionary
        """
        self.db_manager = db_manager
        self.config = config
        self.logger = logging.getLogger("concurrent_processing_manager")
        
        # Processing control
        self.processing_active = False
        self.max_concurrent_tasks = config.get('max_concurrent_processing_tasks', 10)
        self.task_semaphore = threading.Semaphore(self.max_concurrent_tasks)
        
        # Active tasks tracking
        self.active_tasks: Dict[str, ConcurrentProcessingTask] = {}
        self.task_lock = threading.RLock()
        
        # Performance tracking
        self.processing_stats = {
            'documents_processed': 0,
            'chunks_processed': 0,
            'embeddings_generated': 0,
            'processing_times': [],
            'start_time': None
        }
        
        self.logger.info("ConcurrentProcessingManager initialized")
    
    def start_processing(self) -> None:
        """Start concurrent processing"""
        if self.processing_active:
            return
            
        self.processing_active = True
        self.processing_stats['start_time'] = datetime.now(timezone.utc)
        self.logger.info("Concurrent processing started")
    
    def stop_processing(self) -> None:
        """Stop concurrent processing"""
        self.processing_active = False
        self.logger.info("Concurrent processing stopped")
        
        # Wait for active tasks to complete
        with self.task_lock:
            active_task_count = len(self.active_tasks)
            
        if active_task_count > 0:
            self.logger.info(f"Waiting for {active_task_count} active tasks to complete...")
            # Give tasks time to complete gracefully
            time.sleep(2.0)
    
    async def process_document_and_embeddings_concurrently(
        self, 
        document_metadata: DocumentMetadata,
        chunk_metadata_list: List[ChunkMetadata],
        chunk_texts: Dict[str, str]
    ) -> Tuple[bool, List[Tuple[str, str]]]:
        """
        Process document metadata and generate embeddings concurrently.
        
        Args:
            document_metadata: Document metadata to store in PostgreSQL
            chunk_metadata_list: List of chunk metadata to store
            chunk_texts: Dictionary mapping chunk_id to text content
            
        Returns:
            Tuple of (success, list of (chunk_id, embedding_job_id) tuples)
        """
        if not self.processing_active:
            self.logger.warning("Concurrent processing not active, cannot process task")
            return False, []
        
        task_id = str(uuid.uuid4())
        
        try:
            # Create processing task
            task = ConcurrentProcessingTask(
                task_id=task_id,
                document_metadata=document_metadata,
                chunk_metadata_list=chunk_metadata_list,
                chunk_texts=chunk_texts
            )
            
            # Register task
            with self.task_lock:
                self.active_tasks[task_id] = task
            
            self.logger.debug(f"Starting concurrent processing task {task_id} for document {document_metadata.file_name}")
            
            # Start both operations concurrently
            document_task = asyncio.create_task(
                self._process_document_metadata(document_metadata, chunk_metadata_list)
            )
            
            embedding_tasks = asyncio.create_task(
                self._process_embeddings_concurrently(chunk_texts)
            )
            
            # Wait for both operations to complete
            document_success = await document_task
            embedding_job_ids = await embedding_tasks
            
            # Update stats
            with self.task_lock:
                self.processing_stats['documents_processed'] += 1
                self.processing_stats['chunks_processed'] += len(chunk_metadata_list)
                self.processing_stats['embeddings_generated'] += len(embedding_job_ids)
                
                # Remove completed task
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
            
            self.logger.debug(f"Completed concurrent processing task {task_id}: document_success={document_success}, embeddings={len(embedding_job_ids)}")
            
            return document_success, embedding_job_ids
            
        except Exception as e:
            self.logger.error(f"Error in concurrent processing task {task_id}: {e}")
            # Clean up on error
            with self.task_lock:
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
            return False, []
    
    async def _process_document_metadata(
        self, 
        document_metadata: DocumentMetadata,
        chunk_metadata_list: List[ChunkMetadata]
    ) -> bool:
        """
        Process document and chunk metadata in PostgreSQL.
        
        Args:
            document_metadata: Document metadata to store
            chunk_metadata_list: List of chunk metadata to store
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            start_time = time.time()
            
            # Store document metadata
            if not self.db_manager.store_document(document_metadata):
                self.logger.error(f"Failed to store document metadata for {document_metadata.file_name}")
                return False
            
            # Store chunk metadata
            chunks_stored = 0
            for chunk_metadata in chunk_metadata_list:
                if self.db_manager.store_chunk(chunk_metadata):
                    chunks_stored += 1
                else:
                    self.logger.warning(f"Failed to store chunk metadata {chunk_metadata.id}")
            
            processing_time = time.time() - start_time
            
            self.logger.debug(f"Stored document metadata and {chunks_stored}/{len(chunk_metadata_list)} chunks in {processing_time:.2f}s")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing document metadata: {e}")
            return False
    
    async def _process_embeddings_concurrently(
        self, 
        chunk_texts: Dict[str, str]
    ) -> List[Tuple[str, str]]:
        """
        Generate embeddings for chunks concurrently using GPU instances.
        
        Args:
            chunk_texts: Dictionary mapping chunk_id to text content
            
        Returns:
            List of (chunk_id, embedding_job_id) tuples
        """
        try:
            embedding_job_ids = []
            
            # Create embedding jobs for all chunks
            for chunk_id, chunk_text in chunk_texts.items():
                embedding_job = EmbeddingJob(
                    id=str(uuid.uuid4()),
                    chunk_id=chunk_id,
                    text=chunk_text,
                    model=self.config.get('embedding_model') or self.config.get('models', {}).get('embedding_model', 'dengcao/Qwen3-Embedding-0.6B:Q8_0')
                )
                
                # Add job to embedding queue
                embedding_queue.add_job(embedding_job)
                embedding_job_ids.append((chunk_id, embedding_job.id))
                
                self.logger.debug(f"Queued embedding job {embedding_job.id} for chunk {chunk_id}")
            
            self.logger.debug(f"Queued {len(embedding_job_ids)} embedding jobs for concurrent processing")
            
            return embedding_job_ids
            
        except Exception as e:
            self.logger.error(f"Error queuing embedding jobs: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current processing statistics.
        
        Returns:
            Dict with processing statistics
        """
        with self.task_lock:
            stats = self.processing_stats.copy()
            
            # Calculate derived metrics
            if stats['processing_times']:
                stats['avg_processing_time'] = sum(stats['processing_times']) / len(stats['processing_times'])
                stats['min_processing_time'] = min(stats['processing_times'])
                stats['max_processing_time'] = max(stats['processing_times'])
            
            stats['active_tasks'] = len(self.active_tasks)
            stats['max_concurrent_tasks'] = self.max_concurrent_tasks
            
            if stats['start_time']:
                elapsed = datetime.now(timezone.utc) - stats['start_time']
                stats['elapsed_seconds'] = elapsed.total_seconds()
                
                if stats['elapsed_seconds'] > 0:
                    stats['documents_per_second'] = stats['documents_processed'] / stats['elapsed_seconds']
                    stats['embeddings_per_second'] = stats['embeddings_generated'] / stats['elapsed_seconds']
            
            return stats
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.stop_processing()
        self.logger.info("ConcurrentProcessingManager cleaned up")