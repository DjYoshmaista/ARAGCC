"""
High-performance parallel ingestion controller with CPU/GPU workload separation
Handles massive file processing with semaphore-controlled concurrency
"""

import asyncio
import threading
import time
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
import logging
import uuid
import hashlib

# Local imports
from core.ingestion.text_processor import TextChunker
from core.database.manager import DatabaseManager, DocumentMetadata, ChunkMetadata
from core.embedding.queue import embedding_queue, EmbeddingJob, EmbeddingResult

@dataclass
class FileProcessingJob:
    """A job for processing a single file"""
    file_path: str
    file_id: str
    priority: int = 1
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

@dataclass
class ProcessingState:
    """State for the parallel processing system"""
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    total_chunks: int = 0
    processed_chunks: int = 0
    embeddings_generated: int = 0
    start_time: Optional[datetime] = None
    last_save_time: Optional[datetime] = None
    current_batch_id: Optional[str] = None
    current_folder: Optional[str] = None
    processed_file_ids: Set[str] = None
    
    def __post_init__(self):
        if self.processed_file_ids is None:
            self.processed_file_ids = set()

class ParallelIngestionController:
    """
    High-performance parallel ingestion controller with CPU/GPU separation
    
    Architecture:
    - CPU Workers: File I/O, text chunking, metadata generation
    - GPU Workers: Embedding generation (via singleton queue)
    - Database Workers: Batch insertion with connection pooling
    - Semaphore Control: Prevents resource exhaustion
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize parallel ingestion controller with configuration"""
        self.logger = logging.getLogger("parallel_ingestion")
        
        # Configuration
        self.config = config
        
        # Progress tracking (set by orchestrator)
        self.progress_tracker = None
        self.max_cpu_workers = config.get('max_cpu_workers', 8)
        self.max_db_workers = config.get('max_db_workers', 4)
        self.max_concurrent_files = config.get('max_concurrent_files', 50)
        self.chunk_batch_size = config.get('chunk_batch_size', 100)
        
        # State management with thread safety
        self.state = ProcessingState()
        self.state_lock = threading.Lock()  # Protect state from concurrent access
        self.state_file = Path("parallel_ingestion_state.json")
        self.checkpoint_interval = config.get('checkpoint_interval', 30)  # seconds
        
        # Processing queues
        self.file_queue = Queue()
        self.chunk_queue = Queue()
        self.db_batch_queue = Queue()
        
        # Database and text processing
        self.db_manager = None
        self.text_chunker = TextChunker()
        
        # Thread control
        self.workers_running = False
        self.worker_threads = []
        
        # Performance tracking
        self.processing_times = {
            'file_read': [],
            'text_chunk': [],
            'embedding_queue': [],
            'total_per_file': []
        }
        
        self.logger.info("Parallel ingestion controller initialized")
    
    def initialize_database(self, postgresql_config: Dict[str, Any], qdrant_config: Dict[str, Any]):
        """Initialize database connections"""
        self.db_manager = DatabaseManager(postgresql_config, qdrant_config)
        
        # Test connections
        if not self.db_manager.connect_postgresql():
            raise Exception("Failed to connect to PostgreSQL")
        if not self.db_manager.connect_qdrant():
            raise Exception("Failed to connect to Qdrant")
        
        # Initialize schemas
        if not self.db_manager.initialize_schemas():
            raise Exception("Failed to initialize database schemas")
        
        # Create Qdrant collection for embeddings
        vector_size = self.config.get('embedding_dimensions', 384)  # Model produces 384-dim vectors
        collection_name = self.config.get('qdrant_collection', 'document_embeddings')
        if not self.db_manager.create_qdrant_collection(collection_name, vector_size):
            self.logger.warning("Failed to create Qdrant collection (may already exist)")
        
        self.logger.info("Database connections initialized successfully")
    
    def start_workers(self):
        """Start all worker threads for parallel processing"""
        if self.workers_running:
            return
        
        self.workers_running = True
        self.state.start_time = datetime.now(timezone.utc)
        
        # Start embedding queue workers - scale based on GPU capacity
        # Default to 8 workers for better GPU utilization, configurable up to 16
        num_embedding_workers = min(self.config.get('max_embedding_workers', 8), 16)
        self.logger.info(f"Starting {num_embedding_workers} embedding workers for GPU processing")
        
        # Start embedding workers synchronously to avoid event loop conflicts
        try:
            # Create a proper sync wrapper that starts multi-instance system
            def start_workers_sync(num_workers=None):
                """Synchronous wrapper for async start_workers that properly starts instances"""
                if embedding_queue.workers_running:
                    return
                
                if num_workers is not None:
                    embedding_queue.max_workers = num_workers
                
                # Run the async start_workers method properly
                import asyncio
                
                # Create a new event loop if none exists
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If loop is already running, use a separate thread
                        import concurrent.futures
                        import threading
                        
                        def run_in_thread():
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            try:
                                return embedding_queue.start_workers(num_workers)
                            finally:
                                new_loop.close()
                        
                        # Run in separate thread to avoid event loop conflict
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(run_in_thread)
                            future.result(timeout=30)  # 30 second timeout
                    else:
                        # Loop exists but not running
                        embedding_queue.start_workers(num_workers)
                except RuntimeError:
                    # No event loop, create new one
                    embedding_queue.start_workers(num_workers)
                
                self.logger.info(f"Started {embedding_queue.max_workers} embedding workers with multi-instance support")
            
            # Use sync wrapper to avoid async issues
            start_workers_sync(num_embedding_workers)
            
        except Exception as e:
            self.logger.error(f"Failed to start embedding workers: {e}")
            # Fallback: start basic workers without multi-instance support
            self._start_fallback_embedding_workers(num_embedding_workers)
        
        # Start CPU workers for file processing
        for i in range(self.max_cpu_workers):
            worker = threading.Thread(
                target=self._file_processing_worker,
                args=(i,),
                daemon=True,
                name=f"FileWorker-{i}"
            )
            worker.start()
            self.worker_threads.append(worker)
        
        # Start embedding result processor
        result_worker = threading.Thread(
            target=self._embedding_result_worker,
            daemon=True,
            name="EmbeddingResultWorker"
        )
        result_worker.start()
        self.worker_threads.append(result_worker)
        
        # Start checkpoint worker
        checkpoint_worker = threading.Thread(
            target=self._checkpoint_worker,
            daemon=True,
            name="CheckpointWorker"
        )
        checkpoint_worker.start()
        self.worker_threads.append(checkpoint_worker)
    
    def stop_workers(self):
        """Stop all worker threads"""
        self.workers_running = False
        
        # Stop embedding workers
        embedding_queue.stop_workers()
        
        # Add sentinel values to stop workers
        for _ in range(self.max_cpu_workers):
            self.file_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.worker_threads:
            worker.join(timeout=10.0)
        
        self.worker_threads.clear()
        self.logger.info("Stopped all workers")
        
        # Save final state
        self.save_state()
    
    def process_folder(self, folder_path: str, recursive: bool = True, 
                       postgresql_config: Dict[str, Any] = None, 
                       qdrant_config: Dict[str, Any] = None) -> str:
        """
        Process entire folder with high-performance parallel processing
        Returns batch_id for tracking
        """
        batch_id = str(uuid.uuid4())
        self.state.current_batch_id = batch_id
        self.state.current_folder = folder_path  # Track current folder
        
        self.logger.info(f"Starting parallel processing of folder: {folder_path}")
        self.logger.info(f"Batch ID: {batch_id}")
        
        # Initialize database connections
        if postgresql_config is None:
            postgresql_config = self.config.get('databases', {}).get('postgresql', {})
        if qdrant_config is None:
            qdrant_config = self.config.get('databases', {}).get('qdrant', {})
        self.initialize_database(postgresql_config, qdrant_config)
        
        # Load previous state if available for this specific folder
        self.load_state(folder_path)
        
        # Start workers
        self.start_workers()
        
        try:
            # Discover all files
            files = self._discover_files(folder_path, recursive)
            self.state.total_files = len(files)
            
            self.logger.info(f"Discovered {len(files)} files for processing")
            
            # Filter out already processed files
            new_files = [f for f in files if self._get_file_id(f) not in self.state.processed_file_ids]
            
            self.logger.info(f"Found {len(new_files)} new files to process (skipping {len(files) - len(new_files)} already processed)")
            
            if new_files:
                self.logger.info(f"Processing {len(new_files)} new files")
                
                # Queue all files for processing
                for file_path in new_files:
                    job = FileProcessingJob(
                        file_path=file_path,
                        file_id=self._get_file_id(file_path)
                    )
                    self.file_queue.put(job)
                
                # Wait for processing to complete
                self._wait_for_completion()
            else:
                self.logger.info("All files already processed")
            
        finally:
            self.stop_workers()
        
        self.logger.info(f"Parallel processing completed for batch {batch_id}")
        self._log_final_stats()
        
        return batch_id
    
    def _discover_files(self, folder_path: str, recursive: bool) -> List[str]:
        """Discover all processable files in the folder using os.walk for scalability."""
        supported_extensions = {
            '.txt', '.md', '.py', '.js', '.java', '.cpp', '.c', '.h', '.json',
            '.xml', '.html', '.css', '.sql', '.csv', '.tsv', '.xls', '.pdf',
            '.docx', '.doc', '.jsonl'
        }
        
        files = []
        exclude_dirs = ['node_modules', '__pycache__', '.git', 'build', 'dist']

        if recursive:
            for root, dirs, filenames in os.walk(folder_path):
                # Exclude specified directories
                dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
                
                for filename in filenames:
                    if not filename.startswith('.'):
                        file_path = Path(root) / filename
                        if file_path.suffix.lower() in supported_extensions:
                            files.append(str(file_path))
        else:
            for item in os.listdir(folder_path):
                path = Path(folder_path) / item
                if path.is_file() and not item.startswith('.') and path.suffix.lower() in supported_extensions:
                    files.append(str(path))

        # Only sort if we have a reasonable number of files
        if len(files) < 100000:
            return sorted(files)
        else:
            self.logger.info(f"Skipping sorting for large dataset ({len(files)} files)")
            return files
    
    def _get_file_id(self, file_path: str) -> str:
        """Generate consistent file ID for deduplication"""
        return hashlib.md5(file_path.encode()).hexdigest()
    
    def _file_processing_worker(self, worker_id: int):
        """Worker thread for CPU-intensive file processing"""
        self.logger.info(f"File worker {worker_id} started")
        
        # Create a semaphore for this worker thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        semaphore = asyncio.Semaphore(self.max_concurrent_files)
        
        try:
            while self.workers_running:
                try:
                    job = self.file_queue.get(timeout=1.0)
                    if job is None:  # Sentinel to stop
                        break
                    
                    loop.run_until_complete(
                        self._process_single_file(job, worker_id, semaphore)
                    )
                    
                    self.file_queue.task_done()
                    
                except Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"File worker {worker_id} error: {e}")
                    with self.state_lock:
                        self.state.failed_files += 1
                    
        finally:
            loop.close()
            asyncio.set_event_loop(None)
        
        self.logger.info(f"File worker {worker_id} stopped")
    
    async def _process_single_file(self, job: FileProcessingJob, worker_id: int, semaphore: asyncio.Semaphore):
        """Process a single file with semaphore control"""
        async with semaphore:
            start_time = time.time()
            
            try:
                # Import file format handlers
                from file_format_handlers import get_file_handler
                
                file_path_obj = Path(job.file_path)
                file_extension = file_path_obj.suffix.lower()[1:]  # Remove the dot
                file_handler = get_file_handler(file_extension)
                
                # If we have a specialized handler for this file type, use it
                if file_handler and file_extension in ['csv', 'tsv', 'xls', 'pdf', 'docx', 'json', 'jsonl']:
                    # For specialized file formats, extract text content
                    content = self._extract_text_from_file(file_path_obj, file_handler)
                else:
                    # Read file content for regular text files
                    file_start = time.time()
                    with open(job.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    file_read_time = time.time() - file_start
                
                if not content.strip():
                    self.state.skipped_files += 1
                    return
                
                # Generate document metadata
                file_stat = os.stat(job.file_path)
                document = DocumentMetadata(
                    id=job.file_id,
                    file_path=job.file_path,
                    file_name=os.path.basename(job.file_path),
                    folder_path=os.path.dirname(job.file_path),
                    file_size=file_stat.st_size,
                    created_at=datetime.fromtimestamp(file_stat.st_ctime),
                    updated_at=datetime.fromtimestamp(file_stat.st_mtime),
                    total_chunks=0  # Will be updated after chunking
                )
                
                # Chunk the text
                chunk_start = time.time()
                # Create a new TextChunker with the specified parameters
                chunker = TextChunker(
                    chunk_size=self.config.get('chunk_size', 1000),
                    overlap_size=self.config.get('chunk_overlap', 200)
                )
                chunks = chunker.chunk_text(text=content)
                chunk_time = time.time() - chunk_start
                
                document.total_chunks = len(chunks)
                
                # Store document metadata - check if db_manager is initialized
                if not self.db_manager:
                    raise Exception("Database manager not initialized")
                if not self.db_manager.store_document(document):
                    raise Exception("Failed to store document metadata")
                
                # Track PostgreSQL activity
                if self.progress_tracker:
                    self.logger.debug(f"Recording document_stored activity for {os.path.basename(job.file_path)}")
                    self.progress_tracker.record_postgres_activity("document_stored", {
                        "filename": os.path.basename(job.file_path),
                        "file_size": document.file_size,
                        "path": job.file_path
                    })
                else:
                    self.logger.debug("No progress tracker available for document_stored activity")
                
                # Process chunks
                chunk_jobs = []
                for i, chunk_tuple in enumerate(chunks):
                    chunk_text, token_count, overlap_tokens = chunk_tuple
                    chunk_id = f"{job.file_id}_chunk_{i}"
                    
                    # Create chunk metadata
                    chunk_metadata = ChunkMetadata(
                        id=chunk_id,
                        document_id=job.file_id,
                        chunk_number=i,
                        chunk_text=chunk_text,
                        token_count=token_count,
                        overlap_tokens=overlap_tokens,
                        previous_chunk_id=f"{job.file_id}_chunk_{i-1}" if i > 0 else None,
                        next_chunk_id=f"{job.file_id}_chunk_{i+1}" if i < len(chunks) - 1 else None,
                        created_at=datetime.now(timezone.utc)
                    )
                    
                    # Store chunk metadata
                    if not self.db_manager.store_chunk(chunk_metadata):
                        raise Exception(f"Failed to store chunk {chunk_id}")
                    
                    # Track chunk creation activity
                    if self.progress_tracker:
                        self.logger.debug(f"Recording chunk_created activity for chunk {chunk_id}")
                        self.progress_tracker.record_postgres_activity("chunk_created", {
                            "chunk_id": chunk_id,
                            "document_id": job.file_id,
                            "count": 1
                        })
                    else:
                        self.logger.debug("No progress tracker available for chunk_created activity")
                    
                    # Queue embedding job
                    embedding_job = EmbeddingJob(
                        id=str(uuid.uuid4()),
                        chunk_id=chunk_id,
                        text=chunk_text,
                        model=self.config.get('embedding_model') or self.config.get('models', {}).get('embedding_model', 'granite-embedding:latest')
                    )
                    embedding_queue.add_job(embedding_job)
                    
                    chunk_jobs.append((chunk_id, embedding_job.id))
                
                # Update processing stats
                total_time = time.time() - start_time
                self.processing_times['file_read'].append(file_read_time if 'file_read_time' in locals() else 0)
                self.processing_times['text_chunk'].append(chunk_time)
                self.processing_times['total_per_file'].append(total_time)
                
                # Keep only last 1000 times for memory efficiency
                for key in self.processing_times:
                    if len(self.processing_times[key]) > 1000:
                        self.processing_times[key] = self.processing_times[key][-1000:]
                
                # Thread-safe state updates
                with self.state_lock:
                    self.state.processed_files += 1
                    self.state.total_chunks += len(chunks)
                    self.state.processed_file_ids.add(job.file_id)
                
                # Track file completion
                if self.progress_tracker:
                    self.logger.debug(f"Recording file_processed activity for {os.path.basename(job.file_path)}")
                    self.progress_tracker.record_file_activity("file_processed", {
                        "filename": os.path.basename(job.file_path),
                        "path": job.file_path,
                        "chunks_created": len(chunks),
                        "processing_time": total_time
                    })
                else:
                    self.logger.debug("No progress tracker available for file_processed activity")
                
                self.logger.debug(f"Worker {worker_id} processed file: {job.file_path} ({len(chunks)} chunks, {total_time:.2f}s)")
                
            except Exception as e:
                self.logger.error(f"Failed to process file {job.file_path}: {e}")
                # Note: failed_files counter is already incremented by the worker exception handler

    def _extract_text_from_file(self, file_path: Path, file_handler) -> str:
        """Extract text content from specialized file formats"""
        try:
            # For now, we'll just concatenate all the chunk contents
            # In a more sophisticated implementation, we might want to format this better
            content_parts = []
            
            # Process the file in chunks
            for chunk_batch in file_handler(str(file_path), chunk_size=10):
                for chunk in chunk_batch:
                    if 'content' in chunk:
                        content_parts.append(chunk['content'])
            
            return '\n'.join(content_parts)
        except Exception as e:
            self.logger.error(f"Error extracting text from {file_path}: {e}")
            return f"Error processing file {file_path}: {str(e)}"
    
    def _embedding_result_worker(self):
        """Worker to process embedding results and store vectors using batch operations"""
        self.logger.info("Embedding result worker started with batch optimization")
        
        collection_name = self.config.get('qdrant_collection', 'document_embeddings')
        batch_size = self.config.get('embedding_batch_size', 128)  # Larger batches for efficiency
        vector_batch = []
        last_batch_time = time.time()
        batch_timeout = 5.0  # seconds - force batch processing after timeout
        
        while self.workers_running:
            try:
                # Get batch of results for efficiency
                results = embedding_queue.get_results_batch(max_count=batch_size, timeout=1.0)
                
                if not results and not vector_batch:
                    time.sleep(0.1)
                    continue
                
                # Process results and build vector batch
                successful_results = []
                for result in results:
                    if result.success and result.embedding:
                        vector_batch.append((result.chunk_id, result.embedding))
                        successful_results.append(result)
                    elif not result.success:
                        self.logger.error(f"Embedding failed for chunk {result.chunk_id}: {result.error}")
                
                # Check if we should flush the batch
                should_flush = (
                    len(vector_batch) >= batch_size or
                    (time.time() - last_batch_time) > batch_timeout or
                    not self.workers_running
                )
                
                if should_flush and vector_batch:
                    # Batch store vectors in Qdrant
                    stored_count = self.db_manager.store_chunk_vectors_batch(
                        vector_batch, collection_name
                    )
                    
                    if stored_count > 0:
                        with self.state_lock:
                            self.state.embeddings_generated += stored_count
                        
                        # Track vector database activity in batch
                        if self.progress_tracker:
                            total_dimensions = sum(len(vec) for _, vec in vector_batch[:stored_count])
                            avg_dimensions = total_dimensions / stored_count if stored_count > 0 else 0
                            
                            self.progress_tracker.record_vector_activity("embedding_stored_batch", {
                                "chunk_count": stored_count,
                                "avg_embedding_dimensions": avg_dimensions,
                                "batch_size": len(vector_batch),
                                "storage_efficiency": stored_count / len(vector_batch) if vector_batch else 0
                            })
                        
                        self.logger.debug(f"Batch stored {stored_count}/{len(vector_batch)} vectors in Qdrant")
                    else:
                        self.logger.error(f"Failed to store batch of {len(vector_batch)} vectors")
                    
                    # Clear batch
                    vector_batch.clear()
                    last_batch_time = time.time()
                
                # If we processed results but didn't flush, just log progress
                if results and not should_flush:
                    self.logger.debug(f"Accumulated {len(vector_batch)} vectors for batch storage")
                
            except Exception as e:
                self.logger.error(f"Embedding result worker error: {e}")
                # Clear problematic batch on error
                if vector_batch:
                    self.logger.warning(f"Clearing batch of {len(vector_batch)} vectors due to error")
                    vector_batch.clear()
        
        # Final batch flush on shutdown
        if vector_batch:
            self.logger.info(f"Final batch flush: storing {len(vector_batch)} vectors")
            stored_count = self.db_manager.store_chunk_vectors_batch(vector_batch, collection_name)
            if stored_count > 0:
                with self.state_lock:
                    self.state.embeddings_generated += stored_count
        
        self.logger.info("Embedding result worker stopped")
    
    def _database_worker(self, worker_id: int):
        """Worker for batch database operations"""
        self.logger.info(f"Database worker {worker_id} started")
        
        while self.workers_running:
            try:
                # For now, this is handled by individual workers
                # Future: Implement batch operations here
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Database worker {worker_id} error: {e}")
        
        self.logger.info(f"Database worker {worker_id} stopped")
    
    def _checkpoint_worker(self):
        """Worker to periodically save state"""
        self.logger.info("Checkpoint worker started")
        
        while self.workers_running:
            try:
                time.sleep(self.checkpoint_interval)
                if self._should_save_state():
                    self.save_state()
                    
            except Exception as e:
                self.logger.error(f"Checkpoint worker error: {e}")
        
        self.logger.info("Checkpoint worker stopped")
    
    def _wait_for_completion(self):
        """Wait for all processing to complete with timeout"""
        self.logger.info("Waiting for processing to complete...")
        
        timeout = self.config.get('file_processing_timeout', 1800)  # 30 minute timeout for file processing
        start_time = time.time()
        
        # Wait for file queue to empty
        while not self.file_queue.empty():
            # Check for timeout
            if time.time() - start_time > timeout:
                self.logger.warning("File processing timeout reached")
                break
            time.sleep(1.0)
            self._log_progress()
            
            # Also check completion status
            with self.state_lock:
                files_completed = self.state.processed_files + self.state.failed_files + self.state.skipped_files
                if files_completed >= self.state.total_files:
                    break
        
        self.logger.info("File processing completed, waiting for embeddings...")
        
        # Wait for embeddings to complete - increased timeout for large datasets
        embedding_timeout = self.config.get('embedding_timeout', 1800)  # 30 minutes default timeout for embeddings
        embedding_start = time.time()
        
        while True:
            with self.state_lock:
                embeddings_generated = self.state.embeddings_generated
                total_chunks = self.state.total_chunks
                
            # Check completion
            if embeddings_generated >= total_chunks:
                break
                
            if time.time() - embedding_start > embedding_timeout:
                self.logger.warning("Embedding processing timeout reached")
                break
            
            # Check if embedding queue is stuck (no progress for 10 seconds)
            if hasattr(self, '_last_embedding_count'):
                if embeddings_generated == self._last_embedding_count:
                    if time.time() - self._last_embedding_check > 10:
                        self.logger.warning("Embedding queue appears stuck, proceeding...")
                        break
                else:
                    self._last_embedding_check = time.time()
            else:
                self._last_embedding_check = time.time()
            
            self._last_embedding_count = embeddings_generated
            
            time.sleep(1.0)
            self._log_progress()
        
        self.logger.info("All processing completed")
    
    def _log_progress(self):
        """Log current processing progress"""
        with self.state_lock:
            # Make a snapshot of the current state to avoid race conditions
            total_files = self.state.total_files
            processed_files = self.state.processed_files
            failed_files = self.state.failed_files
            skipped_files = self.state.skipped_files
            embeddings_generated = self.state.embeddings_generated
            total_chunks = self.state.total_chunks
        
        if total_files > 0:
            file_percent = (processed_files + failed_files + skipped_files) / total_files * 100
            embedding_percent = embeddings_generated / max(total_chunks, 1) * 100
            
            embedding_stats = embedding_queue.get_stats()
            
            self.logger.info(f"Progress - Files: {processed_files}/{total_files} ({file_percent:.1f}%), "
                           f"Embeddings: {embeddings_generated}/{total_chunks} ({embedding_percent:.1f}%), "
                           f"Queue: {embedding_stats.get('queue_size', 0)} pending")
    
    def _log_final_stats(self):
        """Log final processing statistics"""
        with self.state_lock:
            # Make a snapshot for logging
            start_time = self.state.start_time
            processed_files = self.state.processed_files
            failed_files = self.state.failed_files
            skipped_files = self.state.skipped_files
            total_chunks = self.state.total_chunks
            embeddings_generated = self.state.embeddings_generated
        
        if start_time:
            elapsed = datetime.now(timezone.utc) - start_time
            elapsed_seconds = elapsed.total_seconds()
            
            files_per_second = processed_files / max(elapsed_seconds, 1)
            
            self.logger.info(f"Final Stats - Processed: {processed_files} files, "
                           f"Failed: {failed_files}, Skipped: {skipped_files}")
            self.logger.info(f"Total chunks: {total_chunks}, Embeddings: {embeddings_generated}")
            self.logger.info(f"Processing time: {elapsed_seconds:.1f}s ({files_per_second:.2f} files/sec)")
            
            if self.processing_times['total_per_file']:
                avg_time = sum(self.processing_times['total_per_file']) / len(self.processing_times['total_per_file'])
                self.logger.info(f"Average time per file: {avg_time:.2f}s")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        embedding_stats = embedding_queue.get_stats()
        
        stats = {
            'batch_id': self.state.current_batch_id,
            'total_files': self.state.total_files,
            'processed_files': self.state.processed_files,
            'failed_files': self.state.failed_files,
            'skipped_files': self.state.skipped_files,
            'total_chunks': self.state.total_chunks,
            'embeddings_generated': self.state.embeddings_generated,
            'workers_running': self.workers_running,
            'embedding_queue_stats': embedding_stats
        }
        
        if self.state.start_time:
            elapsed = datetime.now(timezone.utc) - self.state.start_time
            stats['elapsed_seconds'] = elapsed.total_seconds()
            stats['processing_rate'] = self.state.processed_files / max(elapsed.total_seconds(), 1)
        
        return stats
    
    def save_state(self):
        """Save current processing state"""
        try:
            state_data = {
                'total_files': self.state.total_files,
                'processed_files': self.state.processed_files,
                'failed_files': self.state.failed_files,
                'skipped_files': self.state.skipped_files,
                'total_chunks': self.state.total_chunks,
                'embeddings_generated': self.state.embeddings_generated,
                'current_batch_id': self.state.current_batch_id,
                'current_folder': self.state.current_folder,
                'processed_file_ids': list(self.state.processed_file_ids),
                'start_time': self.state.start_time.isoformat() if self.state.start_time else None,
                'last_save_time': datetime.now(timezone.utc).isoformat(),
                'checkpoint_interval': self.checkpoint_interval,
                'config_snapshot': self.config  # Save config for resume validation
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            self.state.last_save_time = datetime.now(timezone.utc)
            self.logger.debug("Processing state saved")
            
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def load_state(self, folder_path: str = None):
        """Load previous processing state"""
        try:
            if folder_path:
                file_path = Path(folder_path) / self.state_file
            else:
                file_path = self.state_file
            if file_path.exists():
                with open(file_path, 'r') as f:
                    state_data = json.load(f)
                
                self.state.total_files = state_data.get('total_files', 0)
                self.state.processed_files = state_data.get('processed_files', 0)
                self.state.failed_files = state_data.get('failed_files', 0)
                self.state.skipped_files = state_data.get('skipped_files', 0)
                self.state.total_chunks = state_data.get('total_chunks', 0)
                self.state.embeddings_generated = state_data.get('embeddings_generated', 0)
                self.state.current_batch_id = state_data.get('current_batch_id')
                self.state.current_folder = state_data.get('current_folder')
                
                processed_ids = state_data.get('processed_file_ids', [])
                self.state.processed_file_ids = set(processed_ids)
                
                # Validate config compatibility for resume
                saved_config = state_data.get('config_snapshot', {})
                if saved_config and folder_path:
                    incompatible_settings = []
                    
                    # Check critical settings
                    for key in ['chunk_size', 'overlap_size', 'embedding_model']:
                        if (key in saved_config and key in self.config and 
                            saved_config[key] != self.config[key]):
                            incompatible_settings.append(f"{key}: was {saved_config[key]}, now {self.config[key]}")
                    
                    if incompatible_settings:
                        self.logger.warning(f"Config changes detected that may affect resume compatibility: {'; '.join(incompatible_settings)}")
                        self.logger.info("Consider clearing state if results are inconsistent")
                
                if state_data.get('start_time'):
                    try:
                        self.state.start_time = datetime.fromisoformat(state_data['start_time'])
                    except:
                        pass
                
                self.logger.info(f"Loaded previous state: {self.state.processed_files} files processed")
                
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
    
    def _should_save_state(self) -> bool:
        """Determine if state should be saved"""
        if self.state.last_save_time is None:
            return True
        
        time_since_save = datetime.now(timezone.utc) - self.state.last_save_time
        return time_since_save.total_seconds() > self.checkpoint_interval
    
    def clear_state(self):
        """Clear all processing state"""
        self.state = ProcessingState()
        embedding_queue.clear_state()
        
        if self.state_file.exists():
            self.state_file.unlink()
        
        self.logger.info("All processing state cleared")
    
    def _start_fallback_embedding_workers(self, num_workers: int):
        """Fallback method to start basic embedding workers"""
        self.logger.info(f"Starting {num_workers} fallback embedding workers")
        
        try:
            from embedding_queue import embedding_queue
            
            # Simple synchronous worker startup
            if not embedding_queue.workers_running:
                embedding_queue.workers_running = True
                embedding_queue.max_workers = num_workers
                
                # Create basic worker threads
                for i in range(num_workers):
                    worker = threading.Thread(
                        target=self._basic_embedding_worker,
                        args=(i,),
                        daemon=True,
                        name=f"FallbackEmbeddingWorker-{i}"
                    )
                    worker.start()
                    embedding_queue.worker_threads.append(worker)
                
                self.logger.info(f"Started {num_workers} fallback embedding workers")
            
        except Exception as e:
            self.logger.error(f"Failed to start fallback workers: {e}")
    
    def _basic_embedding_worker(self, worker_id: int):
        """Basic embedding worker without multi-instance support"""
        from embedding_queue import embedding_queue
        import time
        from queue import Empty
        
        self.logger.info(f"Fallback embedding worker {worker_id} started")
        
        while embedding_queue.workers_running:
            try:
                job = embedding_queue.input_queue.get(timeout=1.0)
                if job is None:  # Sentinel to stop
                    break
                
                start_time = time.time()
                
                # Use model gateway directly for embedding generation
                try:
                    import httpx
                    import asyncio
                    
                    async def generate_embedding():
                        async with httpx.AsyncClient() as client:
                            response = await client.post(
                                "http://localhost:8070/embed",
                                json={
                                    "model": job.model,
                                    "prompt": job.text
                                },
                                timeout=60.0
                            )
                            response.raise_for_status()
                            data = response.json()
                            return data.get("embedding", [])
                    
                    # Run async function in thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    embedding = loop.run_until_complete(generate_embedding())
                    loop.close()
                    
                    processing_time = time.time() - start_time
                    
                    # Create result
                    from embedding_queue import EmbeddingResult
                    result = EmbeddingResult(
                        job_id=job.id,
                        chunk_id=job.chunk_id,
                        embedding=embedding,
                        model=job.model,
                        processing_time=processing_time,
                        success=bool(embedding)
                    )
                    
                    embedding_queue.result_queue.put(result)
                    embedding_queue.processing_stats['total_processed'] += 1
                    
                except Exception as e:
                    # Create error result
                    from embedding_queue import EmbeddingResult
                    result = EmbeddingResult(
                        job_id=job.id,
                        chunk_id=job.chunk_id,
                        embedding=[],
                        model=job.model,
                        processing_time=0.0,
                        success=False,
                        error=str(e)
                    )
                    embedding_queue.result_queue.put(result)
                    embedding_queue.processing_stats['total_failed'] += 1
                
                embedding_queue.input_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Fallback worker {worker_id} error: {e}")
        
        self.logger.info(f"Fallback embedding worker {worker_id} stopped")
