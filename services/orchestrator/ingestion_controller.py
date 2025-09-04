"""
Recursive file folder ingestion controller for AgenticRAG system
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import time
from datetime import datetime
import uuid

# Add the current directory to Python path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Use absolute imports instead of relative imports
from database_utils import DatabaseManager, DocumentMetadata, ChunkMetadata
from text_chunker import TextChunker
import httpx

class IngestionController:
    """Controls the recursive ingestion of files and folders with progress tracking"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chunk_size = config.get('chunk_size', 5000)
        self.overlap_size = config.get('overlap_size', 32)
        self.max_concurrent_files = config.get('max_concurrent_files', 5)
        self.max_concurrent_chunks = config.get('max_concurrent_chunks', 10)
        self.file_semaphore = asyncio.Semaphore(self.max_concurrent_files)
        self.chunk_semaphore = asyncio.Semaphore(self.max_concurrent_chunks)
        
        # Database manager
        self.db_manager = DatabaseManager(
            postgresql_config=config['postgresql'],
            qdrant_config=config['qdrant']
        )
        
        # Text chunker
        self.text_chunker = TextChunker(
            chunk_size=self.chunk_size,
            overlap_size=self.overlap_size
        )
        
        # HTTP client for model gateway
        self.http_client = httpx.AsyncClient(timeout=300.0)  # 5 minutes for large tasks
        self.model_gateway_url = config['model_gateway_url']
        
        # Progress tracking
        self.files_processed = 0
        self.chunks_processed = 0
        self.total_files = 0
        self.total_chunks = 0
        self.file_progress_bar = None
        self.chunk_progress_bar = None
        self.progress_lock = asyncio.Lock()
        
        # Statistics
        self.processing_stats = {
            'files_ingested': 0,
            'chunks_created': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }

    async def initialize(self):
        """Initialize the ingestion controller"""
        # Initialize database schemas
        if not self.db_manager.initialize_schemas():
            raise Exception("Failed to initialize database schemas")
        
        # Create Qdrant collection
        if not self.db_manager.create_qdrant_collection(
            collection_name=self.config['qdrant']['collection_name'],
            vector_size=self.config.get('embedding_dimensions', 640)
        ):
            raise Exception("Failed to create Qdrant collection")
        
        # Connect to databases
        if not self.db_manager.connect_postgresql():
            raise Exception("Failed to connect to PostgreSQL")
        
        if not self.db_manager.connect_qdrant():
            raise Exception("Failed to connect to Qdrant")

    async def count_files(self, paths: List[str], recursive: bool = True) -> int:
        """Count total number of files to process"""
        total_files = 0
        
        for path_str in paths:
            path = Path(path_str)
            if path.is_file():
                if self._is_text_file(path):
                    total_files += 1
            elif path.is_dir():
                if recursive:
                    for file_path in path.rglob('*'):
                        if file_path.is_file() and self._is_text_file(file_path):
                            total_files += 1
                else:
                    for file_path in path.glob('*'):
                        if file_path.is_file() and self._is_text_file(file_path):
                            total_files += 1
        
        return total_files

    def _is_text_file(self, file_path: Path) -> bool:
        """Check if file is a text file based on extension"""
        text_extensions = {
            '.txt', '.md', '.markdown', '.rst', '.py', '.js', '.ts', '.java', 
            '.cpp', '.c', '.h', '.hpp', '.cs', '.rb', '.go', '.rs', '.php',
            '.xml', '.json', '.yaml', '.yml', '.csv', '.log', '.html', '.css',
            '.sh', '.bat', '.sql', '.tex', '.bib', '.cfg', '.conf', '.ini',
            '.toml', '.proto', '.graphql', '.gql', '.vue', '.jsx', '.tsx',
            '.tsv', '.xls', '.pdf', '.docx', '.doc', '.jsonl'
        }
        return file_path.suffix.lower() in text_extensions

    async def _update_progress_bars(self):
        """Update both progress bars"""
        async with self.progress_lock:
            if self.file_progress_bar:
                self.file_progress_bar.n = self.files_processed
                self.file_progress_bar.refresh()
            
            if self.chunk_progress_bar:
                self.chunk_progress_bar.n = self.chunks_processed
                self.chunk_progress_bar.refresh()

    async def _progress_updater(self):
        """Continuously update progress bars every 100ms"""
        while not self.processing_stats.get('completed', False):
            await self._update_progress_bars()
            await asyncio.sleep(0.1)  # Update every 100ms

    async def generate_embedding(self, text: str, model: str = "dengcao/Qwen3-Embedding-0.6B:Q8_0") -> List[float]:
        """Generate embedding for text using model gateway"""
        try:
            response = await self.http_client.post(
                f"{self.model_gateway_url}/embed",
                json={
                    "model": model,
                    "prompt": text
                },
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
            return data.get("embedding", [])
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    async def process_chunk(self, chunk_text: str, chunk_metadata: ChunkMetadata, 
                          document_id: str, model: str) -> bool:
        """Process a single chunk: generate embedding and store in databases"""
        async with self.chunk_semaphore:
            try:
                # Generate embedding
                embedding = await self.generate_embedding(chunk_text, model)
                if not embedding:
                    raise Exception("Failed to generate embedding")
                
                # Store chunk metadata in PostgreSQL
                if not self.db_manager.store_chunk(chunk_metadata):
                    raise Exception("Failed to store chunk metadata")
                
                # Store embedding in Qdrant
                if not self.db_manager.store_chunk_vector(
                    chunk_id=chunk_metadata.id,
                    vector=embedding,
                    collection_name=self.config['qdrant']['collection_name']
                ):
                    raise Exception("Failed to store chunk vector")
                
                # Update progress
                async with self.progress_lock:
                    self.chunks_processed += 1
                    self.processing_stats['chunks_created'] += 1
                
                return True
            except Exception as e:
                print(f"Error processing chunk {chunk_metadata.id}: {e}")
                async with self.progress_lock:
                    self.processing_stats['errors'] += 1
                return False

    async def process_file(self, file_path: Path, model: str) -> bool:
        """Process a single file: chunk, generate embeddings, and store"""
        async with self.file_semaphore:
            try:
                # Import file format handlers
                from file_format_handlers import get_file_handler
                
                file_extension = file_path.suffix.lower()[1:]  # Remove the dot
                file_handler = get_file_handler(file_extension)
                
                # If we have a specialized handler for this file type, use it
                if file_handler and file_extension in ['csv', 'tsv', 'xls', 'pdf', 'docx', 'json', 'jsonl']:
                    # For specialized file formats, we'll process them differently
                    # This is a simplified approach - in a full implementation, we would
                    # convert these files to text and then chunk them like regular text files
                    content = self._extract_text_from_file(file_path, file_handler)
                else:
                    # Read file content for text files
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                
                if not content.strip():
                    print(f"Skipping empty file: {file_path}")
                    return True
                
                # Create document metadata
                document_id = str(uuid.uuid4())
                document_metadata = DocumentMetadata(
                    id=document_id,
                    file_path=str(file_path.absolute()),
                    file_name=file_path.name,
                    folder_path=str(file_path.parent.absolute()),
                    file_size=file_path.stat().st_size,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    total_chunks=0
                )
                
                # Store document metadata
                if not self.db_manager.store_document(document_metadata):
                    raise Exception("Failed to store document metadata")
                
                # Chunk the text
                chunks = self.text_chunker.chunk_text_by_sentences(content)
                document_metadata.total_chunks = len(chunks)
                
                # Update document with total chunks
                if not self.db_manager.store_document(document_metadata):
                    print(f"Warning: Could not update document {document_id} with chunk count")
                
                # Process chunks
                chunk_tasks = []
                previous_chunk_id = None
                
                for i, (chunk_text, token_count, overlap_tokens) in enumerate(chunks):
                    chunk_id = str(uuid.uuid4())
                    
                    # Create next chunk ID for linking (except for last chunk)
                    next_chunk_id = str(uuid.uuid4()) if i < len(chunks) - 1 else None
                    
                    # Create chunk metadata
                    chunk_metadata = ChunkMetadata(
                        id=chunk_id,
                        document_id=document_id,
                        chunk_number=i,
                        chunk_text=chunk_text,
                        token_count=token_count,
                        overlap_tokens=overlap_tokens,
                        previous_chunk_id=previous_chunk_id,
                        next_chunk_id=next_chunk_id,
                        created_at=datetime.utcnow()
                    )
                    
                    # Process chunk
                    task = asyncio.create_task(
                        self.process_chunk(chunk_text, chunk_metadata, document_id, model)
                    )
                    chunk_tasks.append(task)
                    
                    # Update previous chunk ID for next iteration
                    previous_chunk_id = chunk_id
                
                # Wait for all chunk processing to complete
                chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
                
                # Check if all chunks were processed successfully
                success_count = sum(1 for result in chunk_results if result is True)
                if success_count != len(chunks):
                    print(f"Warning: Only {success_count}/{len(chunks)} chunks processed successfully for {file_path}")
                
                # Update progress
                async with self.progress_lock:
                    self.files_processed += 1
                    self.processing_stats['files_ingested'] += 1
                
                return True
                
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                async with self.progress_lock:
                    self.processing_stats['errors'] += 1
                return False

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
            print(f"Error extracting text from {file_path}: {e}")
            return f"Error processing file {file_path}: {str(e)}"

    async def ingest_paths(self, paths: List[str], recursive: bool = True, 
                          model: str = "dengcao/Qwen3-Embedding-0.6B:Q8_0") -> Dict[str, Any]:
        """Ingest files from paths with progress tracking"""
        # Count total files
        self.total_files = await self.count_files(paths, recursive)
        if self.total_files == 0:
            print("No files found to process")
            return self.processing_stats
        
        # Initialize progress tracking
        self.processing_stats['total_files'] = self.total_files
        self.processing_stats['processed_files'] = 0
        self.processing_stats['total_chunks'] = 0
        self.processing_stats['processed_chunks'] = 0
        
        # Initialize progress bars
        self.file_progress_bar = tqdm(
            total=self.total_files,
            desc="Files Processed",
            unit="file",
            position=0,
            leave=True
        )
        
        self.chunk_progress_bar = tqdm(
            total=0,  # Will be updated as chunks are discovered
            desc="Chunks Processed",
            unit="chunk",
            position=1,
            leave=True
        )
        
        # Start progress updater
        progress_task = asyncio.create_task(self._progress_updater())
        
        try:
            self.processing_stats['start_time'] = datetime.utcnow()
            
            # Collect all files to process
            files_to_process = []
            for path_str in paths:
                path = Path(path_str)
                if path.is_file():
                    if self._is_text_file(path):
                        files_to_process.append(path)
                elif path.is_dir():
                    if recursive:
                        for file_path in path.rglob('*'):
                            if file_path.is_file() and self._is_text_file(file_path):
                                files_to_process.append(file_path)
                    else:
                        for file_path in path.glob('*'):
                            if file_path.is_file() and self._is_text_file(file_path):
                                files_to_process.append(file_path)
            
            # Process files in parallel
            file_tasks = [
                asyncio.create_task(self.process_file(file_path, model))
                for file_path in files_to_process
            ]
            
            # Wait for all files to be processed
            file_results = await asyncio.gather(*file_tasks, return_exceptions=True)
            
            # Update final progress
            self.processing_stats['end_time'] = datetime.utcnow()
            
            # Mark processing as completed
            self.processing_stats['completed'] = True
            
            # Update progress bars one final time
            await self._update_progress_bars()
            
            # Close progress bars
            if self.file_progress_bar:
                self.file_progress_bar.close()
            if self.chunk_progress_bar:
                self.chunk_progress_bar.close()
            
            return self.processing_stats
            
        except Exception as e:
            print(f"Error during ingestion: {e}")
            self.processing_stats['errors'] += 1
            return self.processing_stats
        finally:
            # Cancel progress updater
            if not progress_task.done():
                progress_task.cancel()
            
            # Close HTTP client
            await self.http_client.aclose()
            
            # Close database connections
            self.db_manager.close_connections()

    def get_progress_info(self) -> Dict[str, Any]:
        """Get current progress information"""
        return {
            'files_processed': self.files_processed,
            'chunks_processed': self.chunks_processed,
            'total_files': self.total_files,
            'total_chunks': self.total_chunks,
            'processing_stats': self.processing_stats.copy()
        }