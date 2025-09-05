"""
Refactored Orchestrator Service using new modular architecture.
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks, Body
from pydantic import BaseModel
import uvicorn
import logging
import yaml
import os
import sys

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import from new modular structure
from core.database import DatabaseManager, DocumentMetadata, ChunkMetadata
from core.embedding import EmbeddingGenerator, EmbeddingQueueManager, EmbeddingJob, EmbeddingResult
from core.monitoring import ProgressTracker, ProgressUpdate, get_progress_tracker, cleanup_progress_tracker
from shared.config import get_config
from shared.utils.rate_limiter import rate_limiter, RateLimit

logger = logging.getLogger("orchestrator_service")

# Task management (simplified version)
class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"

class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class TaskCreateRequest(BaseModel):
    task_type: str
    description: str
    parameters: Optional[Dict[str, Any]] = {}
    dependencies: Optional[List[str]] = []
    priority: Optional[str] = "normal"

@dataclass
class Task:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[str] = None
    error: Optional[str] = None

class Orchestrator:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.task_queue: List[Task] = []
        self.running = False
        
        # Load configuration
        self.config = get_config()
        self.model_gateway_url = f"http://{self.config['services']['model_gateway']['host']}:{self.config['services']['model_gateway']['port']}"
        
        # Initialize components
        self.database_manager = DatabaseManager(
            self.config['databases']['postgresql'],
            self.config['databases']['qdrant']
        )
        
        self.embedding_generator = EmbeddingGenerator(self.model_gateway_url)
        
        logger.info("Orchestrator initialized with modular architecture")
        
    async def start(self):
        """Start the orchestrator"""
        self.running = True
        asyncio.create_task(self.process_task_queue())
        logger.info("Orchestrator started")
    
    async def stop(self):
        """Stop the orchestrator"""
        self.running = False
        await self.embedding_generator.close()
        self.database_manager.disconnect()
        logger.info("Orchestrator stopped")
    
    async def process_task_queue(self):
        """Main task processing loop"""
        while self.running:
            if self.task_queue:
                self.task_queue.sort(key=lambda t: (t.priority.value, t.created_at), reverse=True)
                task = self.task_queue.pop(0)
                await self.execute_task(task)
            await asyncio.sleep(1)
    
    def add_task(self, task: Task) -> str:
        """Add a new task to the system"""
        self.tasks[task.id] = task
        self.task_queue.append(task)
        return task.id
    
    async def execute_task(self, task: Task):
        """Execute a single task"""
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now(timezone.utc)
            logger.info(f"Executing task {task.id}: {task.description}")
            
            if task.type == "llm_inference":
                result = await self.handle_llm_task(task)
            elif task.type == "rag_ingest":
                result = await self.handle_rag_ingest_task(task)
            elif task.type == "folder_ingestion":
                result = await self.handle_folder_ingestion_task(task)
            else:
                result = f"Completed task: {task.description}"
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(timezone.utc)
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now(timezone.utc)
            logger.error(f"Task {task.id} failed: {e}")
    
    async def handle_llm_task(self, task: Task) -> str:
        """Handle LLM inference tasks"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.model_gateway_url}/generate",
                    json={
                        "model": task.parameters.get("model", "qwen3:8b"),
                        "prompt": task.parameters.get("prompt", ""),
                        "stream": False
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
        except Exception as e:
            raise Exception(f"LLM task failed: {e}")
    
    async def handle_rag_ingest_task(self, task: Task) -> str:
        """Handle RAG document ingestion"""
        try:
            document_id = task.parameters.get("document_id")
            content = task.parameters.get("content")
            
            # Generate embedding
            embedding = await self.embedding_generator.generate_embedding(content)
            if not embedding:
                raise Exception("Failed to generate embedding")
            
            # Store in vector database
            collection_name = self.config['databases']['qdrant']['collection_name']
            success = self.database_manager.store_chunk_vector(
                document_id, embedding, collection_name
            )
            
            if success:
                return f"Document {document_id} successfully ingested"
            else:
                raise Exception("Failed to store vector")
                
        except Exception as e:
            raise Exception(f"RAG ingestion failed: {e}")
    
    async def handle_folder_ingestion_task(self, task: Task) -> str:
        """Handle folder ingestion with enhanced progress tracking"""
        progress_tracker = None
        try:
            paths = task.parameters.get("paths", [])
            heartbeat_timeout = task.parameters.get("heartbeat_timeout", 300)
            
            # Initialize progress tracker
            progress_tracker = get_progress_tracker(task.id, heartbeat_timeout)
            progress_tracker.start()
            
            logger.info(f"Starting folder ingestion for task {task.id} with paths: {paths}")
            
            # Discover files first
            all_files = []
            supported_extensions = ['.txt', '.md', '.py', '.pdf', '.docx', '.json']
            
            for path_str in paths:
                path = Path(path_str)
                if path.is_file():
                    if path.suffix.lower() in supported_extensions:
                        all_files.append(path)
                        progress_tracker.record_file_activity('file_discovered', {'file_path': str(path)})
                elif path.is_dir():
                    # Recursively discover files
                    recursive = task.parameters.get("recursive", True)
                    if recursive:
                        for ext in supported_extensions:
                            files_found = list(path.rglob(f"*{ext}"))
                            all_files.extend(files_found)
                            for file_path in files_found:
                                progress_tracker.record_file_activity('file_discovered', {'file_path': str(file_path)})
                    else:
                        for ext in supported_extensions:
                            files_found = list(path.glob(f"*{ext}"))
                            all_files.extend(files_found)
                            for file_path in files_found:
                                progress_tracker.record_file_activity('file_discovered', {'file_path': str(file_path)})
            
            # Update totals
            total_files = len(all_files)
            progress_tracker.update_totals(total_files=total_files, total_embeddings=total_files)
            
            logger.info(f"Discovered {total_files} files for processing")
            
            # Process files
            stats = {'processed_files': 0, 'failed_files': 0, 'total_chunks': 0}
            
            for i, file_path in enumerate(all_files, 1):
                try:
                    logger.debug(f"Processing file {i}/{total_files}: {file_path}")
                    
                    # Read file content
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    
                    # Create document metadata
                    document = DocumentMetadata(
                        id=str(uuid.uuid4()),
                        file_path=str(file_path),
                        file_name=file_path.name,
                        folder_path=str(file_path.parent),
                        file_size=file_path.stat().st_size,
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                        total_chunks=1
                    )
                    
                    # Store document in PostgreSQL
                    success = self.database_manager.store_document(document)
                    if success:
                        progress_tracker.record_database_activity('postgres', 'document_stored', {
                            'document_id': document.id,
                            'file_path': str(file_path)
                        })
                        
                        # Create a chunk for the document
                        progress_tracker.record_file_activity('chunks_created', {'chunk_count': 1})
                        progress_tracker.record_database_activity('postgres', 'chunk_stored', {
                            'document_id': document.id
                        })
                        
                        # Generate embedding
                        progress_tracker.record_embedding_activity('embedding_queued', {
                            'chunk_id': document.id
                        })
                        
                        embedding = await self.embedding_generator.generate_embedding(content)
                        
                        if embedding:
                            # Record embedding generation
                            progress_tracker.record_embedding_activity('embedding_generated', {
                                'chunk_id': document.id,
                                'dimensions': len(embedding)
                            })
                            
                            # Store vector in Qdrant
                            collection_name = self.config['databases']['qdrant']['collection_name']
                            vector_success = self.database_manager.store_chunk_vector(
                                document.id, embedding, collection_name
                            )
                            
                            if vector_success:
                                progress_tracker.record_database_activity('vector', 'vector_stored', {
                                    'chunk_id': document.id,
                                    'vector_size': len(embedding)
                                })
                                stats['processed_files'] += 1
                                stats['total_chunks'] += 1
                                
                                # Record successful file processing
                                progress_tracker.record_file_activity('file_processed', {
                                    'file_path': str(file_path),
                                    'file_size': file_path.stat().st_size
                                })
                            else:
                                logger.error(f"Failed to store vector for {file_path}")
                                progress_tracker.record_embedding_activity('embedding_failed', {
                                    'chunk_id': document.id,
                                    'error': 'Vector storage failed'
                                })
                                stats['failed_files'] += 1
                                progress_tracker.record_file_activity('file_failed', {
                                    'file_path': str(file_path),
                                    'error': 'Vector storage failed'
                                })
                        else:
                            logger.error(f"Failed to generate embedding for {file_path}")
                            progress_tracker.record_embedding_activity('embedding_failed', {
                                'chunk_id': document.id,
                                'error': 'Embedding generation failed'
                            })
                            stats['failed_files'] += 1
                            progress_tracker.record_file_activity('file_failed', {
                                'file_path': str(file_path),
                                'error': 'Embedding generation failed'
                            })
                    else:
                        logger.error(f"Failed to store document metadata for {file_path}")
                        stats['failed_files'] += 1
                        progress_tracker.record_file_activity('file_failed', {
                            'file_path': str(file_path),
                            'error': 'Document storage failed'
                        })
                        
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    stats['failed_files'] += 1
                    progress_tracker.record_file_activity('file_failed', {
                        'file_path': str(file_path),
                        'error': str(e)
                    })
            
            logger.info(f"Folder ingestion completed: {stats}")
            return json.dumps(stats)
            
        except Exception as e:
            logger.error(f"Folder ingestion failed: {e}")
            raise Exception(f"Folder ingestion failed: {e}")
        finally:
            # Clean up progress tracker
            if progress_tracker:
                progress_tracker.stop()
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID"""
        return self.tasks.get(task_id)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        status_counts = {}
        for status in TaskStatus:
            status_counts[status.value] = sum(
                1 for task in self.tasks.values() if task.status == status
            )
        
        return {
            "total_tasks": len(self.tasks),
            "queue_length": len(self.task_queue),
            "status_breakdown": status_counts,
            "running": self.running
        }

# FastAPI Application
app = FastAPI(title="Orchestrator Service (Refactored)", version="2.0.0")
orchestrator = Orchestrator()

@app.on_event("startup")
async def startup_event():
    logger.info("Orchestrator service starting with new modular architecture...")
    # Initialize database
    orchestrator.database_manager.initialize_databases()
    await orchestrator.start()

@app.on_event("shutdown") 
async def shutdown_event():
    logger.info("Orchestrator service shutting down...")
    await orchestrator.stop()

@app.post("/tasks")
async def create_task(task_request: TaskCreateRequest = Body(...)): 
    """Create a new task"""
    priority_enum = TaskPriority.NORMAL
    if task_request.priority.lower() == "high":
        priority_enum = TaskPriority.HIGH
    elif task_request.priority.lower() == "critical":
        priority_enum = TaskPriority.CRITICAL
    elif task_request.priority.lower() == "low":
        priority_enum = TaskPriority.LOW

    task = Task(
        type=task_request.task_type,
        description=task_request.description,
        parameters=task_request.parameters or {},
        dependencies=task_request.dependencies or [],
        priority=priority_enum
    )

    task_id = orchestrator.add_task(task)
    logger.info(f"Task created with ID: {task_id}")
    return {"task_id": task_id, "status": "created"}

@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get task status and details"""
    task = orchestrator.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return {
        "id": task.id,
        "type": task.type,
        "description": task.description,
        "status": task.status.value,
        "created_at": task.created_at.isoformat(),
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "result": task.result,
        "error": task.error
    }

@app.get("/status")
async def get_system_status():
    """Get overall system status"""
    return orchestrator.get_system_status()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/tasks/{task_id}/progress")
async def get_task_progress(task_id: str):
    """Get real-time progress for a specific task"""
    try:
        # Check if progress tracker exists for this task
        progress_tracker = get_progress_tracker(task_id)
        
        if not progress_tracker.active:
            return {"error": "No active progress tracking for this task"}
        
        stats = progress_tracker.get_current_stats()
        return {
            "task_id": task_id,
            "progress": stats,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting task progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def setup_logging():
    """Configure logging to write to logs directory"""
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging with both file and console handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "orchestrator-app.log"),
            logging.StreamHandler()
        ]
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    args = parser.parse_args()
    
    setup_logging()
    logger.info("Starting Orchestrator Service (Refactored)...")
    logger.info(f"Logging configured - writing to logs/orchestrator-app.log")
    uvicorn.run(app, host=args.host, port=args.port)