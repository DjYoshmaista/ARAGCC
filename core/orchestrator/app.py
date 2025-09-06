"""
Advanced Multi-Agent Orchestrator Service

Provides centralized task orchestration, dependency management, and
high-performance parallel processing for the AgenticRAG system.
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks, Body
from pydantic import BaseModel, Field
import uvicorn
import logging
import logging.config
import yaml
import os
import sys
from dotenv import load_dotenv
from contextlib import asynccontextmanager

load_dotenv()

# Add the current directory to Python path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Use absolute imports instead of relative imports
from core.database.manager import DatabaseManager, DocumentMetadata, ChunkMetadata
from core.ingestion.text_processor import TextChunker
from core.ingestion.controller import ParallelIngestionController
from core.ingestion.progress_tracker import get_progress_tracker, cleanup_progress_tracker, ProgressUpdate

def load_config() -> Dict[str, Any]:
    """
    Load system configuration from YAML file with error handling.
    
    Returns:
        Dictionary containing system configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'configs', 'system.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if not isinstance(config, dict):
                raise ValueError("Configuration file must contain a YAML dictionary")
            return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in configuration file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}")
        raise

# --- Configure Logging ---
config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'configs', 'system.yaml')
logger = logging.getLogger("orchestrator") # Create logger for this module

def setup_logging():
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging_config = config.get('logging', {})
        log_level = logging_config.get('level', 'INFO')
        log_format = logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = logging_config.get('file', 'logs/orchestrator-app.log') # Default if not in conf

        # Ensure log dir exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file), # Log to file
                logging.StreamHandler()         # Also log to console captured by start-system.sh
            ]
        )
        logger.info("Orchestrator logging configured successfullly.")
    except Exception as e:
        # Fallback logging setup
        logging.basicConfig(level=logging.INFO)
        logger.error(f"Failed to load logging config from {config_path}: {e}.  Using Default settings.")


setup_logging()

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class TaskCreateRequest(BaseModel):
    """Request model for creating new tasks"""
    task_type: str = Field(..., description="Type of task to execute")
    description: str = Field(..., description="Human-readable task description")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Task-specific parameters")
    dependencies: Optional[List[str]] = Field(default_factory=list, description="List of task IDs this task depends on")
    priority: Optional[str] = Field(default="normal", description="Task priority: low, normal, high, critical")
    deadline: Optional[str] = Field(None, description="Task deadline in ISO format")
    
    class Config:
        schema_extra = {
            "example": {
                "task_type": "llm_inference",
                "description": "Generate response for user query",
                "parameters": {
                    "model": "qwen3:8b",
                    "prompt": "What is machine learning?",
                    "temperature": 0.7
                },
                "dependencies": [],
                "priority": "normal"
            }
        }

@dataclass
class Task:
    """
    Represents a task in the orchestration system.
    
    Tracks task lifecycle, dependencies, and execution metadata.
    """
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
    deadline: Optional[datetime] = None
    result: Optional[str] = None
    error: Optional[str] = None
    agent_id: Optional[str] = None
    progress: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization"""
        task_dict = asdict(self)
        # Convert enums and datetimes to serializable formats
        task_dict['priority'] = self.priority.value if isinstance(self.priority, TaskPriority) else self.priority
        task_dict['status'] = self.status.value if isinstance(self.status, TaskStatus) else self.status
        
        # Convert datetime objects
        for dt_field in ['created_at', 'started_at', 'completed_at', 'deadline']:
            if task_dict[dt_field] and isinstance(task_dict[dt_field], datetime):
                task_dict[dt_field] = task_dict[dt_field].isoformat()
        
        return task_dict
    
    def is_overdue(self) -> bool:
        """Check if task is past its deadline"""
        if not self.deadline:
            return False
        return datetime.now(timezone.utc) > self.deadline
    
    def get_execution_time(self) -> Optional[timedelta]:
        """Get task execution time if completed"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

@dataclass
class Agent:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    status: str = "idle"
    capabilities: List[str] = field(default_factory=list)
    current_task: Optional[str] = None
    performance_score: float = 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class Orchestrator:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        # --- Use port from system.yaml or default ---
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.model_gateway_url = config['services']['model_gateway']['ollama_url']
            # Use typical structure and ports
            mg_host = config['services']['model_gateway'].get('host', 'localhost')
            mg_port = config['services']['model_gateway'].get('port', 8001)
            self.model_gateway_url = f"http://{mg_host}:{mg_port}"
            logger.info(f"Model Gateway URL configured as {self.model_gateway_url}")
        except Exception as e:
            self.model_gateway_url = "http://localhost:8001" # Hardcoded fallback
            logger.info(f"Failed to load Model URL from config: {e}.  Model Gateway URL configured as default: {self.model_gateway_url}")
        self.agents: Dict[str, Agent] = {}
        self.task_queue: List[Task] = []
        self.model_gateway_url = "http://localhost:8070"
        self.vector_engine_url = "http://localhost:8080"
        self.running = False
        
    async def start(self):
        """Start the orchestrator"""
        self.running = True
        # Start background task processing
        asyncio.create_task(self.process_task_queue())
        print("Orchestrator started")
    
    async def stop(self):
        """Stop the orchestrator"""
        self.running = False
        print("Orchestrator stopped")
    
    async def process_task_queue(self):
        """Main task processing loop"""
        while self.running:
            if self.task_queue:
                # Sort by priority and creation time
                self.task_queue.sort(
                    key=lambda t: (t.priority.value, t.created_at), 
                    reverse=True
                )
                
                task = self.task_queue.pop(0)
                await self.execute_task(task)
            
            await asyncio.sleep(1)  # Check queue every second
    
    def add_task(self, task: Task) -> str:
        """Add a new task to the system"""
        self.tasks[task.id] = task
        
        # Check if dependencies are met
        if self.check_dependencies(task):
            self.task_queue.append(task)
        
        return task.id
    
    def check_dependencies(self, task: Task) -> bool:
        """Check if all task dependencies are completed"""
        for dep_id in task.dependencies:
            if dep_id not in self.tasks:
                return False
            if self.tasks[dep_id].status != TaskStatus.COMPLETED:
                return False
        return True
    
    async def execute_task(self, task: Task):
        """Execute a single task"""
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now(timezone.utc)
            
            logger.info(f"Executing task {task.id}: {task.description}")
            
            # Start progress monitoring for long-running tasks
            progress_task = None
            if task.type in ["folder_ingestion", "rag_ingest"]:
                progress_task = asyncio.create_task(self._update_task_progress(task))
            
            # Route task based on type
            if task.type == "llm_inference":
                result = await self.handle_llm_task(task)
            elif task.type == "vector_search":
                result = await self.handle_vector_task(task)
            elif task.type == "rag_ingest":
                result = await self.handle_rag_ingest_task(task)
            elif task.type == "rag_query":
                result = await self.handle_rag_query_task(task)
            elif task.type == "folder_ingestion":
                result = await self.handle_folder_ingestion_task(task)
            elif task.type == "decomposition":
                result = await self.handle_decomposition_task(task)
            else:
                result = await self.handle_generic_task(task)
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(timezone.utc)
            
            # Cancel progress monitoring
            if progress_task and not progress_task.done():
                progress_task.cancel()
            
            # Check for tasks that were waiting on this one
            await self.check_dependent_tasks(task.id)
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now(timezone.utc)
            logger.error(f"Task {task.id} failed: {e}")
    
    async def handle_llm_task(self, task: Task) -> str:
        """Handle LLM inference tasks"""
        logger.info(f"Handling LLM task {task.id} with model {task.parameters.get('model', 'default')}")
        try:
            async with httpx.AsyncClient() as client:
                request_data = {
                        "model": task.parameters.get("model", "granite-embedding:latest"), # <-- Fixed: task.parameters
                    "prompt": task.parameters.get("prompt", ""),     # <-- Fixed: task.parameters
                    "stream": False,
                    "parameters": task.parameters.get("model_params", {}) # <-- Fixed: task.parameters
                }
                logger.debug(f"Sending request to Model Gateway ({self.model_gateway_url}/generate): {request_data}")
                response = await client.post(
                        f"{self.model_gateway_url}/generate",
                        json=request_data,
                        timeout=60.0
                    )
                logger.debug(f"Received response from Model Gateway: Status {response.status_code}")
                response.raise_for_status()
                data = response.json()
                logger.info(f"LLM task {task.id} completed successfully.")
                return data.get("response", "")
        except httpx.RequestError as e:
            logger.error(f"Network error during LLM task {task.id}: {e}")
            raise Exception(f"Network error calling model gateway: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from Model Gateway for task {task.id}: {e.response.status_code} - {e.response.text}")
            raise Exception(f"Model gateway returned error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            logger.error(f"Unexpected error handling LLM task {task.id}: {e}")
            raise Exception(f"LLM task failed: {e}")
    
    async def handle_vector_task(self, task: Task) -> str:
        """Handle vector search tasks"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.vector_engine_url}/search",
                    json={
                        "type": "search",
                        "vector": task.parameters.get("vector", []),
                        "top_k": task.parameters.get("top_k", 5)
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                return response.text
        except Exception as e:
            raise Exception(f"Vector task failed: {e}")
    
    async def handle_rag_ingest_task(self, task: Task) -> str:
        """Handle RAG document ingestion tasks"""
        try:
            document_id = task.parameters.get("document_id")
            content = task.parameters.get("content")
            # Load config
            config = load_config()
            embedding_model = task.parameters.get("embedding_model") or config.get("models", {}).get("embedding_model", "granite-embedding:latest")
            
            # Step 1: Generate embedding using model gateway
            async with httpx.AsyncClient() as client:
                embedding_response = await client.post(
                    f"{self.model_gateway_url}/embed",
                    json={
                        "model": embedding_model,
                        "prompt": content
                    },
                    timeout=60.0
                )
                embedding_response.raise_for_status()
                embedding_data = embedding_response.json()
                embedding = embedding_data.get("embedding")
            
            # Step 2: Store vector in vector engine
            vector_response = await client.post(
                f"{self.vector_engine_url}/vectors",
                json={
                    "id": document_id,
                    "vector": embedding
                },
                timeout=30.0
            )
            vector_response.raise_for_status()
            
            return f"Document {document_id} successfully ingested into RAG"
        except Exception as e:
            raise Exception(f"RAG ingestion failed: {e}")
    
    async def handle_rag_query_task(self, task: Task) -> str:
        """Handle RAG query tasks"""
        try:
            query_text = task.parameters.get("query")
            top_k = task.parameters.get("top_k", 5)
            # Load config
            config = load_config()
            embedding_model = task.parameters.get("embedding_model") or config.get("models", {}).get("embedding_model", "granite-embedding:latest")
            
            # Step 1: Generate embedding for query
            async with httpx.AsyncClient() as client:
                embedding_response = await client.post(
                    f"{self.model_gateway_url}/embed",
                    json={
                        "model": embedding_model,
                        "prompt": query_text
                    },
                    timeout=60.0
                )
                embedding_response.raise_for_status()
                embedding_data = embedding_response.json()
                query_embedding = embedding_data.get("embedding")
            
            # Step 2: Search vector engine
            search_response = await client.post(
                f"{self.vector_engine_url}/search",
                json={
                    "vector": query_embedding,
                    "top_k": top_k
                },
                timeout=30.0
            )
            search_response.raise_for_status()
            search_results = search_response.json()
            
            return json.dumps(search_results)
        except Exception as e:
            raise Exception(f"RAG query failed: {e}")
    
    async def handle_decomposition_task(self, task: Task) -> str:
        """Handle task decomposition"""
        # This is where we'll implement the scoring metrics later
        description = task.parameters.get("original_task", "")
        
        # Simple decomposition for now - we'll enhance this
        subtasks = await self.decompose_task_simple(description)
        
        # Create subtasks
        subtask_ids = []
        for i, subtask_desc in enumerate(subtasks):
            subtask = Task(
                type="llm_inference",
                description=subtask_desc,
                parameters={
                    "model": "qwen3:30b",
                    "prompt": f"Complete this subtask: {subtask_desc}"
                },
                dependencies=[task.id] if i == 0 else [subtask_ids[-1]]
            )
            subtask_id = self.add_task(subtask)
            subtask_ids.append(subtask_id)
        
        return json.dumps({"subtask_ids": subtask_ids})
    
    async def handle_folder_ingestion_task(self, task: Task) -> str:
        """Handle folder ingestion task with enhanced progress tracking"""
        tracker = None
        try:
            # Get task parameters
            paths = task.parameters.get("paths", [])
            recursive = task.parameters.get("recursive", True)
            model = task.parameters.get("embedding_model", "granite-embedding:latest")
            heartbeat_timeout = task.parameters.get("heartbeat_timeout", 300)  # 5 minutes default
            
            logger.info(f"Processing folder ingestion task with paths: {paths}")
            
            # Initialize progress tracker for enhanced monitoring
            tracker = get_progress_tracker(task.id, heartbeat_timeout)
            tracker.start()
            logger.info(f"Progress tracker started for task {task.id}")
            
            # Load system configuration
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'configs', 'system.yaml')
            with open(config_path, 'r') as f:
                system_config = yaml.safe_load(f)
            
            # Create ingestion configuration
            ingestion_config = {
                "postgresql": system_config["databases"]["postgresql"],
                "qdrant": system_config["databases"]["qdrant"],
                "model_gateway_url": f"http://{system_config['services']['model_gateway']['host']}:{system_config['services']['model_gateway']['port']}",
                "chunk_size": task.parameters.get("chunk_size", 5000),
                "overlap_size": task.parameters.get("overlap_size", 32),
                "max_concurrent_files": task.parameters.get("max_concurrent_files", 5),
                "max_concurrent_chunks": task.parameters.get("max_concurrent_chunks", 10),
                "embedding_dimensions": 384  # For granite-embedding:latest
            }
            
            # Create and initialize ingestion controller
            controller = ParallelIngestionController(ingestion_config)
            
            # Set progress tracker on controller for enhanced monitoring
            controller.progress_tracker = tracker
            
            # Store controller reference in task for progress tracking
            task.controller = controller
            
            # Initialize database connections
            postgresql_config = system_config.get('databases', {}).get('postgresql', {})
            qdrant_config = system_config.get('databases', {}).get('qdrant', {})
            controller.initialize_database(postgresql_config, qdrant_config)
            
            # Process all paths (folders and files)
            batch_id = None
            stats = {'processed_files': 0, 'failed_files': 0, 'total_chunks': 0}
            
            for path_str in paths:
                path = Path(path_str)
                if path.is_dir():
                    # Process folder with parallel ingestion
                    logger.info(f"Processing folder: {path}")
                    batch_id = controller.process_folder(str(path), recursive, postgresql_config, qdrant_config)
                elif path.is_file():
                    # For single files, create temporary folder processing
                    logger.info(f"Processing single file: {path}")
                    batch_id = controller.process_folder(str(path.parent), False, postgresql_config, qdrant_config)
                else:
                    logger.warning(f"Path does not exist: {path}")
            
            # Get final statistics
            stats = controller.get_stats()
            stats['batch_id'] = batch_id
            
            # Clean up controller reference
            if hasattr(task, 'controller'):
                delattr(task, 'controller')
            
            # Convert datetime objects to ISO format strings for JSON serialization
            # Handle nested dictionaries that might contain datetime objects
            def serialize_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: serialize_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [serialize_datetime(item) for item in obj]
                else:
                    return obj
            
            json_safe_stats = serialize_datetime(stats)
            logger.info(f"Folder ingestion completed with stats: {json_safe_stats}")
            return json.dumps(json_safe_stats)
        except Exception as e:
            logger.error(f"Folder ingestion failed: {e}", exc_info=True)
            # Stop progress tracker
            if tracker:
                tracker.stop()
            raise Exception(f"Folder ingestion failed: {e}")
        finally:
            # Clean up progress tracker
            if tracker:
                tracker.stop()
                logger.info(f"Progress tracker stopped for task {task.id}")
    
    async def _update_task_progress(self, task: Task):
        """Periodically update task progress information"""
        while task.status == TaskStatus.RUNNING:
            try:
                # Check if controller is available
                if hasattr(task, 'controller') and task.controller:
                    # Get progress information from controller
                    progress_info = task.controller.get_stats()
                    task.progress = progress_info
                
                # Wait before next update
                await asyncio.sleep(1)  # Update every second
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error updating task progress: {e}")
                await asyncio.sleep(1)
    
    async def decompose_task_simple(self, task_description: str) -> List[str]:
        """Simple task decomposition using LLM"""
        prompt = f"""
        Break down this complex task into 3-5 smaller, manageable subtasks:
        
        Task: {task_description}
        
        Return only a numbered list of subtasks, one per line.
        """
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.model_gateway_url}/generate",
                    json={
                        "model": "qwen3:30b",
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                data = response.json()
                
                # Parse the response into subtasks
                response_text = data.get("response", "")
                lines = [line.strip() for line in response_text.split('\n') if line.strip()]
                subtasks = []
                
                for line in lines:
                    # Remove numbering and clean up
                    clean_line = line.lstrip('0123456789. ').strip()
                    if clean_line:
                        subtasks.append(clean_line)
                
                return subtasks[:5]  # Limit to 5 subtasks
        except Exception as e:
            print(f"Decomposition failed, using fallback: {e}")
            return [f"Subtask 1: {task_description}"]
    
    async def handle_generic_task(self, task: Task) -> str:
        """Handle generic tasks"""
        # Simulate some work
        await asyncio.sleep(1)
        return f"Completed generic task: {task.description}"
    
    async def check_dependent_tasks(self, completed_task_id: str):
        """Check for tasks that can now run because their dependencies are met"""
        for task in self.tasks.values():
            if (task.status == TaskStatus.PENDING and 
                completed_task_id in task.dependencies and
                self.check_dependencies(task) and
                task not in self.task_queue):
                self.task_queue.append(task)
    
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
            "agents_count": len(self.agents),
            "running": self.running
        }

# FastAPI Application
app = FastAPI(title="Orchestrator Service", version="1.0.0")
orchestrator = Orchestrator()

@app.on_event("startup")
async def startup_event():
    logger.info("Orchestrator service is starting up...")
    await orchestrator.start()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Orchestrator service is shutting down...")
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
        type=task_request.task_type, # Access via task_request
        description=task_request.description, # Access via task_request
        parameters=task_request.parameters or {}, # Access via task_request
        dependencies=task_request.dependencies or [], # Access via task_request
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
    
    # Add progress information if available
    progress_info = {}
    if hasattr(task, 'progress') and task.progress:
        progress_info = task.progress
    
    return {
        "id": task.id,
        "type": task.type,
        "description": task.description,
        "status": task.status.value,
        "created_at": task.created_at.isoformat(),
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "result": task.result,
        "error": task.error,
        "progress": progress_info
    }

@app.get("/status")
async def get_system_status():
    """Get overall system status"""
    return orchestrator.get_system_status()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.debug("Health code endpoint called.")
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/tasks/{task_id}/progress")
async def get_task_progress(task_id: str):
    """Get real-time progress for a specific task"""
    try:
        # Import the progress tracker registry
        from progress_tracker import _progress_trackers, get_progress_tracker
        
        # Check if progress tracker exists for this task
        if task_id not in _progress_trackers:
            logger.warning(f"No progress tracker found for task {task_id}")
            return {"error": "No progress tracking initialized for this task"}
        
        tracker = get_progress_tracker(task_id)
        if not tracker.active:
            logger.warning(f"Progress tracker for task {task_id} is not active")
            return {"error": "No active progress tracking for this task"}
        
        stats = tracker.get_current_stats()
        logger.debug(f"Progress stats for task {task_id}: {stats}")
        return {
            "task_id": task_id,
            "progress": stats,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting task progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tasks/{task_id}/progress/resume")
async def resume_task_progress(task_id: str):
    """Resume paused task after timeout"""
    try:
        tracker = get_progress_tracker(task_id)
        if not tracker.active:
            return {"error": "No active progress tracking for this task"}
        
        tracker.resume()
        return {
            "task_id": task_id,
            "status": "resumed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error resuming task progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tasks/{task_id}/progress/pause")
async def pause_task_progress(task_id: str):
    """Pause task processing"""
    try:
        tracker = get_progress_tracker(task_id)
        if not tracker.active:
            return {"error": "No active progress tracking for this task"}
        
        tracker.pause()
        return {
            "task_id": task_id,
            "status": "paused",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error pausing task progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ingestion/state")
async def get_ingestion_state():
    """Get current ingestion processing state"""
    try:
        from embedding_queue import embedding_queue
        
        # Get embedding queue stats
        embedding_stats = embedding_queue.get_stats()
        
        # Try to load parallel ingestion state
        from parallel_ingestion_controller import ParallelIngestionController
        temp_controller = ParallelIngestionController({})
        temp_controller.load_state()
        parallel_stats = temp_controller.get_stats()
        
        return {
            "embedding_queue": embedding_stats,
            "parallel_ingestion": parallel_stats,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get ingestion state: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get state: {str(e)}")

@app.post("/ingestion/state/clear")
async def clear_ingestion_state():
    """Clear all ingestion processing state"""
    try:
        from embedding_queue import embedding_queue
        from parallel_ingestion_controller import ParallelIngestionController
        
        # Clear embedding queue state
        embedding_queue.clear_state()
        
        # Clear parallel ingestion state
        temp_controller = ParallelIngestionController({})
        temp_controller.clear_state()
        
        return {
            "status": "success",
            "message": "All ingestion state cleared",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to clear ingestion state: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear state: {str(e)}")

@app.post("/ingestion/state/save")
async def save_ingestion_state():
    """Force save current ingestion state"""
    try:
        from embedding_queue import embedding_queue
        
        # Save embedding queue state
        embedding_queue.save_state()
        
        return {
            "status": "success", 
            "message": "Ingestion state saved",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to save ingestion state: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save state: {str(e)}")

@app.get("/ingestion/stats")
async def get_ingestion_stats():
    """Get detailed ingestion processing statistics"""
    try:
        from embedding_queue import embedding_queue
        
        embedding_stats = embedding_queue.get_stats()
        
        return {
            "embedding_queue": embedding_stats,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get ingestion stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.post("/ingest-parallel")
async def ingest_parallel(request: dict):
    """Start parallel ingestion processing with progress tracking"""
    try:
        folder_path = request.get("folder_path", "/tmp")
        max_workers = request.get("max_workers", 4)
        task_id = request.get("task_id", str(uuid.uuid4()))
        heartbeat_timeout = request.get("heartbeat_timeout", 300)  # 5 minutes default
        
        logger.info(f"Starting parallel ingestion for {folder_path} with {max_workers} workers")
        
        # Initialize progress tracker
        tracker = get_progress_tracker(task_id, heartbeat_timeout)
        tracker.start()
        
        # Import and use the parallel ingestion controller
        from parallel_ingestion_controller import ParallelIngestionController
        
        # Load configuration
        config = load_config()
        
        # Create and start parallel ingestion
        controller = ParallelIngestionController(config)
        
        # Set progress tracker on controller
        controller.progress_tracker = tracker
        
        # Start processing in background
        import threading
        def process_in_background():
            try:
                batch_id = controller.process_folder(folder_path, recursive=True)
                tracker.record_file_activity("ingestion_completed", {
                    "batch_id": batch_id,
                    "folder_path": folder_path
                })
                logger.info(f"Parallel ingestion completed with batch_id: {batch_id}")
            except Exception as e:
                tracker.record_file_activity("ingestion_failed", {
                    "error": str(e),
                    "folder_path": folder_path
                })
                logger.error(f"Parallel ingestion failed: {e}")
            finally:
                # Don't stop tracker here - let client decide when to cleanup
                pass
        
        thread = threading.Thread(target=process_in_background, daemon=True)
        thread.start()
        
        return {
            "status": "started",
            "task_id": task_id,
            "folder_path": folder_path,
            "max_workers": max_workers,
            "heartbeat_timeout": heartbeat_timeout,
            "message": "Parallel ingestion started in background with progress tracking"
        }
        
    except Exception as e:
        logger.error(f"Failed to start parallel ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start ingestion: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting Orchestrator application...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
