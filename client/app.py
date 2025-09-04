#!/usr/bin/env python3
"""
AgenticRAG CLI - Advanced Multi-Agent System Command Line Interface

This CLI provides comprehensive interaction with the AgenticRAG multi-agent system,
supporting file processing, RAG database operations, model configuration, and more.

Usage:
  agentic-rag <command> [options] [files/folders...]
  agentic-rag --help
  agentic-rag --version
"""

import argparse
import asyncio
import json
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import httpx
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import subprocess
import tempfile
import threading
import time
from tqdm import tqdm
import warnings

# CLI Configuration
CLI_VERSION = "1.0.0"
CLI_NAME = "agentic-rag"
CONFIG_DIR = Path.home() / ".config" / "agentic-rag"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
LOG_DIR = Path.home() / ".local" / "share" / "agentic-rag" / "logs"

@dataclass
class CLIConfig:
    """CLI configuration structure"""
    # Service endpoints
    orchestrator_url: str = "http://localhost:8001"
    model_gateway_url: str = "http://localhost:8070"
    vector_engine_url: str = "http://localhost:8080"
    
    # Database settings
    postgresql_host: str = "localhost"
    postgresql_port: int = 5432
    postgresql_db: str = "agentic_system"
    postgresql_user: str = "postgres"
    postgresql_password: str = "password"
    
    # RAG settings
    embedding_model: str = "dengcaoQwen3-Embedding-0.6B:Q8_0"
    embedding_dimensions: int = 640  # Dimensions for dengcaoQwen3-Embedding-0.6B
    rag_top_k: int = 5
    rag_weight: float = 0.0  # -10.0 to 10.0, knowledge vs RAG balance
    
    # Agent settings
    max_agents: int = 5
    agent_memory_timeout: int = 3600  # seconds
    default_context_window: int = 4096
    default_temperature: float = 0.7
    
    # Model settings
    default_model: str = "qwen3:8b"
    models: Dict[str, Dict] = None
    
    # Plugin settings
    plugin_dir: Path = CONFIG_DIR / "plugins"
    enabled_plugins: List[str] = None
    
    # Interface settings
    interactive_mode: bool = True
    color_output: bool = True
    verbose: bool = False
    
    def __post_init__(self):
        if self.models is None:
            self.models = {
                "orchestration": {"model": "qwen3:8b", "temperature": 0.3},
                "inference": {"model": "qwen3:8b", "temperature": 0.7},
                "embedding": {"model": "dengcaoQwen3-Embedding-0.6B:Q8_0"},
                "rag": {"model": "qwen3:8b", "temperature": 0.5}
            }
        if self.enabled_plugins is None:
            self.enabled_plugins = []

class CLIManager:
    """Main CLI management class"""
    
    def __init__(self):
        self.config: CLIConfig = CLIConfig()
        self.http_client: Optional[httpx.AsyncClient] = None
        self.logger = self._setup_logging()
        self._ensure_directories()
        self._load_config()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        logger = logging.getLogger(CLI_NAME)
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(LOG_DIR / f"{CLI_NAME}.log")
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _ensure_directories(self):
        """Ensure necessary directories exist"""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.config.plugin_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self):
        """Load configuration from file"""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Update config with loaded data
                for key, value in config_data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                
                self.logger.info("Configuration loaded successfully")
            except Exception as e:
                self.logger.error(f"Error loading config: {e}")
                print(f"Warning: Could not load config file: {e}")
    
    def _save_config(self):
        """Save current configuration to file"""
        try:
            with open(CONFIG_FILE, 'w') as f:
                yaml.dump(asdict(self.config), f, default_flow_style=False)
            self.logger.info("Configuration saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            print(f"Error: Could not save config file: {e}")
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=300.0)  # 5 minutes for large ingestion tasks
        return self.http_client
    
    async def close(self):
        """Cleanup resources"""
        if self.http_client:
            await self.http_client.aclose()

class CommandParser:
    """Command line argument parser"""
    
    def __init__(self, cli_manager: CLIManager):
        self.cli_manager = cli_manager
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser"""
        parser = argparse.ArgumentParser(
            prog=CLI_NAME,
            description="Advanced Multi-Agent RAG System CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  agentic-rag query "What is machine learning?" --rag-weight 5.0
  agentic-rag ingest /path/to/documents --recursive
  agentic-rag config --set embedding_model=sentence-transformers/all-mpnet-base-v2
  agentic-rag agents --list
  agentic-rag db --create-table embeddings --dimensions 768
  agentic-rag tui  # Launch terminal UI
            """
        )
        
        parser.add_argument('--version', action='version', version=f'{CLI_NAME} {CLI_VERSION}')
        parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
        parser.add_argument('--config', help='Path to config file')
        parser.add_argument('--no-color', action='store_true', help='Disable colored output')
        
        # Create subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Query command
        query_parser = subparsers.add_parser('query', help='Submit a query to the system')
        query_parser.add_argument('prompt', help='The query prompt')
        query_parser.add_argument('--model', help='Model to use for inference')
        query_parser.add_argument('--temperature', type=float, help='Temperature for generation')
        query_parser.add_argument('--rag-weight', type=float, help='RAG vs knowledge weight (-10.0 to 10.0)')
        query_parser.add_argument('--top-k', type=int, help='Number of RAG results to retrieve')
        query_parser.add_argument('--context-window', type=int, help='Context window size')
        query_parser.add_argument('--stream', action='store_true', help='Stream the response')
        query_parser.add_argument('--save-context', help='Save conversation context to file')
        
        # Ingest command
        ingest_parser = subparsers.add_parser('ingest', help='Ingest documents into RAG database')
        ingest_parser.add_argument('paths', nargs='+', help='Files or directories to ingest')
        ingest_parser.add_argument('--recursive', '-r', action='store_true', help='Process directories recursively')
        ingest_parser.add_argument('--file-types', help='Comma-separated file extensions to process')
        ingest_parser.add_argument('--chunk-size', type=int, default=5000, help='Text chunk size for embeddings')
        ingest_parser.add_argument('--overlap', type=int, default=32, help='Chunk overlap size')
        ingest_parser.add_argument('--embedding-model', help='Model for generating embeddings')
        ingest_parser.add_argument('--batch-size', type=int, default=100, help='Processing batch size')
        ingest_parser.add_argument('--max-concurrent-files', type=int, default=5, help='Maximum concurrent file processing')
        ingest_parser.add_argument('--max-concurrent-chunks', type=int, default=10, help='Maximum concurrent chunk processing')
        ingest_parser.add_argument('--resume', action='store_true', help='Resume previous interrupted ingestion')
        ingest_parser.add_argument('--clear-state', action='store_true', help='Clear saved state before starting')
        
        # Embedding generation command
        embed_parser = subparsers.add_parser('embed', help='Generate embeddings with state recovery')
        embed_parser.add_argument('--continue', action='store_true', help='Continue from saved state')
        embed_parser.add_argument('--clear-state', action='store_true', help='Clear saved embedding state')
        embed_parser.add_argument('--stats', action='store_true', help='Show embedding processing statistics')
        embed_parser.add_argument('--workers', type=int, default=4, help='Number of embedding workers')
        embed_parser.add_argument('--batch-size', type=int, default=32, help='Embedding batch size')
        
        # Configuration command
        config_parser = subparsers.add_parser('config', help='Manage configuration')
        config_parser.add_argument('--show', action='store_true', help='Show current configuration')
        config_parser.add_argument('--set', action='append', help='Set config value (key=value)')
        config_parser.add_argument('--reset', action='store_true', help='Reset to default configuration')
        config_parser.add_argument('--validate', action='store_true', help='Validate configuration')
        
        # Agent management command
        agent_parser = subparsers.add_parser('agents', help='Manage agents')
        agent_parser.add_argument('--list', action='store_true', help='List active agents')
        agent_parser.add_argument('--create', help='Create new agent with specified type')
        agent_parser.add_argument('--stop', help='Stop agent by ID')
        agent_parser.add_argument('--info', help='Get agent information by ID')
        agent_parser.add_argument('--max-agents', type=int, help='Set maximum number of agents')
        
        # Database management command
        db_parser = subparsers.add_parser('db', help='Database operations')
        db_parser.add_argument('--create-table', help='Create table with specified name')
        db_parser.add_argument('--list-tables', action='store_true', help='List all tables')
        db_parser.add_argument('--table-info', help='Get table information')
        db_parser.add_argument('--dimensions', type=int, help='Vector dimensions for table creation')
        db_parser.add_argument('--migrate', action='store_true', help='Run database migrations')
        db_parser.add_argument('--backup', help='Backup database to file')
        db_parser.add_argument('--restore', help='Restore database from file')
        
        # Plugin management command
        plugin_parser = subparsers.add_parser('plugins', help='Plugin management')
        plugin_parser.add_argument('--list', action='store_true', help='List available plugins')
        plugin_parser.add_argument('--enable', help='Enable plugin by name')
        plugin_parser.add_argument('--disable', help='Disable plugin by name')
        plugin_parser.add_argument('--install', help='Install plugin from path or URL')
        plugin_parser.add_argument('--create', help='Create new plugin template')
        
        # System status command
        status_parser = subparsers.add_parser('status', help='System status')
        status_parser.add_argument('--services', action='store_true', help='Check service status')
        status_parser.add_argument('--health', action='store_true', help='Run health checks')
        status_parser.add_argument('--metrics', action='store_true', help='Show system metrics')
        
        # Terminal UI command
        tui_parser = subparsers.add_parser('tui', help='Launch terminal user interface')
        tui_parser.add_argument('--no-mouse', action='store_true', help='Disable mouse support')
        
        # Prompt management command
        prompt_parser = subparsers.add_parser('prompts', help='Manage custom prompts')
        prompt_parser.add_argument('--list', action='store_true', help='List saved prompts')
        prompt_parser.add_argument('--create', help='Create new prompt template')
        prompt_parser.add_argument('--edit', help='Edit existing prompt')
        prompt_parser.add_argument('--delete', help='Delete prompt by name')
        prompt_parser.add_argument('--export', help='Export prompts to file')
        prompt_parser.add_argument('--import', help='Import prompts from file')
        
        # RAG management command
        rag_parser = subparsers.add_parser('rag', help='Manage RAG documents')
        rag_parser.add_argument('--ingest', help='Ingest document into RAG')
        rag_parser.add_argument('--query', help='Query RAG documents')
        rag_parser.add_argument('--list', action='store_true', help='List RAG documents')
        rag_parser.add_argument('--delete', help='Delete document from RAG')
        rag_parser.add_argument('--embedding-model', help='Embedding model to use')
        
        return parser
    
    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command line arguments"""
        return self.parser.parse_args(args)

class APIClient:
    """HTTP API client for communicating with backend services"""
    
    def __init__(self, cli_manager: CLIManager):
        self.cli_manager = cli_manager
        self.config = cli_manager.config
        self.logger = cli_manager.logger
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all services"""
        client = await self.cli_manager._get_http_client()
        services = {
            'orchestrator': f"{self.config.orchestrator_url}/health",
            'model_gateway': f"{self.config.model_gateway_url}/health",
            'vector_engine': f"{self.config.vector_engine_url}/"
        }
        
        results = {}
        for service, url in services.items():
            try:
                response = await client.get(url, timeout=5.0)
                results[service] = {
                    'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds()
                }
            except Exception as e:
                results[service] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return results
    
    async def submit_task(self, task_type: str, description: str, 
                         parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a task to the orchestrator"""
        client = await self.cli_manager._get_http_client()
        
        task_data = {
            'task_type': task_type,
            'description': description,
            'parameters': parameters
        }
        
        try:
            response = await client.post(
                f"{self.config.orchestrator_url}/tasks",
                json=task_data
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error submitting task: {e}")
            raise
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status by ID"""
        client = await self.cli_manager._get_http_client()
        
        try:
            response = await client.get(f"{self.config.orchestrator_url}/tasks/{task_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error getting task status: {e}")
            raise
    
    async def generate_response(self, model: str, prompt: str, 
                              parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response using model gateway"""
        client = await self.cli_manager._get_http_client()
        
        request_data = {
            'model': model,
            'prompt': prompt,
            'stream': False,
            'parameters': parameters
        }
        
        try:
            response = await client.post(
                f"{self.config.model_gateway_url}/generate",
                json=request_data
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise

async def main():
    """Main CLI entry point"""
    # Suppress tqdm warnings that might interfere with display
    warnings.filterwarnings("ignore", category=UserWarning, module="tqdm")
    
    cli_manager = CLIManager()
    parser = CommandParser(cli_manager)
    api_client = APIClient(cli_manager)
    
    try:
        args = parser.parse_args()
        
        # Update config based on args
        if args.verbose:
            cli_manager.config.verbose = True
            cli_manager.logger.setLevel(logging.DEBUG)
        
        if args.no_color:
            cli_manager.config.color_output = False
        
        # Handle commands
        if not args.command:
            parser.parser.print_help()
            return
        
        if args.command == 'query':
            await handle_query_command(args, cli_manager, api_client)
        elif args.command == 'ingest':
            await handle_ingest_command(args, cli_manager, api_client)
        elif args.command == 'embed':
            await handle_embed_command(args, cli_manager, api_client)
        elif args.command == 'config':
            await handle_config_command(args, cli_manager)
        elif args.command == 'status':
            await handle_status_command(args, cli_manager, api_client)
        elif args.command == 'tui':
            await handle_tui_command(args, cli_manager)
        elif args.command == 'rag':
            await handle_rag_command(args, cli_manager, api_client)
        else:
            print(f"Command '{args.command}' is not yet implemented.")
            print("This is part of the iterative development process.")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
        cli_manager.logger.error(f"CLI error: {e}")
    finally:
        await cli_manager.close()

# Command handlers (to be implemented in following steps)
async def handle_query_command(args, cli_manager, api_client):
    """Handle query command"""
    print(f"Query: {args.prompt}")
    print("This command will be fully implemented in the next iteration.")

async def handle_config_command(args, cli_manager):
    """Handle configuration command"""
    if args.show:
        print(yaml.dump(asdict(cli_manager.config), default_flow_style=False))
    else:
        print("Configuration management will be implemented in the next iteration.")

async def handle_status_command(args, cli_manager, api_client):
    """Handle status command"""
    print("Checking system status...")
    health = await api_client.health_check()
    
    for service, status in health.items():
        status_icon = "✅" if status.get('status') == 'healthy' else "❌"
        print(f"{status_icon} {service}: {status.get('status', 'unknown')}")

async def handle_tui_command(args, cli_manager):
    """Handle TUI command"""
    print("Terminal UI will be implemented in requirement 5.")
    print("This requires additional input from you for UI specifications.")

async def handle_rag_command(args, cli_manager, api_client):
    """Handle RAG command"""
    if args.ingest:
        # Ingest document into RAG
        try:
            with open(args.ingest, 'r') as f:
                content = f.read()
            
            # Submit RAG ingestion task to orchestrator
            task_response = await api_client.submit_task(
                task_type="rag_ingest",
                description=f"Ingest document {args.ingest} into RAG",
                parameters={
                    "document_id": os.path.basename(args.ingest),
                    "content": content,
                    "embedding_model": args.embedding_model or cli_manager.config.models["embedding"]["model"]
                }
            )
            
            task_id = task_response["task_id"]
            print(f"Submitted document ingestion task with ID: {task_id}")
            
            # Wait for task completion
            while True:
                task_status = await api_client.get_task_status(task_id)
                if task_status["status"] in ["completed", "failed"]:
                    break
                await asyncio.sleep(1)
            
            if task_status["status"] == "completed":
                print(f"Document successfully ingested: {task_status['result']}")
            else:
                print(f"Document ingestion failed: {task_status['error']}")
                
        except Exception as e:
            print(f"Error ingesting document: {e}")
    
    elif args.query:
        # Query RAG documents
        try:
            # Submit RAG query task to orchestrator
            task_response = await api_client.submit_task(
                task_type="rag_query",
                description=f"Query RAG with: {args.query}",
                parameters={
                    "query": args.query,
                    "top_k": cli_manager.config.rag_top_k,
                    "embedding_model": args.embedding_model or cli_manager.config.models["embedding"]["model"]
                }
            )
            
            task_id = task_response["task_id"]
            print(f"Submitted RAG query task with ID: {task_id}")
            
            # Wait for task completion
            while True:
                task_status = await api_client.get_task_status(task_id)
                if task_status["status"] in ["completed", "failed"]:
                    break
                await asyncio.sleep(1)
            
            if task_status["status"] == "completed":
                print("RAG Results:")
                print(task_status["result"])
            else:
                print(f"RAG query failed: {task_status['error']}")
                
        except Exception as e:
            print(f"Error querying RAG: {e}")
    
    else:
        print("RAG command requires either --ingest or --query parameter.")
        print("Use --help for more information.")

async def handle_ingest_command(args, cli_manager, api_client):
    """Handle document ingestion command with dual progress bars and timeout detection"""
    try:
        orchestrator_url = cli_manager.config.orchestrator_url
        
        # Submit parallel ingestion task to orchestrator
        task_response = await api_client.submit_task(
            task_type="folder_ingestion",
            description=f"Ingest folders/files: {', '.join(args.paths)}",
            parameters={
                "paths": args.paths,
                "recursive": args.recursive,
                "chunk_size": args.chunk_size,
                "overlap_size": args.overlap,
                "embedding_model": args.embedding_model or cli_manager.config.models["embedding"]["model"],
                "max_concurrent_files": args.max_concurrent_files,
                "max_concurrent_chunks": args.max_concurrent_chunks,
                "heartbeat_timeout": 300  # 5 minutes timeout
            }
        )
        
        task_id = task_response["task_id"]
        print(f"🚀 Started ingestion task: {task_id}")
        print("📁 Processing files with dual progress tracking...")
        print("⏱️  Timeout: 5 minutes of database inactivity")
        print()
        
        # Initialize dual progress bars
        postgres_pbar = None
        vector_pbar = None
        
        # Progress tracking
        timeout_threshold = 300  # 5 minutes
        
        print("📊 Initializing progress tracking...")
        
        # Wait for task completion with progress updates
        last_progress_shown = None
        fallback_mode = False
        
        while True:
            try:
                # Get progress from new endpoint
                client = await cli_manager._get_http_client()
                progress_response = await client.get(f"{orchestrator_url}/tasks/{task_id}/progress")
                progress_response.raise_for_status()
                progress_data = progress_response.json()
                
                if "error" in progress_data:
                    if not fallback_mode:
                        print(f"📊 Progress endpoint error: {progress_data.get('error')}")
                        print("📊 Switching to fallback progress monitoring...")
                        fallback_mode = True
                    
                    # Fallback to task status
                    task_status = await api_client.get_task_status(task_id)
                    if task_status["status"] in ["completed", "failed"]:
                        break
                    
                    # Show simple progress in fallback mode
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(f"⏳ [{current_time}] Task {task_id[:8]} - Status: {task_status['status']}")
                    
                    await asyncio.sleep(5)  # Longer interval in fallback
                    continue
                
                progress = progress_data.get("progress", {})
                postgres_stats = progress.get("postgres", {})
                vector_stats = progress.get("vector", {})
                
                # Debug output (first time only)
                if last_progress_shown is None:
                    print(f"📊 Progress data structure received:")
                    print(f"   - PostgreSQL stats: {list(postgres_stats.keys()) if postgres_stats else 'None'}")
                    print(f"   - Vector stats: {list(vector_stats.keys()) if vector_stats else 'None'}")
                    last_progress_shown = True
                
                # Check for timeout condition
                if progress.get("paused", False):
                    print("\n⏸️  Processing paused due to database inactivity timeout!")
                    print(f"   No activity detected for {timeout_threshold // 60} minutes")
                    print("   Options:")
                    print("   1. Continue processing (c)")
                    print("   2. Stop processing (s)")
                    print("   3. Check system status (t)")
                    
                    choice = input("   Your choice [c/s/t]: ").lower().strip()
                    
                    if choice == 'c':
                        # Resume processing
                        client = await cli_manager._get_http_client()
                        resume_response = await client.post(f"{orchestrator_url}/tasks/{task_id}/progress/resume")
                        resume_response.raise_for_status()
                        print("✅ Processing resumed")
                        continue
                    elif choice == 's':
                        # Stop processing
                        client = await cli_manager._get_http_client()
                        pause_response = await client.post(f"{orchestrator_url}/tasks/{task_id}/progress/pause")
                        pause_response.raise_for_status()
                        print("🛑 Processing stopped by user")
                        break
                    elif choice == 't':
                        print("🔍 Checking system status...")
                        continue
                    else:
                        print("   Invalid choice, continuing...")
                        continue
                
                # Initialize progress bars if we have any meaningful data
                # Initialize progress bars earlier to ensure they show up
                if postgres_pbar is None and postgres_stats:
                    # Use a more reasonable estimate based on available data
                    total_files = max(
                        postgres_stats.get("total_files", 100),  # Try to get total from stats
                        postgres_stats.get("files_processed", 0) + 100,  # Add buffer for remaining
                        postgres_stats.get("documents_stored", 0) + 50,
                        100  # Minimum reasonable total
                    )
                    postgres_pbar = tqdm(
                        total=total_files,
                        desc="📝 PostgreSQL",
                        unit="files",
                        position=0,
                        leave=True,
                        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                    )
                    # Force refresh to ensure visibility
                    postgres_pbar.refresh()
                    print("📊 PostgreSQL progress bar initialized")
                
                if vector_pbar is None and vector_stats:
                    # Use actual chunks created or a reasonable estimate
                    total_embeddings = max(
                        postgres_stats.get("chunks_created", 0) if postgres_stats else 0,  # Each chunk needs embedding
                        vector_stats.get("embeddings_generated", 0) + 100,
                        vector_stats.get("embeddings_stored", 0) + 100,
                        100  # Minimum reasonable total
                    )
                    vector_pbar = tqdm(
                        total=total_embeddings,
                        desc="🔮 Vector DB",
                        unit="emb",
                        position=1,
                        leave=True,
                        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                    )
                    # Force refresh to ensure visibility
                    vector_pbar.refresh()
                    print("📊 Vector database progress bar initialized")
                
                # Update progress bars or show text progress
                if postgres_pbar:
                    postgres_pbar.n = postgres_stats.get("files_processed", 0)
                    postgres_pbar.set_postfix({
                        "docs": postgres_stats.get("documents_stored", 0),
                        "chunks": postgres_stats.get("chunks_created", 0),
                        "rate": f"{postgres_stats.get('files_per_second', 0):.1f}/s"
                    })
                    postgres_pbar.refresh()
                else:  # Always show text progress to ensure user feedback
                    files_processed = postgres_stats.get("files_processed", 0)
                    docs_stored = postgres_stats.get("documents_stored", 0)
                    chunks_created = postgres_stats.get("chunks_created", 0)
                    if files_processed > 0 or docs_stored > 0 or chunks_created > 0:
                        current_time = datetime.now().strftime("%H:%M:%S")
                        print(f"📝 [{current_time}] PostgreSQL: {files_processed} files, {docs_stored} docs, {chunks_created} chunks")
                
                if vector_pbar:
                    vector_pbar.n = vector_stats.get("embeddings_stored", 0)
                    vector_pbar.set_postfix({
                        "generated": vector_stats.get("embeddings_generated", 0),
                        "queue": vector_stats.get("queue_size", 0),
                        "rate": f"{vector_stats.get('embeddings_per_second', 0):.1f}/s"
                    })
                    vector_pbar.refresh()
                else:  # Always show text progress to ensure user feedback
                    embeddings_gen = vector_stats.get("embeddings_generated", 0)
                    embeddings_stored = vector_stats.get("embeddings_stored", 0)
                    queue_size = vector_stats.get("queue_size", 0)
                    if embeddings_gen > 0 or embeddings_stored > 0 or queue_size > 0:
                        current_time = datetime.now().strftime("%H:%M:%S")
                        print(f"🔮 [{current_time}] Vector DB: {embeddings_stored} stored, {embeddings_gen} generated, queue: {queue_size}")
                
                # Check task completion
                task_status = await api_client.get_task_status(task_id)
                if task_status["status"] in ["completed", "failed"]:
                    # Final update
                    if postgres_pbar:
                        postgres_pbar.n = postgres_pbar.total
                        postgres_pbar.refresh()
                        postgres_pbar.close()
                    
                    if vector_pbar:
                        vector_pbar.n = vector_pbar.total
                        vector_pbar.refresh()
                        vector_pbar.close()
                    
                    # Ensure progress bars are properly cleaned up
                    if postgres_pbar or vector_pbar:
                        print()  # Add a newline after progress bars
                        
                    print(f"✅ Task {task_status['status']}!")
                    break
                
                await asyncio.sleep(2)  # Update every 2 seconds
                
            except Exception as e:
                print(f"\n⚠️  Progress monitoring error: {e}")
                print(f"   Error type: {type(e).__name__}")
                
                # Try to get basic task status as fallback
                try:
                    task_status = await api_client.get_task_status(task_id)
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(f"⏳ [{current_time}] Fallback - Task status: {task_status['status']}")
                    
                    if task_status["status"] in ["completed", "failed"]:
                        print(f"✅ Task completed with status: {task_status['status']}")
                        break
                except Exception as fallback_error:
                    print(f"⚠️  Could not get fallback status: {fallback_error}")
                
                await asyncio.sleep(5)  # Longer delay on error
        
        # Clean up progress bars
        try:
            if postgres_pbar:
                postgres_pbar.close()
            if vector_pbar:
                vector_pbar.close()
        except:
            pass
        
        # Show final results
        try:
            final_task_status = await api_client.get_task_status(task_id)
            if final_task_status["status"] == "completed":
                if "result" in final_task_status:
                    try:
                        stats = json.loads(final_task_status["result"])
                        print(f"📊 Final Stats:")
                        print(f"   Files: {stats.get('files_ingested', 0)}")
                        print(f"   Chunks: {stats.get('chunks_created', 0)}")
                        print(f"   Embeddings: {stats.get('embeddings_generated', 0)}")
                        if stats.get('errors', 0) > 0:
                            print(f"   Errors: {stats.get('errors', 0)}")
                    except:
                        print("   (Stats parsing failed)")
            elif final_task_status["status"] == "failed":
                print(f"❌ Error: {final_task_status.get('error', 'Unknown error')}")
        except:
            print("   (Could not retrieve final status)")
            
    except httpx.TimeoutException:
        print("Error: Task submission timed out after 5 minutes.")
        print("This may happen with very large datasets. Please check:")
        print("1. All services are running (./shared/scripts/status.sh)")
        print("2. Ollama is responding (curl http://localhost:11434/api/version)")
        print("3. Consider processing smaller batches")
    except httpx.ConnectError:
        print("Error: Cannot connect to orchestrator service.")
        print("Please start the system with: ./shared/scripts/start-system.sh")
    except Exception as e:
        print(f"Error ingesting folders/files: {e}")
        print("For troubleshooting, check the service logs in logs/ directory")

async def handle_embed_command(args, cli_manager, api_client):
    """Handle embedding generation command with state recovery"""
    try:
        orchestrator_url = cli_manager.config.orchestrator_url
        
        if args.stats:
            # Show embedding processing statistics
            print("Fetching embedding statistics...")
            client = await cli_manager._get_http_client()
            response = await client.get(f"{orchestrator_url}/ingestion/stats")
            response.raise_for_status()
            stats = response.json()
            
            print("\n=== Embedding Processing Statistics ===")
            embedding_stats = stats.get('embedding_queue', {})
            
            print(f"Total Queued: {embedding_stats.get('total_queued', 0)}")
            print(f"Total Processed: {embedding_stats.get('total_processed', 0)}")
            print(f"Total Failed: {embedding_stats.get('total_failed', 0)}")
            print(f"Queue Size: {embedding_stats.get('queue_size', 0)}")
            print(f"Result Queue Size: {embedding_stats.get('result_queue_size', 0)}")
            print(f"Workers Running: {embedding_stats.get('workers_running', False)}")
            print(f"Number of Workers: {embedding_stats.get('num_workers', 0)}")
            
            if embedding_stats.get('processing_rate'):
                print(f"Processing Rate: {embedding_stats['processing_rate']:.2f} embeddings/sec")
            if embedding_stats.get('avg_processing_time'):
                print(f"Average Processing Time: {embedding_stats['avg_processing_time']:.3f}s")
                
            return
        
        if args.clear_state:
            # Clear embedding state
            print("Clearing embedding processing state...")
            client = await cli_manager._get_http_client()
            response = await client.post(f"{orchestrator_url}/ingestion/state/clear")
            response.raise_for_status()
            result = response.json()
            print(f"✅ {result.get('message', 'State cleared successfully')}")
            return
        
        # Get current ingestion state
        print("Fetching current ingestion state...")
        client = await cli_manager._get_http_client()
        response = await client.get(f"{orchestrator_url}/ingestion/state")
        response.raise_for_status()
        state = response.json()
        
        print("\n=== Current Ingestion State ===")
        parallel_stats = state.get('parallel_ingestion', {})
        embedding_stats = state.get('embedding_queue', {})
        
        print(f"Batch ID: {parallel_stats.get('batch_id', 'None')}")
        print(f"Total Files: {parallel_stats.get('total_files', 0)}")
        print(f"Processed Files: {parallel_stats.get('processed_files', 0)}")
        print(f"Failed Files: {parallel_stats.get('failed_files', 0)}")
        print(f"Total Chunks: {parallel_stats.get('total_chunks', 0)}")
        print(f"Embeddings Generated: {parallel_stats.get('embeddings_generated', 0)}")
        print(f"Workers Running: {parallel_stats.get('workers_running', False)}")
        
        if getattr(args, 'continue', False):
            # Continue embedding generation from saved state
            print("\n🚀 Continuing embedding generation from saved state...")
            
            try:
                # Import required modules for local embedding generation
                import ollama
                import threading
                import time
                from datetime import datetime
                from tqdm import tqdm
                
                # Initialize ollama client
                ollama_client = ollama.Client()
                
                print("✅ Ollama client initialized")
                print(f"🔧 Using {args.workers} workers with batch size {args.batch_size}")
                
                # Get initial statistics
                client = await cli_manager._get_http_client()
                response = await client.get(f"{orchestrator_url}/ingestion/stats")
                response.raise_for_status()
                initial_stats = response.json()
                
                embedding_stats = initial_stats.get('embedding_queue', {})
                queue_size = embedding_stats.get('queue_size', 0)
                total_processed = embedding_stats.get('total_processed', 0)
                
                if queue_size == 0:
                    print("✅ No pending embeddings found. All processing appears complete!")
                    return
                
                print(f"📊 Found {queue_size} pending embeddings")
                
                # Initialize progress bar
                pbar = tqdm(
                    total=queue_size + total_processed,
                    initial=total_processed,
                    desc="Embedding Progress",
                    unit="embeddings"
                )
                # Force refresh to ensure visibility
                pbar.refresh()
                
                # Monitor progress
                start_time = time.time()
                last_processed = total_processed
                
                print("\n🔄 Monitoring embedding generation progress...")
                print("Press Ctrl+C to stop monitoring (embeddings will continue in background)")
                
                try:
                    while True:
                        # Get current statistics
                        client = await cli_manager._get_http_client()
                        response = await client.get(f"{orchestrator_url}/ingestion/stats")
                        response.raise_for_status()
                        current_stats = response.json()
                        
                        current_embedding_stats = current_stats.get('embedding_queue', {})
                        current_processed = current_embedding_stats.get('total_processed', 0)
                        current_queue_size = current_embedding_stats.get('queue_size', 0)
                        current_failed = current_embedding_stats.get('total_failed', 0)
                        
                        # Update progress bar
                        pbar.n = current_processed
                        pbar.refresh()
                        
                        # Check if completed
                        if current_queue_size == 0:
                            pbar.close()
                            elapsed = time.time() - start_time
                            total_generated = current_processed - last_processed
                            rate = total_generated / max(elapsed, 1)
                            
                            print(f"\n✅ Embedding generation completed!")
                            print(f"📊 Generated {total_generated} embeddings in {elapsed:.1f}s ({rate:.2f}/sec)")
                            if current_failed > 0:
                                print(f"⚠️ Failed embeddings: {current_failed}")
                            break
                        
                        await asyncio.sleep(2)  # Update every 2 seconds
                        
                except KeyboardInterrupt:
                    pbar.close()
                    print("\n🛑 Monitoring stopped. Embeddings continue processing in background.")
                    print("💡 Use 'agentic-rag embed --stats' to check progress later.")
                
            except ImportError as e:
                print(f"❌ Missing required dependency: {e}")
                print("💡 Install ollama: pip install ollama")
                print("💡 Install tqdm: pip install tqdm")
        else:
            print("\n💡 Available actions:")
            print("  --continue     Continue embedding generation from saved state")
            print("  --stats        Show detailed embedding statistics")  
            print("  --clear-state  Clear all saved embedding state")
            
    except Exception as e:
        print(f"Error handling embedding command: {e}")

if __name__ == "__main__":
    asyncio.run(main())