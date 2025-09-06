#!/usr/bin/env python3
"""
AgenticRAG CLI - Advanced Multi-Agent System Command Line Interface

This CLI provides comprehensive interaction with the AgenticRAG multi-agent system,
supporting file processing, RAG database operations, model configuration, and more.

Features:
- Task orchestration and monitoring
- Document ingestion with progress tracking
- Model configuration and management
- Health monitoring and diagnostics
- RAG database operations

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
from typing import Dict, List, Optional, Any, Union, Tuple
import httpx
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
import subprocess
import tempfile
import threading
import time
from tqdm import tqdm
import warnings
from contextlib import asynccontextmanager

# CLI Configuration
CLI_VERSION = "1.0.0"
CLI_NAME = "agentic-rag"
CONFIG_DIR = Path.home() / ".config" / "agentic-rag"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
LOG_DIR = Path.home() / ".local" / "share" / "agentic-rag" / "logs"

@dataclass
class CLIConfig:
    """
    CLI configuration structure with comprehensive validation.
    
    Manages all configuration settings for the AgenticRAG CLI including
    service endpoints, database settings, RAG parameters, and user preferences.
    """
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
    models: Optional[Dict[str, Dict]] = None
    
    # Plugin settings
    plugin_dir: Optional[Path] = None
    enabled_plugins: Optional[List[str]] = None
    
    # Interface settings
    interactive_mode: bool = True
    color_output: bool = True
    verbose: bool = False
    
    # Performance settings
    timeout_seconds: int = 300
    max_retries: int = 3
    retry_delay: float = 1.0
    
    def __post_init__(self):
        """Initialize default values and validate configuration."""
        if self.models is None:
            self.models = {
                "orchestration": {"model": "qwen3:8b", "temperature": 0.3},
                "inference": {"model": "qwen3:8b", "temperature": 0.7},
                "embedding": {"model": "dengcaoQwen3-Embedding-0.6B:Q8_0"},
                "rag": {"model": "qwen3:8b", "temperature": 0.5}
            }
        if self.enabled_plugins is None:
            self.enabled_plugins = []
        if self.plugin_dir is None:
            self.plugin_dir = CONFIG_DIR / "plugins"
            
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        # Validate port numbers
        if not (1 <= self.postgresql_port <= 65535):
            raise ValueError(f"Invalid PostgreSQL port: {self.postgresql_port}")
        
        # Validate RAG settings
        if not (1 <= self.rag_top_k <= 100):
            raise ValueError(f"Invalid rag_top_k: {self.rag_top_k}")
        if not (-10.0 <= self.rag_weight <= 10.0):
            raise ValueError(f"Invalid rag_weight: {self.rag_weight}")
        
        # Validate agent settings
        if not (1 <= self.max_agents <= 50):
            raise ValueError(f"Invalid max_agents: {self.max_agents}")
        if not (0.0 <= self.default_temperature <= 2.0):
            raise ValueError(f"Invalid default_temperature: {self.default_temperature}")
        
        # Validate performance settings
        if not (10 <= self.timeout_seconds <= 3600):
            raise ValueError(f"Invalid timeout_seconds: {self.timeout_seconds}")
        if not (1 <= self.max_retries <= 10):
            raise ValueError(f"Invalid max_retries: {self.max_retries}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        config_dict = asdict(self)
        # Convert Path objects to strings
        if config_dict['plugin_dir']:
            config_dict['plugin_dir'] = str(config_dict['plugin_dir'])
        return config_dict

class CLIManager:
    """Main CLI management class.

    Handles configuration, logging, and HTTP client management.
    """
    
    def __init__(self):
        """Initialize the CLI manager."""
        self.config: CLIConfig = CLIConfig()
        self.http_client: Optional[httpx.AsyncClient] = None
        self.logger = self._setup_logging()
        self._ensure_directories()
        self._load_config()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
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
        """Ensure necessary directories exist."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.config.plugin_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self):
        """Load configuration from file."""
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
        """Save current configuration to file."""
        try:
            with open(CONFIG_FILE, 'w') as f:
                yaml.dump(asdict(self.config), f, default_flow_style=False)
            self.logger.info("Configuration saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            print(f"Error: Could not save config file: {e}")
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=300.0)  # 5 minutes for large ingestion tasks
        return self.http_client
    
    async def close(self):
        """Cleanup resources."""
        if self.http_client:
            await self.http_client.aclose()

class CommandParser:
    """Command line argument parser.

    Handles the creation of the main parser and sub-parsers for all commands.
    """
    
    def __init__(self, cli_manager: CLIManager):
        """Initialize the command parser."""
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
    """HTTP API client for communicating with backend services.

    Handles all HTTP requests to the orchestrator, model gateway, and vector engine.
    """
    
    def __init__(self, cli_manager: CLIManager):
        """Initialize the API client."""
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
                response.raise_for_status()
                results[service] = {
                    'status': 'healthy',
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds()
                }
            except httpx.ConnectError as e:
                results[service] = {'status': 'error', 'error': f"Connection error: {e}"}
            except httpx.Timeout as e:
                results[service] = {'status': 'error', 'error': f"Timeout error: {e}"}
            except httpx.HTTPStatusError as e:
                results[service] = {
                    'status': 'unhealthy',
                    'status_code': e.response.status_code,
                    'error': str(e)
                }
            except Exception as e:
                results[service] = {'status': 'error', 'error': str(e)}
        
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
        except httpx.HTTPStatusError as e:
            self.logger.error(f"Error submitting task: {e.response.text}")
            raise
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
        except httpx.HTTPStatusError as e:
            self.logger.error(f"Error getting task status: {e.response.text}")
            raise
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
        except httpx.HTTPStatusError as e:
            self.logger.error(f"Error generating response: {e.response.text}")
            raise
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise

async def main(argv: Optional[List[str]] = None):
    """Main CLI entry point"""
    # Suppress tqdm warnings that might interfere with display
    warnings.filterwarnings("ignore", category=UserWarning, module="tqdm")
    
    cli_manager = CLIManager()
    parser = CommandParser(cli_manager)
    api_client = APIClient(cli_manager)
    
    try:
        args = parser.parse_args(argv)
        
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
    try:
        task_response = await api_client.submit_task(
            task_type="query",
            description=f"Query: {args.prompt}",
            parameters={
                "prompt": args.prompt,
                "model": args.model or cli_manager.config.models["inference"]["model"],
                "temperature": args.temperature or cli_manager.config.default_temperature,
                "rag_weight": args.rag_weight or cli_manager.config.rag_weight,
                "top_k": args.top_k or cli_manager.config.rag_top_k,
                "context_window": args.context_window or cli_manager.config.default_context_window,
            }
        )
        
        task_id = task_response["task_id"]
        print(f"Submitted query task with ID: {task_id}")
        
        # Wait for task completion
        while True:
            task_status = await api_client.get_task_status(task_id)
            if task_status["status"] in ["completed", "failed"]:
                break
            await asyncio.sleep(1)
        
        if task_status["status"] == "completed":
            print("Response:")
            print(task_status["result"])
        else:
            print(f"Query failed: {task_status['error']}")
            
    except Exception as e:
        print(f"Error handling query: {e}")

async def handle_config_command(args, cli_manager):
    """Handle configuration command"""
    if args.show:
        print(yaml.dump(asdict(cli_manager.config), default_flow_style=False))
    
    if args.set:
        for setting in args.set:
            key, value = setting.split('=', 1)
            if hasattr(cli_manager.config, key):
                # Convert value to the correct type
                try:
                    field_type = type(getattr(cli_manager.config, key))
                    setattr(cli_manager.config, key, field_type(value))
                except (ValueError, TypeError):
                    print(f"Invalid value for {key}: {value}")
            else:
                print(f"Unknown config key: {key}")
        cli_manager._save_config()
        print("Configuration updated.")

    if args.reset:
        cli_manager.config = CLIConfig()
        cli_manager._save_config()
        print("Configuration reset to default.")

    if args.validate:
        # Basic validation for now
        print("Configuration validation is not fully implemented yet.")
        print("A basic check confirms that the orchestrator is reachable.")
        api_client = APIClient(cli_manager)
        health = await api_client.health_check()
        if health.get('orchestrator', {}).get('status') == 'healthy':
            print("✅ Orchestrator is reachable.")
        else:
            print("❌ Orchestrator is not reachable.")

async def handle_status_command(args, cli_manager, api_client):
    """Handle status command"""
    if args.health or not (args.services or args.metrics):
        print("Checking system health...")
        health = await api_client.health_check()
        for service, status in health.items():
            status_icon = "✅" if status.get('status') == 'healthy' else "❌"
            print(f"{status_icon} {service}: {status.get('status', 'unknown')}")

    if args.services:
        print("\nChecking service status...")
        health = await api_client.health_check()
        for service, status in health.items():
            print(f"- {service}:")
            for key, value in status.items():
                print(f"  {key}: {value}")

    if args.metrics:
        print("\nSystem metrics are not fully implemented yet.")
        print("Displaying basic health check metrics:")
        health = await api_client.health_check()
        for service, status in health.items():
            if status.get('status') == 'healthy':
                print(f"- {service} response time: {status.get('response_time')}s")

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

    elif args.list:
        try:
            client = await cli_manager._get_http_client()
            response = await client.get(f"{cli_manager.config.orchestrator_url}/rag/documents")
            response.raise_for_status()
            documents = response.json()
            print("RAG Documents:")
            for doc in documents:
                print(f"- {doc['id']}: {doc['file_name']}")
        except Exception as e:
            print(f"Error listing RAG documents: {e}")

    elif args.delete:
        try:
            client = await cli_manager._get_http_client()
            response = await client.delete(f"{cli_manager.config.orchestrator_url}/rag/documents/{args.delete}")
            response.raise_for_status()
            print(f"Document {args.delete} deleted successfully.")
        except Exception as e:
            print(f"Error deleting RAG document: {e}")
    
    else:
        print("RAG command requires either --ingest, --query, --list, or --delete parameter.")
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
        
        # Initialize dual progress bars
        with tqdm(total=100, desc="📝 PostgreSQL", unit="files", position=0) as postgres_pbar, \
             tqdm(total=100, desc="🔮 Vector DB", unit="emb", position=1) as vector_pbar:
            
            while True:
                try:
                    client = await cli_manager._get_http_client()
                    progress_response = await client.get(f"{orchestrator_url}/tasks/{task_id}/progress")
                    progress_response.raise_for_status()
                    progress_data = progress_response.json()

                    if "error" in progress_data:
                        # Fallback to task status
                        task_status = await api_client.get_task_status(task_id)
                        if task_status["status"] in ["completed", "failed"]:
                            break
                        await asyncio.sleep(5)
                        continue

                    progress = progress_data.get("progress", {})
                    postgres_stats = progress.get("postgres", {})
                    vector_stats = progress.get("vector", {})

                    # Update PostgreSQL progress bar
                    if postgres_stats:
                        total_files = postgres_stats.get("total_files", 100)
                        processed_files = postgres_stats.get("files_processed", 0)
                        postgres_pbar.total = total_files
                        postgres_pbar.n = processed_files
                        postgres_pbar.set_postfix({
                            "docs": postgres_stats.get("documents_stored", 0),
                            "chunks": postgres_stats.get("chunks_created", 0),
                            "rate": f"{postgres_stats.get('files_per_second', 0):.1f}/s"
                        })
                        postgres_pbar.refresh()

                    # Update Vector DB progress bar
                    if vector_stats:
                        total_embeddings = postgres_stats.get("chunks_created", 100)
                        processed_embeddings = vector_stats.get("embeddings_stored", 0)
                        vector_pbar.total = total_embeddings
                        vector_pbar.n = processed_embeddings
                        vector_pbar.set_postfix({
                            "generated": vector_stats.get("embeddings_generated", 0),
                            "queue": vector_stats.get("queue_size", 0),
                            "rate": f"{vector_stats.get('embeddings_per_second', 0):.1f}/s"
                        })
                        vector_pbar.refresh()

                    # Check task completion
                    task_status = await api_client.get_task_status(task_id)
                    if task_status["status"] in ["completed", "failed"]:
                        postgres_pbar.n = postgres_pbar.total
                        vector_pbar.n = vector_pbar.total
                        postgres_pbar.refresh()
                        vector_pbar.refresh()
                        print(f"\n✅ Task {task_status['status']}!")
                        break

                    await asyncio.sleep(2)

                except Exception as e:
                    await asyncio.sleep(5)

    except Exception as e:
        print(f"Error ingesting folders/files: {e}")


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
        
        if getattr(args, 'continue', False):
            print("Continuing embedding generation is not yet implemented.")
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
            
    except Exception as e:
        print(f"Error handling embedding command: {e}")

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))