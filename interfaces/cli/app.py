"""
Refactored CLI Interface using new modular architecture.
"""

import asyncio
import sys
import os
import json
import httpx
from pathlib import Path
from typing import Dict, Any, List
import argparse
import logging

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.config import get_config
from interfaces.cli.progress_display import DualProgressDisplay, install_tqdm_if_needed

logger = logging.getLogger("agentic_rag_cli")

class AgenticRAGCLI:
    """Refactored CLI using modular architecture"""
    
    def __init__(self):
        self.config = get_config()
        self.orchestrator_url = f"http://{self.config['services']['orchestrator']['host']}:{self.config['services']['orchestrator']['port']}"
        self.model_gateway_url = f"http://{self.config['services']['model_gateway']['host']}:{self.config['services']['model_gateway']['port']}"
        self.client = httpx.AsyncClient(timeout=300.0)
    
    async def close(self):
        """Cleanup resources"""
        await self.client.aclose()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all services"""
        services = {
            'orchestrator': f"{self.orchestrator_url}/health",
            'model_gateway': f"{self.model_gateway_url}/health"
        }
        
        results = {}
        for service, url in services.items():
            try:
                response = await self.client.get(url, timeout=5.0)
                results[service] = {
                    'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                    'status_code': response.status_code
                }
            except Exception as e:
                results[service] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return results
    
    async def submit_task(self, task_type: str, description: str, parameters: Dict[str, Any]) -> str:
        """Submit a task to the orchestrator"""
        try:
            response = await self.client.post(
                f"{self.orchestrator_url}/tasks",
                json={
                    'task_type': task_type,
                    'description': description,
                    'parameters': parameters
                }
            )
            response.raise_for_status()
            data = response.json()
            return data['task_id']
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            raise
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status"""
        try:
            response = await self.client.get(f"{self.orchestrator_url}/tasks/{task_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get task status: {e}")
            raise
    
    async def wait_for_task(self, task_id: str, show_progress: bool = False) -> Dict[str, Any]:
        """Wait for task completion with optional progress display"""
        progress_display = None
        
        if show_progress:
            # Check if tqdm is available
            install_tqdm_if_needed()
            
            # Initialize progress display
            progress_display = DualProgressDisplay(task_id)
            progress_display.start()
        
        try:
            while True:
                # Get task status
                status = await self.get_task_status(task_id)
                
                # Update progress display if enabled
                if progress_display:
                    try:
                        # Get progress data from orchestrator
                        progress_response = await self.client.get(
                            f"{self.orchestrator_url}/tasks/{task_id}/progress",
                            timeout=5.0
                        )
                        
                        if progress_response.status_code == 200:
                            progress_data = progress_response.json()
                            progress_stats = progress_data.get('progress', {})
                            progress_display.update(progress_stats)
                        
                    except Exception as e:
                        logger.debug(f"Progress update failed: {e}")
                
                # Check if task completed
                if status['status'] in ['completed', 'failed']:
                    if progress_display:
                        # Show final summary
                        try:
                            progress_response = await self.client.get(
                                f"{self.orchestrator_url}/tasks/{task_id}/progress",
                                timeout=5.0
                            )
                            if progress_response.status_code == 200:
                                progress_data = progress_response.json()
                                final_stats = progress_data.get('progress', {})
                                progress_display.show_final_summary(final_stats)
                        except:
                            pass
                    
                    return status
                
                await asyncio.sleep(2)  # Update every 2 seconds
        
        finally:
            if progress_display:
                progress_display.stop()

async def handle_status_command(args, cli: AgenticRAGCLI):
    """Handle status command"""
    print("Checking system status...")
    health = await cli.health_check()
    
    for service, status in health.items():
        status_icon = "✅" if status.get('status') == 'healthy' else "❌"
        print(f"{status_icon} {service}: {status.get('status', 'unknown')}")

async def handle_query_command(args, cli: AgenticRAGCLI):
    """Handle query command"""
    try:
        task_id = await cli.submit_task(
            "llm_inference",
            f"Generate response for: {args.prompt}",
            {
                "model": args.model or "qwen3:8b",
                "prompt": args.prompt
            }
        )
        
        print(f"Submitted query task: {task_id}")
        print("Waiting for response...")
        
        result = await cli.wait_for_task(task_id, show_progress=False)
        
        if result['status'] == 'completed':
            print(f"\nResponse: {result['result']}")
        else:
            print(f"Query failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"Error: {e}")

async def handle_ingest_command(args, cli: AgenticRAGCLI):
    """Handle ingestion command"""
    try:
        task_id = await cli.submit_task(
            "folder_ingestion",
            f"Ingest files: {', '.join(args.paths)}",
            {
                "paths": args.paths,
                "recursive": args.recursive,
                "embedding_model": args.embedding_model or cli.config['models']['embedding_model']
            }
        )
        
        print(f"Submitted ingestion task: {task_id}")
        print("Processing files with progress tracking...")
        print("")
        
        result = await cli.wait_for_task(task_id, show_progress=True)
        
        if result['status'] == 'completed':
            try:
                stats = json.loads(result['result'])
                print(f"\nIngestion completed:")
                print(f"  Files processed: {stats.get('processed_files', 0)}")
                print(f"  Files failed: {stats.get('failed_files', 0)}")
                print(f"  Total chunks: {stats.get('total_chunks', 0)}")
            except:
                print(f"Ingestion completed: {result['result']}")
        else:
            print(f"Ingestion failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"Error: {e}")

def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        prog="agentic-rag",
        description="AgenticRAG CLI (Refactored)"
    )
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check system status')
    
    # Query command  
    query_parser = subparsers.add_parser('query', help='Submit a query')
    query_parser.add_argument('prompt', help='Query prompt')
    query_parser.add_argument('--model', help='Model to use')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documents')
    ingest_parser.add_argument('paths', nargs='+', help='Paths to ingest')
    ingest_parser.add_argument('--recursive', '-r', action='store_true', help='Process recursively')
    ingest_parser.add_argument('--embedding-model', help='Embedding model to use')
    
    return parser

async def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    cli = AgenticRAGCLI()
    
    try:
        if not args.command:
            parser.print_help()
            return
        
        if args.command == 'status':
            await handle_status_command(args, cli)
        elif args.command == 'query':
            await handle_query_command(args, cli)
        elif args.command == 'ingest':
            await handle_ingest_command(args, cli)
        else:
            print(f"Command '{args.command}' not implemented in this refactored version")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
    finally:
        await cli.close()

if __name__ == "__main__":
    asyncio.run(main())