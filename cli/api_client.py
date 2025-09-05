"""
API Client for the AgenticRAG CLI
"""

import httpx
import asyncio
from typing import Dict, Any, Optional
from cli.config_manager import CLIConfig

class APIClient:
    """HTTP API client for communicating with backend services"""
    
    def __init__(self, config: CLIConfig):
        self.config = config
        self.http_client: Optional[httpx.AsyncClient] = None
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=300.0)  # 5 minutes for large ingestion tasks
        return self.http_client
    
    async def close(self):
        """Cleanup resources"""
        if self.http_client:
            await self.http_client.aclose()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all services"""
        client = await self._get_http_client()
        services = {
            'orchestrator': f"{self.config.orchestrator_url}/health",
            'model_gateway': f"{self.config.model_gateway_url}/health"
        }
        
        results = {}
        for service, url in services.items():
            try:
                response = await client.get(url, timeout=5.0)
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
    
    async def submit_task(self, task_type: str, description: str, 
                         parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a task to the orchestrator"""
        client = await self._get_http_client()
        
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
            raise Exception(f"Error submitting task: {e}")