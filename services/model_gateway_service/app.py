"""
Refactored Model Gateway Service using new modular architecture.
"""

import asyncio
import json
from typing import Dict, List, Optional, AsyncGenerator, Any
from dataclasses import dataclass, asdict
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import httpx
import uvicorn
import logging
import os
import sys
from pydantic import BaseModel
from datetime import datetime, timezone
import time

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.config import get_config
from core.embedding import LoadBalancingStrategy, EmbeddingLoadBalancer

logger = logging.getLogger("model_gateway_service")

class InferenceRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = False
    parameters: Dict[str, Any] = {}

class EmbeddingRequest(BaseModel):
    model: str = "granite-embedding:latest"
    prompt: str

class ModelGateway:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.client = httpx.AsyncClient(timeout=300.0)
        
        # Initialize load balancer
        self.load_balancer = EmbeddingLoadBalancer(LoadBalancingStrategy.ROUND_ROBIN)
        self.load_balancer.add_instance(ollama_url)
        
    async def generate_response(self, model_name: str, prompt: str, 
                               stream: bool = False, parameters: Dict = None) -> AsyncGenerator[str, None]:
        """Generate response from model"""
        if parameters is None:
            parameters = {}
            
        request_data = {
            "model": model_name,
            "prompt": prompt,
            "stream": stream,
            **parameters
        }
        
        try:
            async with self.client.stream(
                "POST",
                f"{self.ollama_url}/api/generate",
                json=request_data
            ) as response:
                response.raise_for_status()
                
                if stream:
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if "response" in data:
                                    yield data["response"]
                                if data.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue
                else:
                    content = await response.aread()
                    data = json.loads(content)
                    yield data.get("response", "")
                    
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation error: {e}")
    
    async def generate_embedding(self, model_name: str, prompt: str) -> List[float]:
        """Generate embedding for text using load-balanced instances"""
        try:
            # Get instance URL using load balancing
            instance_url = self.load_balancer.get_next_instance() or self.ollama_url
            
            start_time = time.time()
            
            # Try /api/embed first
            try:
                async with self.client.stream(
                    "POST",
                    f"{instance_url}/api/embed",
                    json={
                        "model": model_name,
                        "prompt": prompt
                    },
                    timeout=60.0
                ) as response:
                    response.raise_for_status()
                    content = await response.aread()
                    data = json.loads(content)
                    
                    if "embedding" in data:
                        embedding = data["embedding"]
                        
                        # Record metrics
                        processing_time = time.time() - start_time
                        self.load_balancer.record_request(instance_url, True, processing_time)
                        
                        logger.debug(f"Generated embedding with {len(embedding)} dimensions")
                        return embedding
                    else:
                        raise Exception("No embedding found in response")
            
            except Exception as embed_error:
                logger.warning(f"/api/embed failed: {embed_error}")
                
                # Fallback to /api/embeddings
                async with self.client.stream(
                    "POST", 
                    f"{instance_url}/api/embeddings",
                    json={
                        "model": model_name,
                        "prompt": prompt
                    },
                    timeout=60.0
                ) as response:
                    response.raise_for_status()
                    content = await response.aread()
                    data = json.loads(content)
                    
                    embedding = None
                    if "embedding" in data:
                        embedding = data["embedding"]
                    elif "embeddings" in data:
                        embeddings_array = data["embeddings"]
                        if isinstance(embeddings_array, list) and len(embeddings_array) > 0:
                            embedding = embeddings_array[0]
                    
                    if embedding:
                        processing_time = time.time() - start_time
                        self.load_balancer.record_request(instance_url, True, processing_time)
                        return embedding
                    else:
                        raise Exception("No embedding found in /api/embeddings response")
        
        except Exception as e:
            # Record failure
            processing_time = time.time() - start_time
            self.load_balancer.record_request(instance_url, False, processing_time)
            logger.error(f"Embedding generation error: {e}")
            raise HTTPException(status_code=500, detail=f"Embedding generation error: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the service"""
        try:
            response = await self.client.get(f"{self.ollama_url}/api/tags")
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "ollama_url": self.ollama_url
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "ollama_url": self.ollama_url
            }

# FastAPI application
app = FastAPI(title="Model Gateway (Refactored)", version="2.0.0")

# Load configuration
config = get_config()
ollama_url = config['services']['model_gateway']['ollama_url']
gateway = ModelGateway(ollama_url)

@app.on_event("startup")
async def startup_event():
    logger.info("Model Gateway service starting with new architecture...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{ollama_url}/api/tags", timeout=10.0)
            response.raise_for_status()
            logger.info("Successfully connected to Ollama API.")
    except Exception as e:
        logger.error(f"Failed to connect to Ollama API on startup: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{ollama_url}/", timeout=5.0)
            if response.status_code == 200:
                return {"status": "healthy", "ollama_connected": True, 
                       "timestamp": datetime.now(timezone.utc).isoformat()}
            else:
                return {"status": "degraded", "ollama_connected": False, 
                       "ollama_status": response.status_code,
                       "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        return {"status": "unhealthy", "ollama_connected": False, 
               "error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}

@app.post("/generate")
async def generate_text(request: InferenceRequest):
    """Generate text using LLM"""
    logger.info(f"Received generation request for model '{request.model}'")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": request.model,
                    "prompt": request.prompt,
                    "stream": request.stream,
                    "options": request.parameters
                },
                timeout=300.0
            )
            response.raise_for_status()
            
            if not request.stream:
                ollama_response = response.json()
                return {"response": ollama_response.get("response", "")}
            else:
                # Handle streaming if needed
                return {"response": "Streaming not implemented in this version"}

    except httpx.RequestError as e:
        logger.error(f"Network error calling Ollama: {e}")
        raise HTTPException(status_code=502, detail=f"Network error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

@app.post("/embed")
async def generate_embedding(request: EmbeddingRequest):
    """Generate embedding for text"""
    logger.info(f"Received embedding request for model '{request.model}'")
    try:
        embedding = await gateway.generate_embedding(request.model, request.prompt)
        return {"embedding": embedding}
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def setup_logging():
    """Configure logging to write to logs directory"""
    # Create logs directory if it doesn't exist
    from pathlib import Path
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging with both file and console handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "model-gateway-app.log"),
            logging.StreamHandler()
        ]
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8070, help="Port to bind to")
    args = parser.parse_args()
    
    setup_logging()
    logger.info("Starting Model Gateway Service (Refactored)...")
    logger.info(f"Logging configured - writing to logs/model-gateway-app.log")
    uvicorn.run(app, host=args.host, port=args.port)