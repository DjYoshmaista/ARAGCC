"""
Advanced Model Gateway for AgenticRAG System

Provides high-performance interface to Ollama models with load balancing,
rate limiting, health monitoring, and comprehensive error handling.
"""

import asyncio
import json
from typing import Dict, List, Optional, AsyncGenerator, Any, Union
from dataclasses import dataclass, asdict, field
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import uvicorn
import logging
import os
from pydantic import BaseModel, Field, validator
from datetime import datetime, timezone, timedelta
import time
from contextlib import asynccontextmanager

logger = logging.getLogger("model_gateway")

# Simple logging settup for model gateway -- no YAML config load here for simplicity
log_file = "logs/model-gateway-app.log"
log_dir = os.path.dirname(log_file)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger.info("Model Gateway logging initialized.")

# Load balancing support for multiple Ollama instances
import random
from typing import List

class OllamaInstancePool:
    """Pool of Ollama instance URLs for load balancing"""
    
    def __init__(self):
        self.instances = ["http://localhost:11434"]  # Default instance
        self.current_index = 0
        self.lock = asyncio.Lock()
    
    async def add_instance(self, url: str):
        """Add a new Ollama instance URL to the pool"""
        async with self.lock:
            if url not in self.instances:
                self.instances.append(url)
                logging.info(f"Added Ollama instance: {url}")
    
    async def get_next_instance(self) -> str:
        """Get next instance URL using round-robin"""
        async with self.lock:
            if not self.instances:
                return "http://localhost:11434"  # Fallback
            
            url = self.instances[self.current_index % len(self.instances)]
            self.current_index += 1
            return url
    
    async def get_random_instance(self) -> str:
        """Get random instance URL for better load distribution"""
        async with self.lock:
            if not self.instances:
                return "http://localhost:11434"  # Fallback
            return random.choice(self.instances)
    
    def get_instance_count(self) -> int:
        """Get number of instances in pool"""
        return len(self.instances)

# Global instance pool
ollama_instance_pool = OllamaInstancePool()

@dataclass
class ModelInfo:
    name: str
    size: str
    format: str
    family: str
    parameters: str
    quantization: str
    loaded: bool = False

class InferenceRequest(BaseModel):
    """Request model for text generation"""
    model: str = Field(..., description="Model name to use for inference")
    prompt: str = Field(..., min_length=1, description="Input prompt for generation")
    stream: bool = Field(False, description="Whether to stream response")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Model parameters")
    
    @validator('model')
    def validate_model(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError('Model name must be a non-empty string')
        return v.strip()
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError('Prompt must be a non-empty string')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "model": "qwen3:8b",
                "prompt": "What is artificial intelligence?",
                "stream": False,
                "parameters": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 2048
                }
            }
        }

class InferenceResponse(BaseModel):
    """Response model for text generation"""
    response: str = Field(..., description="Generated text response")
    done: bool = Field(True, description="Whether generation is complete")
    context: Optional[List[int]] = Field(None, description="Context tokens")
    total_duration: Optional[int] = Field(None, description="Total processing time in nanoseconds")
    load_duration: Optional[int] = Field(None, description="Model loading time in nanoseconds")
    prompt_eval_count: Optional[int] = Field(None, description="Number of prompt tokens")
    prompt_eval_duration: Optional[int] = Field(None, description="Prompt evaluation time")
    eval_count: Optional[int] = Field(None, description="Number of generated tokens")
    eval_duration: Optional[int] = Field(None, description="Generation time")
    model_info: Optional[Dict[str, Any]] = Field(None, description="Model metadata")

class EmbeddingRequest(BaseModel):
    """Request model for embedding generation"""
    model: str = Field(default="dengcao/Qwen3-Embedding-0.6B:Q8_0", description="Embedding model name")
    prompt: str = Field(..., min_length=1, description="Text to embed")
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError('Prompt must be a non-empty string')
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "model": "dengcao/Qwen3-Embedding-0.6B:Q8_0",
                "prompt": "The quick brown fox jumps over the lazy dog."
            }
        }

class EmbeddingResponse(BaseModel):
    """Response model for embedding generation"""
    embedding: List[float] = Field(..., description="Generated embedding vector")
    dimensions: int = Field(..., description="Embedding dimensions")
    total_duration: Optional[int] = Field(None, description="Total processing time")
    load_duration: Optional[int] = Field(None, description="Model loading time")
    model_info: Optional[Dict[str, Any]] = Field(None, description="Model metadata")

class ModelGateway:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.loaded_models: Dict[str, ModelInfo] = {}
        self.client = httpx.AsyncClient(timeout=300.0)  # 5 minutes for large model operations
        
    async def initialize(self):
        """Initialize the gateway and discover available models"""
        try:
            await self.discover_models()
        except Exception as e:
            print(f"Warning: Could not connect to Ollama: {e}")
    
    async def discover_models(self) -> List[ModelInfo]:
        """Discover available models from Ollama"""
        try:
            response = await self.client.get(f"{self.ollama_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            
            models = []
            for model_data in data.get("models", []):
                model_info = ModelInfo(
                    name=model_data["name"],
                    size=model_data.get("size", "unknown"),
                    format=model_data.get("format", "unknown"),
                    family=model_data.get("details", {}).get("family", "unknown"),
                    parameters=model_data.get("details", {}).get("parameter_size", "unknown"),
                    quantization=model_data.get("details", {}).get("quantization_level", "unknown")
                )
                models.append(model_info)
                self.loaded_models[model_info.name] = model_info
                
            return models
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Cannot connect to Ollama: {e}")
    
    async def load_model(self, model_name: str) -> bool:
        """Preload a model for faster inference"""
        try:
            # Send a simple request to load the model
            response = await self.client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "",
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                if model_name in self.loaded_models:
                    self.loaded_models[model_name].loaded = True
                return True
            return False
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return False
    
    async def unload_model(self, model_name: str) -> bool:
        """Unload a model to free resources"""
        try:
            # Ollama doesn't have explicit unload, but we can track loading state
            if model_name in self.loaded_models:
                self.loaded_models[model_name].loaded = False
            return True
        except Exception as e:
            print(f"Error unloading model {model_name}: {e}")
            return False
    
    async def generate_response(
        self, 
        model_name: str, 
        prompt: str, 
        stream: bool = False,
        parameters: Dict = None
    ) -> AsyncGenerator[str, None]:
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
    
    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model"""
        return self.loaded_models.get(model_name)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the Ollama service"""
        try:
            response = await self.client.get(f"{self.ollama_url}/api/tags")
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "ollama_url": self.ollama_url,
                "models_loaded": len([m for m in self.loaded_models.values() if m.loaded])
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "ollama_url": self.ollama_url
            }
    
    async def generate_embedding(self, model_name: str, prompt: str) -> List[float]:
        """Generate embedding for text using load-balanced Ollama instances with GPU support"""
        try:
            # Get instance URL using load balancing
            instance_url = await ollama_instance_pool.get_random_instance()
            
            # Try /api/embed first (as requested)
            try:
                async with self.client.stream(
                    "POST",
                    f"{instance_url}/api/embed",
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "options": {
                            "num_thread": 4,  # Reduced to allow more concurrent requests
                            "num_gpu": 1
                        }
                    },
                    timeout=60.0
                ) as response:
                    response.raise_for_status()
                    content = await response.aread()
                    data = json.loads(content)
                    
                    if "embedding" in data:
                        embedding = data["embedding"]
                        logger.debug(f"Generated embedding with {len(embedding)} dimensions using GPU on {instance_url} via /api/embed")
                        return embedding
                    else:
                        raise Exception("No embedding found in /api/embed response")
            
            except Exception as embed_error:
                logger.warning(f"/api/embed failed for {instance_url}: {embed_error}")
                
                # Fallback to /api/embeddings
                async with self.client.stream(
                    "POST",
                    f"{instance_url}/api/embeddings",
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "options": {
                            "num_thread": 4,  # Reduced to allow more concurrent requests
                            "num_gpu": 1
                        }
                    },
                    timeout=60.0
                ) as response:
                    response.raise_for_status()
                    content = await response.aread()
                    data = json.loads(content)
                    
                    # Handle different response structures
                    embedding = None
                    if "embedding" in data:
                        embedding = data["embedding"]
                    elif "embeddings" in data:
                        embeddings_array = data["embeddings"]
                        if isinstance(embeddings_array, list) and len(embeddings_array) > 0:
                            embedding = embeddings_array[0]
                    
                    if embedding:
                        logger.debug(f"Generated embedding with {len(embedding)} dimensions using GPU on {instance_url} via /api/embeddings fallback")
                        return embedding
                    else:
                        raise Exception("No embedding found in /api/embeddings response")
        
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            raise HTTPException(status_code=500, detail=f"Embedding generation error: {e}")

# FastAPI application
app = FastAPI(title="Model Gateway", version="1.0.0")
gateway = ModelGateway()

@app.on_event("startup")
async def startup_event():
    logger.info("Model Gateway service is starting up...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags", timeout=10.0)
            response.raise_for_status()
            logger.info("Successfully connected to Ollama API.")
    except Exception as e:
        logger.error(f"Failed to connect to Ollama API on startup: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.debug("Health check endpoint called.")
    try:
        async with httpx.AsyncClient() as client:
            # Quick check if Ollama is reachable
            response = await client.get("http://localhost:11434/", timeout=5.0)
            if response.status_code == 200:
                logger.info("Health check: Model Gateway and Ollama connection OK.")
                return {"status": "healthy", "ollama_connected": True, "timestamp": datetime.now(timezone.utc).isoformat()}
            else:
                logger.warning(f"Health check: Model Gateway OK, but Ollama responded with status {response.status_code}.")
                return {"status": "degraded", "ollama_connected": False, "ollama_status": response.status_code, "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        logger.error(f"Health check failed: Error connecting to Ollama: {e}")
        return {"status": "unhealthy", "ollama_connected": False, "error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/models")
async def list_models():
    return {"models": [asdict(model) for model in gateway.loaded_models.values()]}

@app.post("/models/{model_name}/load")
async def load_model(model_name: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(gateway.load_model, model_name)
    return {"message": f"Loading model {model_name}"}

@app.post("/models/{model_name}/unload")
async def unload_model(model_name: str):
    success = await gateway.unload_model(model_name)
    return {"success": success, "message": f"Model {model_name} unload requested"}

@app.post("/generate")
async def generate_text(request: InferenceRequest):
    logger.info(f"Received generation request for model '{request.model}'")
    logger.debug(f"Request details: prompt='{request.prompt[:50]}...', stream={request.stream}, params={request.parameters}")
    try:
        async with httpx.AsyncClient() as client:
            # Forward the request to Ollama
            ollama_request = {
                "model": request.model,
                "prompt": request.prompt,
                "stream": request.stream,
                "options": request.parameters # Pass parameters as options to Ollama
            }
            logger.debug(f"Forwarding request to Ollama: {ollama_request}")
            response = await client.post(
                "http://localhost:11434/api/generate", # Standard Ollama endpoint
                json=ollama_request,
                timeout=300.0 # Increase timeout for generation
            )
            logger.debug(f"Received response from Ollama: Status {response.status_code}")
            response.raise_for_status()

            if request.stream:
                # Handle streaming response if needed (your existing logic)
                # ...
                pass
            else:
                # Return the full response
                ollama_response = response.json()
                logger.info(f"Generation completed successfully for model '{request.model}'.")
                # Return the response in the format expected by the Orchestrator
                return {"response": ollama_response.get("response", "")}

    except httpx.RequestError as e:
        logger.error(f"Network error calling Ollama: {e}")
        raise HTTPException(status_code=502, detail=f"Network error calling Ollama: {e}")
    except httpx.HTTPStatusError as e:
        logger.error(f"Ollama returned error {e.response.status_code}: {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Ollama error: {e.response.text}")
    except Exception as e:
        logger.error(f"Unexpected error during generation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/models/{model_name}/info")
async def get_model_info(model_name: str):
    info = await gateway.get_model_info(model_name)
    if not info:
        raise HTTPException(status_code=404, detail="Model not found")
    return asdict(info)

@app.post("/embed")
async def generate_embedding(request: EmbeddingRequest):
    """Generate embedding for text using load-balanced Ollama instances"""
    logger.info(f"Received embedding request for model '{request.model}'")
    logger.debug(f"Request details: prompt='{request.prompt[:50]}...'")
    try:
        embedding = await gateway.generate_embedding(request.model, request.prompt)
        logger.info(f"Embedding generation completed successfully for model '{request.model}'")
        return {"embedding": embedding}
    except httpx.RequestError as e:
        logger.error(f"Network error calling Ollama: {e}")
        raise HTTPException(status_code=502, detail=f"Network error calling Ollama: {e}")
    except httpx.HTTPStatusError as e:
        logger.error(f"Ollama returned error {e.response.status_code}: {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Ollama error: {e.response.text}")
    except Exception as e:
        logger.error(f"Unexpected error during embedding generation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.post("/instances/register")
async def register_ollama_instance(request: dict):
    """Register a new Ollama instance for load balancing"""
    instance_url = request.get("url")
    if not instance_url:
        raise HTTPException(status_code=400, detail="Missing 'url' parameter")
    
    try:
        # Validate instance is reachable
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{instance_url}/api/tags", timeout=5.0)
            response.raise_for_status()
        
        # Add to pool
        await ollama_instance_pool.add_instance(instance_url)
        
        logger.info(f"Successfully registered Ollama instance: {instance_url}")
        return {
            "status": "success",
            "message": f"Registered instance: {instance_url}",
            "total_instances": ollama_instance_pool.get_instance_count()
        }
        
    except Exception as e:
        logger.error(f"Failed to register instance {instance_url}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to register instance: {str(e)}")

@app.get("/instances/stats")
async def get_instance_stats():
    """Get statistics about registered Ollama instances"""
    return {
        "total_instances": ollama_instance_pool.get_instance_count(),
        "instances": ollama_instance_pool.instances,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

if __name__ == "__main__":
    logger.info("Starting Model Gateway application...")
    uvicorn.run(app, host="0.0.0.0", port=8070)