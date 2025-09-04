#!/usr/bin/env python3
"""Debug script to identify and fix embedding issues"""

import asyncio
import httpx
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_embedding_endpoints():
    """Test both embedding endpoints with different models"""
    
    base_url = "http://localhost:11434"
    
    # Test models available
    models_to_test = ["nomic-embed-text", "granite-embedding"]
    endpoints_to_test = ["/api/embed", "/api/embeddings"]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # First check available models
        logger.info("=== Checking available models ===")
        try:
            response = await client.get(f"{base_url}/api/tags")
            models_data = response.json()
            available_models = [model['name'] for model in models_data['models']]
            logger.info(f"Available models: {available_models}")
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return
        
        # Test each model with each endpoint
        for model in models_to_test:
            if not any(model in available_model for available_model in available_models):
                logger.warning(f"Model {model} not found in available models")
                continue
                
            logger.info(f"\n=== Testing model: {model} ===")
            
            for endpoint in endpoints_to_test:
                logger.info(f"Testing {endpoint} with {model}")
                
                try:
                    payload = {
                        "model": model,
                        "prompt": "test text for embedding"
                    }
                    
                    response = await client.post(f"{base_url}{endpoint}", json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        logger.info(f"✅ {endpoint} SUCCESS")
                        
                        # Check response structure
                        if endpoint == "/api/embed":
                            embedding = data.get("embedding", [])
                            if embedding:
                                logger.info(f"   Embedding length: {len(embedding)}")
                            else:
                                logger.warning(f"   No 'embedding' field found. Keys: {data.keys()}")
                                logger.info(f"   Full response: {json.dumps(data, indent=2)[:200]}...")
                        
                        elif endpoint == "/api/embeddings":
                            # Check both possible response structures
                            embedding = data.get("embedding")
                            if embedding:
                                logger.info(f"   Embedding length: {len(embedding)}")
                            else:
                                embeddings = data.get("embeddings", [])
                                if embeddings:
                                    logger.info(f"   Embeddings array length: {len(embeddings)}")
                                    if embeddings:
                                        logger.info(f"   First embedding length: {len(embeddings[0])}")
                                else:
                                    logger.warning(f"   No embedding found. Keys: {data.keys()}")
                                    logger.info(f"   Full response: {json.dumps(data, indent=2)[:200]}...")
                    else:
                        logger.error(f"❌ {endpoint} FAILED: HTTP {response.status_code}")
                        logger.error(f"   Response: {response.text[:200]}")
                        
                except Exception as e:
                    logger.error(f"❌ {endpoint} ERROR: {e}")

async def test_with_different_models():
    """Test with different embedding models that are definitely available"""
    base_url = "http://localhost:11434"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Get available models first
        try:
            response = await client.get(f"{base_url}/api/tags")
            models_data = response.json()
            
            # Find embedding models
            embedding_models = []
            for model in models_data['models']:
                name = model['name']
                # Look for common embedding model names
                if any(keyword in name.lower() for keyword in ['embed', 'nomic', 'granite']):
                    embedding_models.append(name)
            
            if not embedding_models:
                logger.warning("No obvious embedding models found, trying first few models")
                embedding_models = [model['name'] for model in models_data['models'][:3]]
            
            logger.info(f"Testing embedding models: {embedding_models}")
            
            for model in embedding_models:
                logger.info(f"\n=== Testing model: {model} ===")
                
                # Test /api/embeddings endpoint (more standard)
                try:
                    payload = {
                        "model": model,
                        "prompt": "test embedding text"
                    }
                    
                    response = await client.post(f"{base_url}/api/embeddings", json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        embedding = data.get("embedding")
                        
                        if embedding and isinstance(embedding, list) and len(embedding) > 0:
                            logger.info(f"✅ SUCCESS: {model} - Embedding length: {len(embedding)}")
                            
                            # Test with this working model
                            logger.info(f"Using {model} for system configuration")
                            return model
                        else:
                            logger.warning(f"⚠️  Model {model} returned empty embedding")
                    else:
                        logger.error(f"❌ Model {model} failed: HTTP {response.status_code}")
                        
                except Exception as e:
                    logger.error(f"❌ Model {model} error: {e}")
            
        except Exception as e:
            logger.error(f"Failed to test models: {e}")
    
    return None

async def main():
    logger.info("🔍 Starting embedding endpoint diagnostics...")
    
    # Test basic endpoints
    await test_embedding_endpoints()
    
    # Find working model
    working_model = await test_with_different_models()
    
    if working_model:
        logger.info(f"\n✅ Found working embedding model: {working_model}")
        logger.info("💡 Recommendation: Update system configuration to use this model")
    else:
        logger.error("\n❌ No working embedding models found!")
        logger.info("💡 Try pulling a embedding model: ollama pull nomic-embed-text")

if __name__ == "__main__":
    asyncio.run(main())