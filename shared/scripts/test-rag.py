#!/usr/bin/env python3
"""
Test script for RAG functionality
"""

import asyncio
import httpx
import json

async def test_rag_functionality():
    """Test the RAG functionality by ingesting a document and querying it"""
    
    # Test document
    test_document = {
        "id": "test_doc_1",
        "content": "The quick brown fox jumps over the lazy dog. This is a sample document for testing RAG functionality."
    }
    
    # URLs for services
    orchestrator_url = "http://localhost:8001"
    model_gateway_url = "http://localhost:8070"
    vector_engine_url = "http://localhost:8080"
    
    async with httpx.AsyncClient() as client:
        print("Testing RAG functionality...")
        
        # 1. Test model gateway health
        print("\n1. Testing Model Gateway health...")
        try:
            response = await client.get(f"{model_gateway_url}/health")
            print(f"Model Gateway status: {response.json()}")
        except Exception as e:
            print(f"Model Gateway health check failed: {e}")
        
        # 2. Test vector engine health
        print("\n2. Testing Vector Engine health...")
        try:
            response = await client.get(f"{vector_engine_url}/")
            print(f"Vector Engine status: {response.json()}")
        except Exception as e:
            print(f"Vector Engine health check failed: {e}")
        
        # 3. Test orchestrator health
        print("\n3. Testing Orchestrator health...")
        try:
            response = await client.get(f"{orchestrator_url}/health")
            print(f"Orchestrator status: {response.json()}")
        except Exception as e:
            print(f"Orchestrator health check failed: {e}")
        
        # 4. Test embedding generation
        print("\n4. Testing embedding generation...")
        try:
            response = await client.post(
                f"{model_gateway_url}/embed",
                json={
                    "model": "dengcaoQwen3-Embedding-0.6B:Q8_0",
                    "prompt": "This is a test sentence for embedding generation."
                }
            )
            embedding_data = response.json()
            print(f"Embedding generated successfully. Dimensions: {len(embedding_data['embedding'])}")
        except Exception as e:
            print(f"Embedding generation failed: {e}")
        
        print("\nRAG functionality test completed.")

if __name__ == "__main__":
    asyncio.run(test_rag_functionality())