#!/usr/bin/env python3
"""
Example script demonstrating direct usage of the AgenticRAG RAG functionality
"""

import asyncio
import httpx
import json

class RAGClient:
    def __init__(self, orchestrator_url="http://localhost:8001", 
                 model_gateway_url="http://localhost:8070",
                 vector_engine_url="http://localhost:8080"):
        self.orchestrator_url = orchestrator_url
        self.model_gateway_url = model_gateway_url
        self.vector_engine_url = vector_engine_url
        self.client = httpx.AsyncClient()
    
    async def ingest_document(self, document_id: str, content: str, embedding_model: str = "dengcaoQwen3-Embedding-0.6B:Q8_0"):
        """Ingest a document into the RAG system"""
        # Submit task to orchestrator
        task_data = {
            "task_type": "rag_ingest",
            "description": f"Ingest document {document_id}",
            "parameters": {
                "document_id": document_id,
                "content": content,
                "embedding_model": embedding_model
            }
        }
        
        response = await self.client.post(f"{self.orchestrator_url}/tasks", json=task_data)
        response.raise_for_status()
        task_result = response.json()
        task_id = task_result["task_id"]
        
        # Wait for task completion
        while True:
            status_response = await self.client.get(f"{self.orchestrator_url}/tasks/{task_id}")
            status_response.raise_for_status()
            status_result = status_response.json()
            
            if status_result["status"] == "completed":
                return status_result["result"]
            elif status_result["status"] == "failed":
                raise Exception(f"Task failed: {status_result['error']}")
            
            await asyncio.sleep(1)
    
    async def query_documents(self, query: str, top_k: int = 5, embedding_model: str = "dengcaoQwen3-Embedding-0.6B:Q8_0"):
        """Query documents in the RAG system"""
        # Submit task to orchestrator
        task_data = {
            "task_type": "rag_query",
            "description": f"Query RAG with: {query}",
            "parameters": {
                "query": query,
                "top_k": top_k,
                "embedding_model": embedding_model
            }
        }
        
        response = await self.client.post(f"{self.orchestrator_url}/tasks", json=task_data)
        response.raise_for_status()
        task_result = response.json()
        task_id = task_result["task_id"]
        
        # Wait for task completion
        while True:
            status_response = await self.client.get(f"{self.orchestrator_url}/tasks/{task_id}")
            status_response.raise_for_status()
            status_result = status_response.json()
            
            if status_result["status"] == "completed":
                return json.loads(status_result["result"])
            elif status_result["status"] == "failed":
                raise Exception(f"Task failed: {status_result['error']}")
            
            await asyncio.sleep(1)
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

# Example usage
async def main():
    rag_client = RAGClient()
    
    try:
        # Example document to ingest
        sample_document = """
        The Python programming language is a high-level, interpreted programming language 
        known for its simplicity and readability. Created by Guido van Rossum and first 
        released in 1991, Python emphasizes code readability and allows programmers to 
        express concepts in fewer lines of code than languages like C++ or Java. Python 
        supports multiple programming paradigms, including procedural, object-oriented, 
        and functional programming.
        """
        
        # Ingest document
        print("Ingesting document...")
        result = await rag_client.ingest_document("python_info", sample_document)
        print(f"Ingestion result: {result}")
        
        # Query documents
        print("\nQuerying documents...")
        results = await rag_client.query_documents("What is Python programming language?")
        print("Query results:")
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await rag_client.close()

if __name__ == "__main__":
    asyncio.run(main())