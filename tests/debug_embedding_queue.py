#!/usr/bin/env python3
"""
Debug script to check Ollama availability and embedding queue state
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'services', 'orchestrator'))

# Import the embedding queue
from embedding_queue import embedding_queue, OLLAMA_AVAILABLE

def debug_embedding_queue():
    print("=== Embedding Queue Debug Info ===")
    print(f"OLLAMA_AVAILABLE: {OLLAMA_AVAILABLE}")
    print(f"Number of Ollama clients: {len(embedding_queue.ollama_clients)}")
    print(f"Available Ollama clients: {sum(1 for client in embedding_queue.ollama_clients if client is not None)}/{len(embedding_queue.ollama_clients)}")
    
    # Check if any clients are None
    for i, client in enumerate(embedding_queue.ollama_clients):
        print(f"  Client {i}: {'Available' if client is not None else 'None'}")
    
    # Try to create a test embedding
    if OLLAMA_AVAILABLE and any(embedding_queue.ollama_clients):
        print("\nAttempting to generate test embedding...")
        try:
            from embedding_queue import EmbeddingJob
            test_job = EmbeddingJob(
                id="debug_test",
                chunk_id="debug_chunk",
                text="This is a debug test sentence.",
                model="dengcao/Qwen3-Embedding-0.6B:Q8_0"
            )
            
            # Try with the first available client
            client = None
            for i, c in enumerate(embedding_queue.ollama_clients):
                if c is not None:
                    client = c
                    break
            
            if client:
                response = client.embeddings(
                    model=test_job.model,
                    prompt=test_job.text,
                    options={"num_thread": 4, "num_gpu": 1}
                )
                embedding = response['embedding']
                print(f"✅ Test embedding successful! Dimensions: {len(embedding)}")
            else:
                print("❌ No working Ollama client found")
        except Exception as e:
            print(f"❌ Test embedding failed: {e}")
    else:
        print("❌ Ollama not available or no clients initialized")

if __name__ == "__main__":
    debug_embedding_queue()