#!/usr/bin/env python3
"""
Debug script to test GPU embedding generation
"""

import ollama
import time

def test_gpu_embedding():
    print("Testing GPU embedding generation...")
    
    # Create client
    client = ollama.Client()
    print("Client created successfully")
    
    # Test embedding with GPU
    start_time = time.time()
    try:
        response = client.embeddings(
            model="dengcao/Qwen3-Embedding-0.6B:Q8_0",
            prompt="This is a test sentence for GPU embedding generation.",
            options={
                "num_thread": 4,
                "num_gpu": 1
            }
        )
        end_time = time.time()
        
        embedding = response['embedding']
        print(f"Embedding generated successfully!")
        print(f"  - Time taken: {end_time - start_time:.2f} seconds")
        print(f"  - Embedding dimensions: {len(embedding)}")
        print(f"  - First 5 values: {embedding[:5]}")
        
        return True
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return False

if __name__ == "__main__":
    success = test_gpu_embedding()
    if success:
        print("\n✅ GPU embedding test completed successfully!")
    else:
        print("\n❌ GPU embedding test failed!")
