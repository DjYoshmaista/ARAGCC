#!/usr/bin/env python3
"""
Test script to check Ollama import in the same context as the running system
"""

import sys
import os
import logging

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging to match the system
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_ollama_import():
    """Test Ollama import exactly as in embedding_queue.py"""
    print("Testing Ollama import...")
    
    try:
        import ollama
        OLLAMA_AVAILABLE = True
        print(f"Ollama library imported successfully")
        print(f"Ollama version: {ollama.__version__ if hasattr(ollama, '__version__') else 'Unknown'}")
        
        # Try to create a client
        client = ollama.Client(host="http://localhost:11434")
        print("Ollama client created successfully")
        
        # Try a simple embedding
        response = client.embeddings(
            model="dengcao/Qwen3-Embedding-0.6B:Q8_0",
            prompt="test"
        )
        print(f"Embedding test successful: {len(response['embedding'])} dimensions")
        
    except ImportError as e:
        OLLAMA_AVAILABLE = False
        print(f"Ollama import failed: {e}")
    except Exception as e:
        print(f"Ollama test failed: {e}")
        import traceback
        traceback.print_exc()
        
    print(f"OLLAMA_AVAILABLE: {OLLAMA_AVAILABLE}")
    return OLLAMA_AVAILABLE

if __name__ == "__main__":
    test_ollama_import()