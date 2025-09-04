#!/usr/bin/env python3
"""
Test granite-embedding model integration with load balancer
"""
import asyncio
import sys
import os

# Add orchestrator to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'services', 'orchestrator'))

from embedding_load_balancer import EmbeddingLoadBalancer
from ollama_instance_manager import OllamaInstanceManager, OllamaInstance

async def test_granite_integration():
    """Test the granite-embedding model with load balancer"""
    print("=== Testing Granite Embedding Integration ===\n")
    
    # Create load balancer
    balancer = EmbeddingLoadBalancer()
    
    # Initialize load balancer connections (this will discover instances)
    print("1. Initializing load balancer and discovering instances...")
    await balancer.initialize_connections()
    
    # Check what instances were found
    instances = balancer.instance_manager.instances
    if instances:
        print(f"   Found {len(instances)} instances:")
        for instance in instances:
            print(f"     - Port {instance.port}: {instance.status} (model_loaded: {instance.model_loaded})")
    else:
        print("   No instances found, manually adding port 11434...")
        instance = OllamaInstance(port=11434, status="running")
        instance.model_loaded = True
        balancer.instance_manager.instances.append(instance)
        await balancer.initialize_connections()
        print(f"   Added instance: {instance.port}")
    
    # Test embedding generation
    print("\n2. Testing embedding generation...")
    test_text = "This is a test sentence for embedding generation using the granite model."
    
    embedding, success, message = await balancer.generate_embedding(
        text=test_text,
        model="granite-embedding:latest"
    )
    
    if success and embedding:
        print(f"   ✅ SUCCESS: Generated embedding with {len(embedding)} dimensions")
        print(f"   📊 First 5 values: {embedding[:5]}")
        print(f"   📝 Response message: {message}")
    else:
        print(f"   ❌ FAILED: {message}")
        return False
    
    # Test multiple requests to verify load balancing
    print("\n3. Testing multiple requests for load balancing...")
    test_texts = [
        "First test sentence",
        "Second test sentence", 
        "Third test sentence"
    ]
    
    success_count = 0
    for i, text in enumerate(test_texts, 1):
        embedding, success, msg = await balancer.generate_embedding(
            text=text,
            model="granite-embedding:latest"
        )
        if success and embedding:
            print(f"   ✅ Request {i}: SUCCESS ({len(embedding)} dims)")
            success_count += 1
        else:
            print(f"   ❌ Request {i}: FAILED - {msg}")
    
    # Get final stats
    print(f"\n4. Final Statistics:")
    stats = balancer.get_stats()
    print(f"   Total requests: {stats.get('total_requests', 0)}")
    print(f"   Successful requests: {stats.get('successful_requests', 0)}")
    print(f"   Success rate: {stats.get('success_rate', 0)*100:.1f}%")
    print(f"   Average response time: {stats.get('avg_response_time', 0):.3f}s")
    
    if success_count == len(test_texts):
        print("\n🎉 All tests passed! Granite embedding integration is working correctly.")
        return True
    else:
        print(f"\n⚠️  Only {success_count}/{len(test_texts)} requests succeeded.")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_granite_integration())
    sys.exit(0 if success else 1)