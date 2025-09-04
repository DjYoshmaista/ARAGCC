#!/usr/bin/env python3
"""
Debug script to check multi-instance load balancer status
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# Add the orchestrator to the path
sys.path.insert(0, str(Path(__file__).parent / "services" / "orchestrator"))

from ollama_instance_manager import get_instance_manager
from embedding_load_balancer import get_load_balancer, LoadBalancingStrategy

async def debug_multi_instance():
    """Debug multi-instance system"""
    print("🔍 Debugging Multi-Instance Load Balancer...")
    
    # Get managers
    instance_manager = get_instance_manager()
    load_balancer = get_load_balancer(LoadBalancingStrategy.LEAST_CONNECTIONS)
    
    # Check instance manager
    print(f"\n📊 Instance Manager Status:")
    stats = instance_manager.get_stats()
    print(f"  Total instances: {stats['total_instances']}")
    print(f"  Running instances: {stats['running_instances']}")
    print(f"  Loaded instances: {stats['loaded_instances']}")
    
    print(f"\n🔍 Individual Instance Status:")
    for instance_stats in stats['instances']:
        print(f"  Port {instance_stats['port']}: {instance_stats['status']} "
              f"(model_loaded: {instance_stats['model_loaded']}, "
              f"requests: {instance_stats['requests_handled']})")
    
    # Test load balancer selection
    print(f"\n⚖️ Load Balancer Test:")
    available_instances = [
        instance for instance in instance_manager.instances
        if (instance.status == "running" and 
            instance.model_loaded and 
            load_balancer._is_instance_available(instance.port))
    ]
    
    print(f"  Available instances for load balancing: {len(available_instances)}")
    for instance in available_instances:
        print(f"    Port {instance.port}: {instance.base_url}")
    
    # Try to get next instance
    selected_instance = load_balancer._select_instance()
    if selected_instance:
        print(f"  Selected instance: {selected_instance.port} ({selected_instance.base_url})")
    else:
        print(f"  ❌ No instance selected by load balancer")
    
    # Test direct embedding generation
    print(f"\n🧪 Testing Direct Embedding Generation:")
    try:
        embedding, success, error_msg = await load_balancer.generate_embedding(
            "test text", "dengcao/Qwen3-Embedding-0.6B:Q8_0"
        )
        if success:
            print(f"  ✅ Embedding generated successfully (length: {len(embedding) if embedding else 0})")
        else:
            print(f"  ❌ Embedding failed: {error_msg}")
    except Exception as e:
        print(f"  ❌ Exception during embedding: {e}")
    
    # Test individual instance health
    print(f"\n🏥 Individual Instance Health Check:")
    for instance in instance_manager.instances:
        if instance.status == "running":
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{instance.base_url}/api/tags", timeout=5.0)
                    if response.status_code == 200:
                        print(f"  Port {instance.port}: ✅ Healthy")
                    else:
                        print(f"  Port {instance.port}: ❌ HTTP {response.status_code}")
            except Exception as e:
                print(f"  Port {instance.port}: ❌ Exception: {e}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(debug_multi_instance())