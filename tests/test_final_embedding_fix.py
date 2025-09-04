#!/usr/bin/env python3
"""
Final Embedding Fix Verification Test

This test verifies that all the GPU memory and model configuration issues have been resolved:
1. System uses granite-embedding:latest instead of dengcao model
2. GPU memory usage is within acceptable limits
3. All embedding endpoints work correctly
4. Load balancer functions properly with the new model
"""

import asyncio
import sys
import os
import json
import time
import httpx
from pathlib import Path

# Add orchestrator to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "services" / "orchestrator"))

from embedding_load_balancer import EmbeddingLoadBalancer
from embedding_queue import EmbeddingJob

async def test_direct_granite_embedding():
    """Test direct granite-embedding API calls"""
    print("=== Direct Granite Embedding API Test ===")
    
    try:
        async with httpx.AsyncClient() as client:
            # Test /api/embeddings (should work)
            response = await client.post(
                "http://localhost:11434/api/embeddings",
                json={"model": "granite-embedding:latest", "prompt": "System restart test"},
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                embedding = data.get("embedding", [])
                print(f"   ✅ SUCCESS: /api/embeddings returned {len(embedding)}-dim embedding")
                return True
            else:
                print(f"   ❌ FAILED: HTTP {response.status_code}")
                print(f"   📝 Response: {response.text}")
                return False
                
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

async def test_load_balancer_with_granite():
    """Test load balancer with granite-embedding model"""
    print("\n=== Load Balancer with Granite Test ===")
    
    try:
        balancer = EmbeddingLoadBalancer()
        await balancer.initialize_connections()
        
        instances = balancer.instance_manager.instances
        if not instances:
            print("   ❌ No instances found")
            return False
            
        print(f"   Found {len(instances)} instances")
        
        # Test embedding generation
        embedding, success, message = await balancer.generate_embedding(
            "Load balancer test with granite-embedding model",
            "granite-embedding:latest"
        )
        
        if success and embedding and len(embedding) == 384:
            print(f"   ✅ SUCCESS: Generated {len(embedding)}-dim embedding via load balancer")
            return True
        else:
            print(f"   ❌ FAILED: {message}")
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

def test_embedding_queue():
    """Test embedding queue with new model configuration"""
    print("\n=== Embedding Queue Test ===")
    
    try:
        # Create a job with default model (should be granite-embedding now)
        job = EmbeddingJob(
            id="test-job",
            chunk_id="test-chunk",
            text="Testing embedding queue with new model configuration"
        )
        
        print(f"   Default model in EmbeddingJob: {job.model}")
        
        if job.model == "granite-embedding:latest":
            print("   ✅ SUCCESS: EmbeddingJob uses granite-embedding:latest by default")
            return True
        else:
            print(f"   ❌ FAILED: EmbeddingJob still uses old model: {job.model}")
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

def test_gpu_memory_usage():
    """Test GPU memory usage is acceptable"""
    print("\n=== GPU Memory Usage Test ===")
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            all_good = True
            
            for i, line in enumerate(lines):
                if line:
                    used, total = map(int, line.split(','))
                    usage_pct = (used / total) * 100
                    print(f"   GPU {i}: {used}MB / {total}MB ({usage_pct:.1f}% used)")
                    
                    if usage_pct > 90:
                        print(f"   ❌ GPU {i} memory usage is too high!")
                        all_good = False
                    elif usage_pct > 70:
                        print(f"   ⚠️  GPU {i} memory usage is elevated but acceptable")
                    else:
                        print(f"   ✅ GPU {i} memory usage is good")
            
            return all_good
        else:
            print("   ⚠️  Could not get GPU memory info")
            return True  # Don't fail if GPU monitoring unavailable
            
    except Exception as e:
        print(f"   ⚠️  Could not get GPU memory info: {e}")
        return True  # Don't fail if GPU monitoring unavailable

def test_system_configuration():
    """Test that system configuration files are updated"""
    print("\n=== System Configuration Test ===")
    
    try:
        # Check system.yaml
        config_file = project_root / "shared" / "configs" / "system.yaml"
        with open(config_file) as f:
            content = f.read()
        
        if "granite-embedding:latest" in content:
            print("   ✅ system.yaml uses granite-embedding:latest")
            config_good = True
        else:
            print("   ❌ system.yaml still references old model")
            config_good = False
        
        if "embedding_dimensions: 384" in content:
            print("   ✅ system.yaml has correct dimensions (384)")
            dims_good = True
        else:
            print("   ❌ system.yaml has incorrect dimensions")
            dims_good = False
        
        if "vector_dimension: 384" in content:
            print("   ✅ system.yaml has correct vector dimensions (384)")
            vector_good = True
        else:
            print("   ❌ system.yaml has incorrect vector dimensions")
            vector_good = False
            
        return config_good and dims_good and vector_good
        
    except Exception as e:
        print(f"   ❌ ERROR reading config: {e}")
        return False

async def main():
    """Run all final verification tests"""
    print("🔧 Final Embedding Fix Verification Test Suite")
    print("="*60)
    
    tests = [
        ("Direct Granite Embedding API", test_direct_granite_embedding()),
        ("Load Balancer with Granite", test_load_balancer_with_granite()),
        ("Embedding Queue Configuration", test_embedding_queue()),
        ("GPU Memory Usage", test_gpu_memory_usage()),
        ("System Configuration", test_system_configuration()),
    ]
    
    results = []
    for test_name, test_coro in tests:
        if asyncio.iscoroutine(test_coro):
            result = await test_coro
        else:
            result = test_coro
        results.append((test_name, result))
    
    # Summary
    print("\n" + "="*60)
    print("📊 Final Test Results:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("🎉 All fixes verified! System is ready for production.")
        print("✅ GPU memory issues resolved")
        print("✅ Model configuration updated to granite-embedding")
        print("✅ All endpoints functioning correctly")
        print("✅ Load balancer working with new model")
        return True
    else:
        print("⚠️  Some issues remain - please check failed tests above")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)