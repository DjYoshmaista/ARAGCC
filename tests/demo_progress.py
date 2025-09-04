#!/usr/bin/env python3

import asyncio
import httpx
import json
from pathlib import Path

async def demonstrate_enhanced_progress():
    """Live demonstration of enhanced progress monitoring"""
    
    print("🎯 Enhanced Progress Monitoring System - Live Demo")
    print("=" * 55)
    
    # Create test file
    test_file = Path("demo_test.txt") 
    test_file.write_text("Enhanced Progress Monitoring Demo File\n" * 5)
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Submit task using the client's API format
            print("📤 Submitting ingestion task...")
            
            # Use the same format as the client
            response = await client.post("http://localhost:8001/tasks", json={
                "task_type": "folder_ingestion",
                "description": "Demo enhanced progress monitoring",
                "parameters": {
                    "paths": [str(test_file.absolute())],
                    "recursive": False,
                    "chunk_size": 5000,
                    "overlap_size": 32,
                    "embedding_model": "dengcao/Qwen3-Embedding-0.6B:Q8_0",
                    "max_concurrent_files": 5,
                    "max_concurrent_chunks": 10,
                    "heartbeat_timeout": 300
                }
            })
            
            if response.status_code != 200:
                print(f"❌ Task submission failed: {response.status_code}")
                print(response.text)
                return
                
            task_data = response.json()
            task_id = task_data["task_id"]
            print(f"✅ Task submitted: {task_id}")
            
            # Monitor progress in real-time
            print("\n📊 Live Progress Monitoring:")
            print("-" * 40)
            
            for i in range(8):
                await asyncio.sleep(1.5)
                
                try:
                    progress_resp = await client.get(f"http://localhost:8001/tasks/{task_id}/progress")
                    
                    if progress_resp.status_code == 200:
                        data = progress_resp.json()["progress"]
                        pg = data["postgres"] 
                        vec = data["vector"]
                        
                        print(f"🔄 [{i+1}/8] PostgreSQL: {pg['files_processed']} files, {pg['chunks_created']} chunks")
                        print(f"        Vector DB: {vec['embeddings_generated']} embeddings ({vec['embeddings_per_second']:.2f}/s)")
                        
                        if not data["active"]:
                            print("✅ Task completed!")
                            break
                    else:
                        print(f"📋 Progress API: {progress_resp.status_code}")
                        
                except Exception as e:
                    print(f"⚠️  Monitoring error: {e}")
            
            print("\n🎉 Enhanced Progress Monitoring Demo Complete!")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    finally:
        if test_file.exists():
            test_file.unlink()

if __name__ == "__main__":
    asyncio.run(demonstrate_enhanced_progress())