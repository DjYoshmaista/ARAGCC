#!/usr/bin/env python3

"""
Test script for the enhanced progress monitoring system
"""

import asyncio
import httpx
import time
from pathlib import Path

async def test_enhanced_progress_monitoring():
    """Test the complete enhanced progress monitoring system"""
    
    # Create a small test file
    test_file = Path("small_test.txt")
    test_file.write_text("""
This is a test document for the enhanced progress monitoring system.

Features implemented:
- Database activity monitoring with 5-minute inactivity timeouts
- Dual progress bars for PostgreSQL and Vector DB operations
- Interactive user prompts when timeouts occur
- Real-time progress updates via messaging system
- Heartbeat monitoring to detect processing stalls

This system now provides comprehensive monitoring of ingestion tasks.
""")
    
    try:
        print("🚀 Testing Enhanced Progress Monitoring System")
        print("=" * 50)
        
        # Submit ingestion task
        async with httpx.AsyncClient(timeout=30.0) as client:
            print("📤 Submitting ingestion task...")
            task_response = await client.post("http://localhost:8001/tasks", json={
                "type": "folder_ingestion",
                "description": f"Test enhanced progress monitoring",
                "parameters": {
                    "paths": [str(test_file.absolute())],
                    "recursive": False,
                    "heartbeat_timeout": 300
                }
            })
            task_response.raise_for_status()
            task_data = task_response.json()
            task_id = task_data["task_id"]
            print(f"✅ Task submitted: {task_id}")
            
            # Monitor progress using enhanced API
            print("\n📊 Monitoring Progress...")
            print("-" * 30)
            
            for i in range(10):  # Monitor for 10 iterations
                await asyncio.sleep(2)
                
                try:
                    progress_response = await client.get(f"http://localhost:8001/tasks/{task_id}/progress")
                    if progress_response.status_code == 200:
                        progress_data = progress_response.json()
                        pg_progress = progress_data["progress"]["postgres"]
                        vec_progress = progress_data["progress"]["vector"]
                        
                        print(f"📁 PostgreSQL: {pg_progress['files_processed']} files, {pg_progress['chunks_created']} chunks")
                        print(f"🔍 Vector DB: {vec_progress['embeddings_generated']} embeddings, {vec_progress['embeddings_stored']} stored")
                        print(f"📈 Activity: PG={pg_progress['files_per_second']:.2f}/s, Vec={vec_progress['embeddings_per_second']:.2f}/s")
                        print()
                    else:
                        print(f"❌ Progress API returned {progress_response.status_code}")
                        
                except Exception as e:
                    print(f"⚠️  Progress check failed: {e}")
                    
            # Check final task status
            print("🔍 Checking final task status...")
            final_status = await client.get(f"http://localhost:8001/tasks/{task_id}")
            if final_status.status_code == 200:
                status_data = final_status.json()
                print(f"✅ Task completed with status: {status_data['status']}")
            
            print("\n🎉 Enhanced Progress Monitoring Test Complete!")
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        # Clean up test file
        if test_file.exists():
            test_file.unlink()

if __name__ == "__main__":
    asyncio.run(test_enhanced_progress_monitoring())