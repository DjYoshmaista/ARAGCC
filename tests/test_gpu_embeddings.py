#!/usr/bin/env python3
"""
Test script to verify GPU utilization for embeddings
"""

import asyncio
import time
import subprocess
from services.orchestrator.embedding_queue import embedding_queue, EmbeddingJob

def get_gpu_status():
    """Get current GPU utilization"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        print(f"Could not get GPU status: {e}")
    return None

def test_embedding_generation():
    """Test embedding generation with GPU monitoring"""
    print("Testing embedding generation with GPU monitoring...")
    
    # Get initial GPU status
    print("Initial GPU status:")
    initial_gpu = get_gpu_status()
    if initial_gpu:
        print(initial_gpu)
    
    # Start embedding workers
    embedding_queue.start_workers(num_workers=2)
    
    # Create test jobs
    test_texts = [
        "This is a test sentence for embedding generation.",
        "Another test sentence to verify GPU utilization.",
        "We are testing if the GPU is being properly utilized for embeddings.",
        "The system should be using the GPU for faster embedding generation.",
        "If GPU utilization increases, we know the fix is working correctly."
    ]
    
    # Submit jobs
    jobs = []
    for i, text in enumerate(test_texts):
        job = EmbeddingJob(
            id=f"test_job_{i}",
            chunk_id=f"test_chunk_{i}",
            text=text,
            model="dengcao/Qwen3-Embedding-0.6B:Q8_0"
        )
        embedding_queue.add_job(job)
        jobs.append(job)
    
    print(f"Submitted {len(jobs)} embedding jobs. Monitoring GPU utilization...")
    
    # Monitor GPU while jobs are processing
    start_time = time.time()
    results = []
    
    # Check GPU status periodically
    for i in range(30):  # Check for 30 seconds
        # Get results
        result = embedding_queue.get_result(timeout=0.1)
        if result:
            results.append(result)
            print(f"Received result {len(results)}/{len(jobs)}")
        
        # Check if all jobs are done
        if len(results) >= len(jobs):
            break
            
        # Check GPU status every 2 seconds
        if i % 2 == 0:
            gpu_status = get_gpu_status()
            if gpu_status:
                print(f"GPU status at {time.time() - start_time:.1f}s: {gpu_status}")
        
        time.sleep(1)
    
    # Final GPU status
    print("Final GPU status:")
    final_gpu = get_gpu_status()
    if final_gpu:
        print(final_gpu)
    
    # Stop workers
    embedding_queue.stop_workers()
    
    # Report results
    print(f"\nResults:")
    print(f"  Jobs submitted: {len(jobs)}")
    print(f"  Results received: {len(results)}")
    print(f"  Success rate: {len([r for r in results if r.success])}/{len(results)}")
    
    if results:
        avg_time = sum([r.processing_time for r in results if r.processing_time > 0]) / len([r for r in results if r.processing_time > 0])
        print(f"  Average processing time: {avg_time:.2f}s")
        
        # Check embedding dimensions
        if results[0].success:
            print(f"  Embedding dimensions: {len(results[0].embedding)}")
    
    return len(results) == len(jobs)

if __name__ == "__main__":
    print("GPU Embedding Test")
    print("=" * 50)
    
    success = test_embedding_generation()
    
    if success:
        print("\n✅ Test completed successfully! GPU should be utilized for embeddings.")
    else:
        print("\n❌ Test failed. Check logs for details.")