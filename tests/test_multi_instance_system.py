#!/usr/bin/env python3
"""
Comprehensive Test Suite for Multi-Instance Embedding Generation System

Tests the entire multi-instance architecture including:
- Ollama instance management and spawning
- Load balancing across multiple instances
- Embedding queue performance and throughput
- Qdrant batch insertion optimization
- GPU utilization and memory management
"""

import asyncio
import time
import json
import logging
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add the services directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'services', 'orchestrator'))

# Import our components
from ollama_instance_manager import OllamaInstanceManager, get_instance_manager
from embedding_load_balancer import EmbeddingLoadBalancer, get_load_balancer, LoadBalancingStrategy
from embedding_queue import embedding_queue, EmbeddingJob
from performance_monitor import get_performance_monitor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_multi_instance.log')
    ]
)
logger = logging.getLogger("multi_instance_test")

class MultiInstanceTester:
    """Comprehensive tester for multi-instance embedding system"""
    
    def __init__(self):
        self.instance_manager = None
        self.load_balancer = None
        self.performance_monitor = get_performance_monitor()
        self.test_results = {}
        
    async def run_full_test_suite(self):
        """Run the complete test suite"""
        logger.info("=" * 60)
        logger.info("STARTING MULTI-INSTANCE EMBEDDING SYSTEM TEST SUITE")
        logger.info("=" * 60)
        
        self.test_results = {
            'start_time': time.time(),
            'tests': {},
            'overall_success': True
        }
        
        try:
            # Test 1: Instance Manager
            await self.test_instance_manager()
            
            # Test 2: Load Balancer
            await self.test_load_balancer()
            
            # Test 3: Embedding Queue Performance
            await self.test_embedding_queue_performance()
            
            # Test 4: GPU Utilization
            await self.test_gpu_utilization()
            
            # Test 5: Stress Test
            await self.test_system_stress()
            
            # Test 6: Performance Monitoring
            await self.test_performance_monitoring()
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            self.test_results['overall_success'] = False
        
        finally:
            await self.cleanup_test_environment()
            self.generate_test_report()
    
    async def test_instance_manager(self):
        """Test Ollama instance manager functionality"""
        logger.info("\n--- Testing Ollama Instance Manager ---")
        test_name = "instance_manager"
        
        try:
            # Initialize instance manager
            self.instance_manager = get_instance_manager()
            
            # Test GPU detection
            gpu_info = self.instance_manager.get_gpu_info()
            logger.info(f"Detected {len(gpu_info)} GPUs")
            
            for gpu in gpu_info:
                logger.info(f"  GPU {gpu['index']}: {gpu['name']} "
                           f"({gpu['memory_used']}/{gpu['memory_total']}MB, {gpu['utilization']}% util)")
            
            # Calculate optimal instances
            optimal_instances = self.instance_manager.calculate_optimal_instances()
            logger.info(f"Optimal instance count: {optimal_instances}")
            
            # Start instances
            logger.info("Starting Ollama instances...")
            success = await self.instance_manager.start_instances()
            
            if success:
                stats = self.instance_manager.get_stats()
                logger.info(f"Successfully started {stats['running_instances']}/{stats['total_instances']} instances")
                
                # Test instance health
                await asyncio.sleep(5)  # Wait for instances to stabilize
                await self.instance_manager._check_instance_health()
                
                stats = self.instance_manager.get_stats()
                logger.info(f"Health check: {stats['loaded_instances']} instances have models loaded")
                
                self.test_results['tests'][test_name] = {
                    'success': True,
                    'instances_started': stats['running_instances'],
                    'instances_healthy': stats['loaded_instances'],
                    'optimal_count': optimal_instances
                }
                
            else:
                logger.error("Failed to start instances")
                self.test_results['tests'][test_name] = {'success': False, 'error': 'Failed to start instances'}
        
        except Exception as e:
            logger.error(f"Instance manager test failed: {e}")
            self.test_results['tests'][test_name] = {'success': False, 'error': str(e)}
    
    async def test_load_balancer(self):
        """Test load balancer functionality"""
        logger.info("\n--- Testing Load Balancer ---")
        test_name = "load_balancer"
        
        try:
            # Initialize load balancer
            self.load_balancer = get_load_balancer(LoadBalancingStrategy.LEAST_CONNECTIONS)
            await self.load_balancer.initialize_connections()
            
            # Test single embedding generation
            text = "This is a test sentence for embedding generation."
            model = "dengcao/Qwen3-Embedding-0.6B:Q8_0"
            
            logger.info("Testing single embedding generation...")
            start_time = time.time()
            embedding, success, error = await self.load_balancer.generate_embedding(text, model)
            response_time = time.time() - start_time
            
            if success:
                logger.info(f"Single embedding: SUCCESS ({len(embedding)} dims, {response_time:.3f}s)")
            else:
                logger.error(f"Single embedding failed: {error}")
                self.test_results['tests'][test_name] = {'success': False, 'error': error}
                return
            
            # Test batch embedding generation
            logger.info("Testing batch embedding generation...")
            batch_requests = [(f"Test sentence {i}", model) for i in range(10)]
            
            start_time = time.time()
            batch_results = await self.load_balancer.batch_generate_embeddings(batch_requests)
            batch_time = time.time() - start_time
            
            successful_batch = sum(1 for _, success, _ in batch_results if success)
            logger.info(f"Batch embeddings: {successful_batch}/10 successful ({batch_time:.3f}s total)")
            
            # Get load balancer stats
            stats = self.load_balancer.get_stats()
            logger.info(f"Load balancer stats: {stats['total_requests']} requests, "
                       f"{stats['success_rate']:.1%} success rate")
            
            self.test_results['tests'][test_name] = {
                'success': True,
                'single_embedding_time': response_time,
                'batch_success_rate': successful_batch / 10,
                'batch_total_time': batch_time,
                'load_balancer_stats': stats
            }
            
        except Exception as e:
            logger.error(f"Load balancer test failed: {e}")
            self.test_results['tests'][test_name] = {'success': False, 'error': str(e)}
    
    async def test_embedding_queue_performance(self):
        """Test embedding queue performance and throughput"""
        logger.info("\n--- Testing Embedding Queue Performance ---")
        test_name = "embedding_queue"
        
        try:
            # Start embedding queue workers
            await embedding_queue.start_workers(num_workers=8)
            
            # Wait for workers to be ready
            await asyncio.sleep(3)
            
            # Create test jobs
            logger.info("Generating test embedding jobs...")
            test_jobs = []
            for i in range(50):  # Test with 50 jobs
                job = EmbeddingJob(
                    id=f"test_job_{i}",
                    chunk_id=f"chunk_{i}",
                    text=f"This is test text number {i} for embedding generation testing.",
                    model="dengcao/Qwen3-Embedding-0.6B:Q8_0"
                )
                test_jobs.append(job)
            
            # Submit jobs
            start_time = time.time()
            for job in test_jobs:
                embedding_queue.add_job(job)
            
            logger.info(f"Submitted {len(test_jobs)} jobs to embedding queue")
            
            # Wait for results
            results = []
            timeout = 60.0  # 1 minute timeout
            deadline = time.time() + timeout
            
            while len(results) < len(test_jobs) and time.time() < deadline:
                result = embedding_queue.get_result(timeout=1.0)
                if result:
                    results.append(result)
                    if len(results) % 10 == 0:
                        logger.info(f"Received {len(results)}/{len(test_jobs)} results...")
            
            total_time = time.time() - start_time
            successful_results = sum(1 for r in results if r.success)
            
            logger.info(f"Embedding queue test completed:")
            logger.info(f"  - Total jobs: {len(test_jobs)}")
            logger.info(f"  - Successful: {successful_results}")
            logger.info(f"  - Total time: {total_time:.2f}s")
            logger.info(f"  - Throughput: {successful_results/total_time:.2f} embeddings/sec")
            
            # Get queue stats
            queue_stats = embedding_queue.get_stats()
            
            self.test_results['tests'][test_name] = {
                'success': True,
                'jobs_submitted': len(test_jobs),
                'jobs_successful': successful_results,
                'total_time': total_time,
                'throughput': successful_results / total_time,
                'queue_stats': queue_stats
            }
            
        except Exception as e:
            logger.error(f"Embedding queue test failed: {e}")
            self.test_results['tests'][test_name] = {'success': False, 'error': str(e)}
    
    async def test_gpu_utilization(self):
        """Test GPU utilization during processing"""
        logger.info("\n--- Testing GPU Utilization ---")
        test_name = "gpu_utilization"
        
        try:
            if not self.instance_manager:
                logger.warning("Instance manager not available for GPU testing")
                return
            
            # Get initial GPU state
            initial_gpu = self.instance_manager.get_gpu_info()
            logger.info("Initial GPU state:")
            for gpu in initial_gpu:
                logger.info(f"  GPU {gpu['index']}: {gpu['utilization']}% util, "
                           f"{gpu['memory_used']}/{gpu['memory_total']}MB")
            
            # Create intensive workload
            logger.info("Creating intensive embedding workload...")
            intensive_jobs = []
            for i in range(100):
                job = EmbeddingJob(
                    id=f"intensive_job_{i}",
                    chunk_id=f"intensive_chunk_{i}",
                    text="This is a longer test text for intensive GPU utilization testing. " * 10,
                    model="dengcao/Qwen3-Embedding-0.6B:Q8_0"
                )
                intensive_jobs.append(job)
                embedding_queue.add_job(job)
            
            # Monitor GPU during processing
            await asyncio.sleep(5)  # Let processing start
            
            peak_gpu = self.instance_manager.get_gpu_info()
            logger.info("Peak GPU utilization during processing:")
            for gpu in peak_gpu:
                logger.info(f"  GPU {gpu['index']}: {gpu['utilization']}% util, "
                           f"{gpu['memory_used']}/{gpu['memory_total']}MB")
            
            # Calculate improvements
            initial_avg_util = sum(g['utilization'] for g in initial_gpu) / len(initial_gpu) if initial_gpu else 0
            peak_avg_util = sum(g['utilization'] for g in peak_gpu) / len(peak_gpu) if peak_gpu else 0
            
            self.test_results['tests'][test_name] = {
                'success': True,
                'initial_avg_utilization': initial_avg_util,
                'peak_avg_utilization': peak_avg_util,
                'utilization_improvement': peak_avg_util - initial_avg_util,
                'gpus_detected': len(peak_gpu)
            }
            
        except Exception as e:
            logger.error(f"GPU utilization test failed: {e}")
            self.test_results['tests'][test_name] = {'success': False, 'error': str(e)}
    
    async def test_system_stress(self):
        """Stress test the entire system"""
        logger.info("\n--- System Stress Test ---")
        test_name = "stress_test"
        
        try:
            logger.info("Starting system stress test with 200 embedding jobs...")
            
            # Create large workload
            stress_jobs = []
            for i in range(200):
                job = EmbeddingJob(
                    id=f"stress_job_{i}",
                    chunk_id=f"stress_chunk_{i}",
                    text=f"Stress test text {i}: " + "Lorem ipsum dolor sit amet. " * 20,
                    model="dengcao/Qwen3-Embedding-0.6B:Q8_0"
                )
                stress_jobs.append(job)
            
            # Submit all jobs
            start_time = time.time()
            for job in stress_jobs:
                embedding_queue.add_job(job)
            
            logger.info(f"Submitted {len(stress_jobs)} stress test jobs")
            
            # Monitor progress
            results_count = 0
            timeout = 180.0  # 3 minute timeout for stress test
            deadline = time.time() + timeout
            
            while results_count < len(stress_jobs) and time.time() < deadline:
                result = embedding_queue.get_result(timeout=1.0)
                if result:
                    results_count += 1
                    if results_count % 50 == 0:
                        elapsed = time.time() - start_time
                        rate = results_count / elapsed
                        logger.info(f"Stress test progress: {results_count}/{len(stress_jobs)} "
                                   f"({rate:.1f} embeddings/sec)")
            
            total_time = time.time() - start_time
            final_rate = results_count / total_time
            
            logger.info(f"Stress test completed:")
            logger.info(f"  - Processed: {results_count}/{len(stress_jobs)} jobs")
            logger.info(f"  - Total time: {total_time:.1f}s")
            logger.info(f"  - Final throughput: {final_rate:.1f} embeddings/sec")
            
            self.test_results['tests'][test_name] = {
                'success': True,
                'jobs_submitted': len(stress_jobs),
                'jobs_processed': results_count,
                'total_time': total_time,
                'final_throughput': final_rate,
                'completion_rate': results_count / len(stress_jobs)
            }
            
        except Exception as e:
            logger.error(f"Stress test failed: {e}")
            self.test_results['tests'][test_name] = {'success': False, 'error': str(e)}
    
    async def test_performance_monitoring(self):
        """Test performance monitoring system"""
        logger.info("\n--- Testing Performance Monitoring ---")
        test_name = "performance_monitoring"
        
        try:
            # Configure performance monitor
            self.performance_monitor.set_components(
                instance_manager=self.instance_manager,
                embedding_queue=embedding_queue,
                load_balancer=self.load_balancer
            )
            
            # Start monitoring
            self.performance_monitor.start_monitoring()
            
            # Let it collect some metrics
            await asyncio.sleep(15)
            
            # Generate performance report
            report = self.performance_monitor.get_performance_report(duration_minutes=5)
            
            logger.info("Performance monitoring report:")
            if 'averages' in report:
                avg = report['averages']
                logger.info(f"  - Avg GPU utilization: {avg.get('gpu_utilization_percent', 0):.1f}%")
                logger.info(f"  - Avg embedding rate: {avg.get('embedding_processing_rate', 0):.1f}/sec")
                logger.info(f"  - Avg queue size: {avg.get('queue_size', 0):.1f}")
                logger.info(f"  - Success rate: {avg.get('success_rate_percent', 0):.1f}%")
            
            logger.info(f"  - Alerts generated: {report.get('alerts_count', 0)}")
            
            # Save metrics to file
            self.performance_monitor.save_metrics_to_file('test_performance_metrics.json')
            
            self.test_results['tests'][test_name] = {
                'success': True,
                'report': report,
                'monitoring_duration': 15
            }
            
        except Exception as e:
            logger.error(f"Performance monitoring test failed: {e}")
            self.test_results['tests'][test_name] = {'success': False, 'error': str(e)}
        finally:
            self.performance_monitor.stop_monitoring()
    
    async def cleanup_test_environment(self):
        """Clean up test environment"""
        logger.info("\n--- Cleaning Up Test Environment ---")
        
        try:
            # Stop embedding queue
            if hasattr(embedding_queue, 'stop_workers'):
                embedding_queue.stop_workers()
            
            # Clean up load balancer
            if self.load_balancer:
                await self.load_balancer.cleanup()
            
            # Stop instances
            if self.instance_manager:
                self.instance_manager.stop_all_instances()
            
            logger.info("Test environment cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("\n" + "=" * 60)
        logger.info("MULTI-INSTANCE EMBEDDING SYSTEM TEST REPORT")
        logger.info("=" * 60)
        
        self.test_results['end_time'] = time.time()
        self.test_results['total_duration'] = self.test_results['end_time'] - self.test_results['start_time']
        
        # Summary
        total_tests = len(self.test_results['tests'])
        successful_tests = sum(1 for test in self.test_results['tests'].values() if test.get('success', False))
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Successful: {successful_tests}")
        logger.info(f"Failed: {total_tests - successful_tests}")
        logger.info(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        logger.info(f"Total Duration: {self.test_results['total_duration']:.1f}s")
        
        # Individual test results
        logger.info("\nIndividual Test Results:")
        for test_name, result in self.test_results['tests'].items():
            status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
            logger.info(f"  {test_name}: {status}")
            if not result.get('success', False) and 'error' in result:
                logger.info(f"    Error: {result['error']}")
        
        # Performance highlights
        if 'embedding_queue' in self.test_results['tests']:
            eq_test = self.test_results['tests']['embedding_queue']
            if eq_test.get('success') and 'throughput' in eq_test:
                logger.info(f"\nPerformance Highlights:")
                logger.info(f"  - Embedding Throughput: {eq_test['throughput']:.1f} embeddings/sec")
        
        if 'gpu_utilization' in self.test_results['tests']:
            gpu_test = self.test_results['tests']['gpu_utilization']
            if gpu_test.get('success'):
                improvement = gpu_test.get('utilization_improvement', 0)
                logger.info(f"  - GPU Utilization Improvement: +{improvement:.1f}%")
        
        # Save full report
        report_file = 'multi_instance_test_report.json'
        try:
            with open(report_file, 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)
            logger.info(f"\nFull test report saved to: {report_file}")
        except Exception as e:
            logger.error(f"Failed to save test report: {e}")
        
        # Final verdict
        if self.test_results['overall_success'] and successful_tests == total_tests:
            logger.info("\n🎉 ALL TESTS PASSED! Multi-instance system is working correctly.")
        else:
            logger.info("\n⚠️  Some tests failed. Check the logs and fix issues before deployment.")

async def main():
    """Main test function"""
    tester = MultiInstanceTester()
    await tester.run_full_test_suite()

if __name__ == "__main__":
    asyncio.run(main())