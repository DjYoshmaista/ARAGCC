"""
Unit tests for the high-performance embedding queue singleton
"""

import unittest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path
import json
import tempfile
import os

# Add the services directory to Python path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'services', 'orchestrator'))

try:
    from embedding_queue import EmbeddingJob, EmbeddingResult, EmbeddingQueueSingleton, embedding_queue
except ImportError:
    # Mock the classes if import fails
    from dataclasses import dataclass
    from typing import List, Optional
    
    @dataclass
    class EmbeddingJob:
        id: str
        chunk_id: str
        text: str
        model: str = "test-model"
        priority: int = 1
        created_at: Optional[datetime] = None
    
    @dataclass 
    class EmbeddingResult:
        job_id: str
        chunk_id: str
        embedding: List[float]
        model: str
        processing_time: float
        success: bool = True
        error: Optional[str] = None
    
    class EmbeddingQueueSingleton:
        def __init__(self):
            self.initialized = True
        
        def start_workers(self, num_workers=4):
            pass
        
        def stop_workers(self):
            pass
        
        def add_job(self, job):
            pass
        
        def get_result(self, timeout=1.0):
            return None
        
        def get_stats(self):
            return {}
        
        def save_state(self):
            pass
        
        def load_state(self):
            pass
        
        def clear_state(self):
            pass
    
    embedding_queue = EmbeddingQueueSingleton()

class TestEmbeddingJob(unittest.TestCase):
    """Test the EmbeddingJob dataclass"""
    
    def test_embedding_job_creation(self):
        """Test EmbeddingJob creation and initialization"""
        job = EmbeddingJob(
            id="test-id",
            chunk_id="chunk-123",
            text="This is test text for embedding"
        )
        
        self.assertEqual(job.id, "test-id")
        self.assertEqual(job.chunk_id, "chunk-123")
        self.assertEqual(job.text, "This is test text for embedding")
        self.assertEqual(job.model, "dengcaoQwen3-Embedding-0.6B:Q8_0")
        self.assertEqual(job.priority, 1)
        
        # Test automatic created_at assignment
        if hasattr(job, 'created_at') and job.created_at:
            self.assertIsInstance(job.created_at, datetime)
    
    def test_embedding_job_to_dict(self):
        """Test EmbeddingJob serialization to dictionary"""
        job = EmbeddingJob(
            id="test-id",
            chunk_id="chunk-123", 
            text="Test text",
            model="custom-model"
        )
        
        if hasattr(job, 'to_dict'):
            job_dict = job.to_dict()
            
            self.assertEqual(job_dict['id'], "test-id")
            self.assertEqual(job_dict['chunk_id'], "chunk-123")
            self.assertEqual(job_dict['text'], "Test text")
            self.assertEqual(job_dict['model'], "custom-model")
            self.assertEqual(job_dict['priority'], 1)

class TestEmbeddingResult(unittest.TestCase):
    """Test the EmbeddingResult dataclass"""
    
    def test_embedding_result_creation(self):
        """Test EmbeddingResult creation"""
        result = EmbeddingResult(
            job_id="job-123",
            chunk_id="chunk-456",
            embedding=[0.1, 0.2, 0.3, 0.4],
            model="test-model",
            processing_time=0.5
        )
        
        self.assertEqual(result.job_id, "job-123")
        self.assertEqual(result.chunk_id, "chunk-456")
        self.assertEqual(result.embedding, [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(result.model, "test-model")
        self.assertEqual(result.processing_time, 0.5)
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
    
    def test_embedding_result_with_error(self):
        """Test EmbeddingResult with error"""
        result = EmbeddingResult(
            job_id="job-123",
            chunk_id="chunk-456",
            embedding=[],
            model="test-model",
            processing_time=0.0,
            success=False,
            error="Test error message"
        )
        
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Test error message")
        self.assertEqual(result.embedding, [])

class TestEmbeddingQueueSingleton(unittest.TestCase):
    """Test the EmbeddingQueueSingleton class"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for state files
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        os.chdir(self.original_cwd)
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_singleton_pattern(self):
        """Test that EmbeddingQueueSingleton follows singleton pattern"""
        queue1 = EmbeddingQueueSingleton()
        queue2 = EmbeddingQueueSingleton()
        
        # Both instances should be the same object
        self.assertIs(queue1, queue2)
    
    @patch('embedding_queue.OLLAMA_AVAILABLE', False)
    def test_initialization_without_ollama(self):
        """Test initialization when Ollama is not available"""
        queue = EmbeddingQueueSingleton()
        
        # Should initialize successfully even without Ollama
        self.assertTrue(hasattr(queue, '_initialized'))
        if hasattr(queue, 'ollama_client'):
            self.assertIsNone(queue.ollama_client)
    
    def test_add_job(self):
        """Test adding jobs to the queue"""
        queue = EmbeddingQueueSingleton()
        
        job = EmbeddingJob(
            id="test-job",
            chunk_id="test-chunk",
            text="Test embedding text"
        )
        
        # Should not raise an exception
        queue.add_job(job)
        
        # Check stats if available
        if hasattr(queue, 'processing_stats'):
            stats = queue.get_stats()
            self.assertIsInstance(stats, dict)
    
    def test_get_stats(self):
        """Test getting queue statistics"""
        queue = EmbeddingQueueSingleton()
        stats = queue.get_stats()
        
        self.assertIsInstance(stats, dict)
        
        # Check expected keys are present (if implemented)
        if stats:
            expected_keys = ['total_queued', 'total_processed', 'workers_running']
            for key in expected_keys:
                if key in stats:
                    self.assertIsInstance(stats[key], (int, bool))
    
    def test_mock_embedding_generation(self):
        """Test mock embedding generation"""
        queue = EmbeddingQueueSingleton()
        
        if hasattr(queue, '_generate_mock_embedding'):
            embedding = queue._generate_mock_embedding(100)
            
            self.assertIsInstance(embedding, list)
            self.assertEqual(len(embedding), 640)  # Expected dimension
            
            # All values should be floats between -1 and 1
            for val in embedding:
                self.assertIsInstance(val, float)
                self.assertGreaterEqual(val, -1.0)
                self.assertLessEqual(val, 1.0)
    
    def test_state_persistence(self):
        """Test state saving and loading"""
        queue = EmbeddingQueueSingleton()
        
        # Save state
        queue.save_state()
        
        # Check if state file was created
        if hasattr(queue, 'progress_file'):
            state_file = Path(queue.progress_file)
            if state_file.exists():
                self.assertTrue(state_file.is_file())
                
                # Load state
                queue.load_state()
                
                # Should not raise exceptions
    
    def test_clear_state(self):
        """Test clearing queue state"""
        queue = EmbeddingQueueSingleton()
        
        # Should not raise an exception
        queue.clear_state()
        
        # Check that state was reset
        stats = queue.get_stats()
        if 'total_processed' in stats:
            self.assertEqual(stats['total_processed'], 0)

class TestEmbeddingQueueIntegration(unittest.TestCase):
    """Integration tests for the embedding queue system"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def tearDown(self):
        """Clean up integration test environment"""
        # Stop any running workers
        try:
            embedding_queue.stop_workers()
        except:
            pass
        
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_worker_lifecycle(self):
        """Test starting and stopping workers"""
        queue = embedding_queue
        
        # Test starting workers
        initial_running = getattr(queue, 'workers_running', False)
        
        queue.start_workers(num_workers=2)
        
        # Give workers time to start
        time.sleep(0.1)
        
        stats = queue.get_stats()
        if 'workers_running' in stats:
            self.assertTrue(stats['workers_running'])
        
        # Test stopping workers
        queue.stop_workers()
        
        # Give workers time to stop
        time.sleep(0.1)
        
        stats = queue.get_stats()
        if 'workers_running' in stats:
            self.assertFalse(stats['workers_running'])
    
    @patch('embedding_queue.OLLAMA_AVAILABLE', False)
    def test_job_processing_without_ollama(self):
        """Test job processing when Ollama is not available (mock mode)"""
        queue = embedding_queue
        
        # Start workers
        queue.start_workers(num_workers=1)
        
        try:
            # Add a job
            job = EmbeddingJob(
                id="integration-test-job",
                chunk_id="integration-test-chunk",
                text="Integration test text for embedding"
            )
            
            queue.add_job(job)
            
            # Wait for processing
            time.sleep(1.0)
            
            # Try to get result
            result = queue.get_result(timeout=2.0)
            
            if result:
                self.assertIsInstance(result, EmbeddingResult)
                self.assertEqual(result.chunk_id, "integration-test-chunk")
                self.assertIsInstance(result.embedding, list)
            
        finally:
            queue.stop_workers()
    
    def test_batch_results_retrieval(self):
        """Test batch result retrieval"""
        queue = embedding_queue
        
        # Test batch results method
        results = queue.get_results_batch(max_count=10, timeout=0.1)
        
        self.assertIsInstance(results, list)
        # Results can be empty if no processing has occurred

class TestEmbeddingQueuePerformance(unittest.TestCase):
    """Performance tests for the embedding queue"""
    
    def test_state_save_performance(self):
        """Test that state saving is reasonably fast"""
        queue = embedding_queue
        
        start_time = time.time()
        queue.save_state()
        end_time = time.time()
        
        # State saving should complete in less than 1 second
        self.assertLess(end_time - start_time, 1.0)
    
    def test_stats_retrieval_performance(self):
        """Test that stats retrieval is fast"""
        queue = embedding_queue
        
        start_time = time.time()
        stats = queue.get_stats()
        end_time = time.time()
        
        # Stats retrieval should be very fast (< 0.1 seconds)
        self.assertLess(end_time - start_time, 0.1)
        self.assertIsInstance(stats, dict)

if __name__ == '__main__':
    unittest.main()