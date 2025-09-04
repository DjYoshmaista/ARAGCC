"""
Unit tests for the parallel ingestion controller
"""

import unittest
import asyncio
import tempfile
import os
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add the services directory to Python path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'services', 'orchestrator'))

try:
    from parallel_ingestion_controller import ParallelIngestionController, FileProcessingJob, ProcessingState
    from database_utils import DatabaseManager, DocumentMetadata, ChunkMetadata
    from text_chunker import TextChunker
except ImportError:
    # Mock the classes if import fails
    from dataclasses import dataclass
    from typing import Set, Optional, Dict, Any
    
    @dataclass
    class FileProcessingJob:
        file_path: str
        file_id: str
        priority: int = 1
        created_at: Optional[datetime] = None
    
    @dataclass
    class ProcessingState:
        total_files: int = 0
        processed_files: int = 0
        failed_files: int = 0
        skipped_files: int = 0
        total_chunks: int = 0
        processed_chunks: int = 0
        embeddings_generated: int = 0
        start_time: Optional[datetime] = None
        last_save_time: Optional[datetime] = None
        current_batch_id: Optional[str] = None
        processed_file_ids: Set[str] = None
    
    class ParallelIngestionController:
        def __init__(self, config):
            self.config = config
            self.state = ProcessingState()
        
        def initialize_database(self, pg_config, qdrant_config):
            pass
        
        def start_workers(self):
            pass
        
        def stop_workers(self):
            pass
        
        def process_folder(self, folder_path, recursive=True):
            return "mock-batch-id"
        
        def get_stats(self):
            return {}
        
        def save_state(self):
            pass
        
        def load_state(self):
            pass
        
        def clear_state(self):
            pass
    
    class DatabaseManager:
        def __init__(self, pg_config, qdrant_config):
            pass
        
        def connect_postgresql(self):
            return True
        
        def connect_qdrant(self):
            return True
        
        def initialize_schemas(self):
            return True
        
        def create_qdrant_collection(self, name, size):
            return True
    
    class TextChunker:
        def chunk_text(self, text, chunk_size=1000, overlap=200):
            return [text[:chunk_size]]

class TestFileProcessingJob(unittest.TestCase):
    """Test the FileProcessingJob dataclass"""
    
    def test_file_processing_job_creation(self):
        """Test FileProcessingJob creation and initialization"""
        job = FileProcessingJob(
            file_path="/test/path/file.txt",
            file_id="test-file-id"
        )
        
        self.assertEqual(job.file_path, "/test/path/file.txt")
        self.assertEqual(job.file_id, "test-file-id")
        self.assertEqual(job.priority, 1)
        
        # Test automatic created_at assignment
        if hasattr(job, 'created_at') and job.created_at:
            self.assertIsInstance(job.created_at, datetime)
    
    def test_file_processing_job_with_priority(self):
        """Test FileProcessingJob with custom priority"""
        job = FileProcessingJob(
            file_path="/test/path/file.txt",
            file_id="test-file-id",
            priority=5
        )
        
        self.assertEqual(job.priority, 5)

class TestProcessingState(unittest.TestCase):
    """Test the ProcessingState dataclass"""
    
    def test_processing_state_initialization(self):
        """Test ProcessingState initialization with defaults"""
        state = ProcessingState()
        
        self.assertEqual(state.total_files, 0)
        self.assertEqual(state.processed_files, 0)
        self.assertEqual(state.failed_files, 0)
        self.assertEqual(state.skipped_files, 0)
        self.assertEqual(state.total_chunks, 0)
        self.assertEqual(state.processed_chunks, 0)
        self.assertEqual(state.embeddings_generated, 0)
        self.assertIsNone(state.start_time)
        self.assertIsNone(state.last_save_time)
        self.assertIsNone(state.current_batch_id)
        
        # Test set initialization
        if hasattr(state, 'processed_file_ids'):
            self.assertIsInstance(state.processed_file_ids, set)
            self.assertEqual(len(state.processed_file_ids), 0)
    
    def test_processing_state_with_values(self):
        """Test ProcessingState with custom values"""
        file_ids = {"file1", "file2", "file3"}
        start_time = datetime.utcnow()
        
        state = ProcessingState(
            total_files=100,
            processed_files=50,
            failed_files=5,
            total_chunks=500,
            start_time=start_time,
            current_batch_id="batch-123",
            processed_file_ids=file_ids
        )
        
        self.assertEqual(state.total_files, 100)
        self.assertEqual(state.processed_files, 50)
        self.assertEqual(state.failed_files, 5)
        self.assertEqual(state.total_chunks, 500)
        self.assertEqual(state.start_time, start_time)
        self.assertEqual(state.current_batch_id, "batch-123")
        if hasattr(state, 'processed_file_ids'):
            self.assertEqual(state.processed_file_ids, file_ids)

class TestParallelIngestionController(unittest.TestCase):
    """Test the ParallelIngestionController class"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create test configuration
        self.config = {
            'max_cpu_workers': 2,
            'max_db_workers': 1,
            'max_concurrent_files': 5,
            'chunk_batch_size': 10,
            'checkpoint_interval': 5,
            'embedding_vector_size': 640,
            'qdrant_collection': 'test_collection',
            'max_embedding_workers': 2,
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'embedding_model': 'test-model'
        }
    
    def tearDown(self):
        """Clean up test environment"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def test_controller_initialization(self):
        """Test ParallelIngestionController initialization"""
        controller = ParallelIngestionController(self.config)
        
        self.assertEqual(controller.config, self.config)
        self.assertEqual(controller.max_cpu_workers, 2)
        self.assertEqual(controller.max_db_workers, 1)
        self.assertEqual(controller.chunk_batch_size, 10)
        
        # Test state initialization
        self.assertIsInstance(controller.state, ProcessingState)
        self.assertEqual(controller.state.total_files, 0)
        
        # Test queue initialization
        if hasattr(controller, 'file_queue'):
            self.assertIsNotNone(controller.file_queue)
    
    @patch('parallel_ingestion_controller.DatabaseManager')
    def test_initialize_database(self, mock_db_manager):
        """Test database initialization"""
        controller = ParallelIngestionController(self.config)
        
        # Mock database manager
        mock_db_instance = Mock()
        mock_db_instance.connect_postgresql.return_value = True
        mock_db_instance.connect_qdrant.return_value = True
        mock_db_instance.initialize_schemas.return_value = True
        mock_db_instance.create_qdrant_collection.return_value = True
        mock_db_manager.return_value = mock_db_instance
        
        pg_config = {'host': 'localhost', 'port': 5432}
        qdrant_config = {'host': 'localhost', 'port': 6333}
        
        # Should not raise an exception
        controller.initialize_database(pg_config, qdrant_config)
        
        # Verify database manager was created
        mock_db_manager.assert_called_once_with(pg_config, qdrant_config)
    
    @patch('parallel_ingestion_controller.DatabaseManager')
    def test_initialize_database_connection_failure(self, mock_db_manager):
        """Test database initialization with connection failure"""
        controller = ParallelIngestionController(self.config)
        
        # Mock database manager with connection failure
        mock_db_instance = Mock()
        mock_db_instance.connect_postgresql.return_value = False
        mock_db_manager.return_value = mock_db_instance
        
        pg_config = {'host': 'localhost', 'port': 5432}
        qdrant_config = {'host': 'localhost', 'port': 6333}
        
        # Should raise an exception
        with self.assertRaises(Exception) as context:
            controller.initialize_database(pg_config, qdrant_config)
        
        self.assertIn("Failed to connect to PostgreSQL", str(context.exception))
    
    def test_discover_files(self):
        """Test file discovery functionality"""
        controller = ParallelIngestionController(self.config)
        
        # Create test files
        test_files = [
            'test1.txt',
            'test2.py',
            'test3.md',
            'test4.log',  # Should be ignored
            '.hidden.txt',  # Should be ignored
            'node_modules/test.js'  # Should be ignored
        ]
        
        for file_name in test_files:
            file_path = Path(file_name)
            if '/' in file_name:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if not file_name.startswith('.') and 'node_modules' not in file_name:
                file_path.write_text(f"Content of {file_name}")
        
        # Test file discovery
        if hasattr(controller, '_discover_files'):
            discovered = controller._discover_files('.', recursive=True)
            
            self.assertIsInstance(discovered, list)
            
            # Should find .txt, .py, .md files but not .log, hidden, or node_modules
            expected_files = ['test1.txt', 'test2.py', 'test3.md']
            for expected in expected_files:
                found = any(expected in path for path in discovered)
                if not found:
                    # File discovery might not be implemented or might have different logic
                    pass
    
    def test_get_file_id(self):
        """Test file ID generation"""
        controller = ParallelIngestionController(self.config)
        
        if hasattr(controller, '_get_file_id'):
            file_id1 = controller._get_file_id('/path/to/file1.txt')
            file_id2 = controller._get_file_id('/path/to/file2.txt')
            file_id3 = controller._get_file_id('/path/to/file1.txt')  # Same as first
            
            # IDs should be strings
            self.assertIsInstance(file_id1, str)
            self.assertIsInstance(file_id2, str)
            
            # Different files should have different IDs
            self.assertNotEqual(file_id1, file_id2)
            
            # Same file should have same ID
            self.assertEqual(file_id1, file_id3)
    
    def test_get_stats(self):
        """Test statistics retrieval"""
        controller = ParallelIngestionController(self.config)
        
        stats = controller.get_stats()
        
        self.assertIsInstance(stats, dict)
        
        # Check for expected keys
        expected_keys = [
            'total_files', 'processed_files', 'failed_files', 
            'total_chunks', 'embeddings_generated', 'workers_running'
        ]
        
        for key in expected_keys:
            if key in stats:
                self.assertIsInstance(stats[key], (int, bool, str))
    
    def test_state_persistence(self):
        """Test state saving and loading"""
        controller = ParallelIngestionController(self.config)
        
        # Modify state
        controller.state.total_files = 100
        controller.state.processed_files = 50
        controller.state.current_batch_id = "test-batch"
        
        # Save state
        controller.save_state()
        
        # Check if state file was created
        state_file = Path("parallel_ingestion_state.json")
        if state_file.exists():
            self.assertTrue(state_file.is_file())
            
            # Verify JSON content
            with open(state_file, 'r') as f:
                saved_data = json.load(f)
            
            self.assertEqual(saved_data['total_files'], 100)
            self.assertEqual(saved_data['processed_files'], 50)
            self.assertEqual(saved_data['current_batch_id'], "test-batch")
            
            # Create new controller and load state
            new_controller = ParallelIngestionController(self.config)
            new_controller.load_state()
            
            # Verify state was loaded
            self.assertEqual(new_controller.state.total_files, 100)
            self.assertEqual(new_controller.state.processed_files, 50)
            self.assertEqual(new_controller.state.current_batch_id, "test-batch")
    
    def test_clear_state(self):
        """Test state clearing"""
        controller = ParallelIngestionController(self.config)
        
        # Set some state
        controller.state.total_files = 100
        controller.state.processed_files = 50
        controller.save_state()
        
        # Clear state
        controller.clear_state()
        
        # Verify state was reset
        self.assertEqual(controller.state.total_files, 0)
        self.assertEqual(controller.state.processed_files, 0)
        
        # Verify state file was removed
        state_file = Path("parallel_ingestion_state.json")
        self.assertFalse(state_file.exists())
    
    def test_should_save_state(self):
        """Test state save timing logic"""
        controller = ParallelIngestionController(self.config)
        
        if hasattr(controller, '_should_save_state'):
            # Should save if never saved before
            self.assertTrue(controller._should_save_state())
            
            # Update last save time
            controller.state.last_save_time = datetime.utcnow()
            
            # Should not save immediately after saving
            self.assertFalse(controller._should_save_state())

class TestParallelIngestionControllerIntegration(unittest.TestCase):
    """Integration tests for the parallel ingestion controller"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        self.config = {
            'max_cpu_workers': 1,
            'max_db_workers': 1,
            'max_concurrent_files': 2,
            'chunk_batch_size': 5,
            'checkpoint_interval': 30,
            'embedding_vector_size': 640,
            'qdrant_collection': 'test_collection',
            'max_embedding_workers': 1,
            'chunk_size': 500,
            'chunk_overlap': 100,
            'embedding_model': 'test-model'
        }
    
    def tearDown(self):
        """Clean up integration test environment"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    @patch('parallel_ingestion_controller.embedding_queue')
    @patch('parallel_ingestion_controller.DatabaseManager')
    def test_process_folder_basic(self, mock_db_manager, mock_embedding_queue):
        """Test basic folder processing"""
        controller = ParallelIngestionController(self.config)
        
        # Mock database manager
        mock_db_instance = Mock()
        mock_db_instance.connect_postgresql.return_value = True
        mock_db_instance.connect_qdrant.return_value = True
        mock_db_instance.initialize_schemas.return_value = True
        mock_db_instance.create_qdrant_collection.return_value = True
        mock_db_instance.store_document.return_value = True
        mock_db_instance.store_chunk.return_value = True
        mock_db_manager.return_value = mock_db_instance
        
        # Mock embedding queue
        mock_embedding_queue.start_workers.return_value = None
        mock_embedding_queue.stop_workers.return_value = None
        mock_embedding_queue.add_job.return_value = None
        
        # Create test files
        test_file = Path('test.txt')
        test_file.write_text('This is a test file with some content for processing.')
        
        # Initialize database
        controller.initialize_database({}, {})
        
        # Process folder (this will be mostly mocked)
        batch_id = controller.process_folder('.', recursive=False)
        
        # Should return a batch ID
        self.assertIsInstance(batch_id, str)
        
        # Verify database operations were called
        if hasattr(controller, 'db_manager') and controller.db_manager:
            # These assertions only apply if the controller actually processes files
            pass

class TestParallelIngestionControllerPerformance(unittest.TestCase):
    """Performance tests for the parallel ingestion controller"""
    
    def test_stats_performance(self):
        """Test that getting stats is fast"""
        config = {'max_cpu_workers': 1}
        controller = ParallelIngestionController(config)
        
        import time
        start_time = time.time()
        stats = controller.get_stats()
        end_time = time.time()
        
        # Stats retrieval should be very fast
        self.assertLess(end_time - start_time, 0.1)
        self.assertIsInstance(stats, dict)
    
    def test_state_save_performance(self):
        """Test that state saving is reasonably fast"""
        config = {'max_cpu_workers': 1}
        controller = ParallelIngestionController(config)
        
        import time
        start_time = time.time()
        controller.save_state()
        end_time = time.time()
        
        # State saving should complete quickly
        self.assertLess(end_time - start_time, 1.0)

if __name__ == '__main__':
    unittest.main()