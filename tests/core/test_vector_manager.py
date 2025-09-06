"""
Comprehensive unit tests for the vector manager module.

Tests cover Qdrant vector operations, connection management, error handling,
and vector search functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import uuid as uuid_module

from core.database.vector_manager import VectorManager


class TestVectorManager:
    """Test suite for Qdrant vector manager."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock Qdrant configuration."""
        return {
            'host': 'localhost',
            'port': 6333,
            'timeout': 60
        }
    
    @pytest.fixture
    def sample_vector(self):
        """Sample vector embedding for testing."""
        return [0.1, 0.2, 0.3, 0.4, 0.5] * 128  # 640-dimensional vector
    
    @pytest.fixture
    def sample_metadata(self):
        """Sample metadata for testing."""
        return {
            'document_id': 'test_doc_123',
            'chunk_number': 1,
            'source': 'test_file.txt'
        }
    
    def test_init(self, mock_config):
        """Test VectorManager initialization."""
        manager = VectorManager(mock_config)
        
        assert manager.config == mock_config
        assert manager.client is None
    
    @patch('core.database.vector_manager.QdrantClient')
    def test_connect_success(self, mock_client_class, mock_config):
        """Test successful connection to Qdrant."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        manager = VectorManager(mock_config)
        result = manager.connect()
        
        assert result is True
        assert manager.client == mock_client
        mock_client_class.assert_called_once_with(
            host='localhost',
            port=6333,
            timeout=60
        )
    
    @patch('core.database.vector_manager.QdrantClient')
    def test_connect_failure(self, mock_client_class, mock_config):
        """Test connection failure handling."""
        mock_client_class.side_effect = Exception("Connection failed")
        
        manager = VectorManager(mock_config)
        result = manager.connect()
        
        assert result is False
        assert manager.client is None
    
    def test_disconnect(self, mock_config):
        """Test disconnection from Qdrant."""
        manager = VectorManager(mock_config)
        mock_client = Mock()
        manager.client = mock_client
        
        manager.disconnect()
        
        mock_client.close.assert_called_once()
        assert manager.client is None
    
    @patch('core.database.vector_manager.VectorManager.connect')
    def test_create_collection_new(self, mock_connect, mock_config):
        """Test creating a new collection."""
        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        
        manager = VectorManager(mock_config)
        manager.client = mock_client
        
        result = manager.create_collection('test_collection', 640)
        
        assert result is True
        mock_client.create_collection.assert_called_once()
    
    @patch('core.database.vector_manager.VectorManager.connect')
    def test_create_collection_existing(self, mock_connect, mock_config):
        """Test creating an existing collection."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.name = 'test_collection'
        mock_collections = Mock()
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections
        
        manager = VectorManager(mock_config)
        manager.client = mock_client
        
        result = manager.create_collection('test_collection', 640)
        
        assert result is True
        mock_client.create_collection.assert_not_called()
    
    @patch('core.database.vector_manager.VectorManager.connect')
    def test_recreate_collection_success(self, mock_connect, mock_config):
        """Test successful collection recreation."""
        mock_client = Mock()
        manager = VectorManager(mock_config)
        manager.client = mock_client
        
        result = manager.recreate_collection('test_collection', 640)
        
        assert result is True
        mock_client.recreate_collection.assert_called_once()
    
    @patch('core.database.vector_manager.VectorManager.connect')
    def test_store_vector_success(self, mock_connect, mock_config, sample_vector, sample_metadata):
        """Test successful vector storage."""
        mock_client = Mock()
        manager = VectorManager(mock_config)
        manager.client = mock_client
        
        result = manager.store_vector('chunk_123', sample_vector, 'test_collection', sample_metadata)
        
        assert result is True
        mock_client.upsert.assert_called_once()
        
        # Verify the point structure
        call_args = mock_client.upsert.call_args
        points = call_args[1]['points']
        assert len(points) == 1
        point = points[0]
        assert point.vector == sample_vector
        assert point.payload['chunk_id'] == 'chunk_123'
        assert 'document_id' in point.payload
    
    @patch('core.database.vector_manager.VectorManager.connect')
    def test_store_vector_no_metadata(self, mock_connect, mock_config, sample_vector):
        """Test vector storage without metadata."""
        mock_client = Mock()
        manager = VectorManager(mock_config)
        manager.client = mock_client
        
        result = manager.store_vector('chunk_123', sample_vector, 'test_collection')
        
        assert result is True
        call_args = mock_client.upsert.call_args
        points = call_args[1]['points']
        point = points[0]
        assert point.payload == {'chunk_id': 'chunk_123'}
    
    @patch('core.database.vector_manager.VectorManager.connect')
    def test_store_vectors_batch_success(self, mock_connect, mock_config):
        """Test successful batch vector storage."""
        mock_client = Mock()
        manager = VectorManager(mock_config)
        manager.client = mock_client
        
        chunk_vectors = [
            ('chunk_1', [0.1] * 640, {'doc_id': 'doc_1'}),
            ('chunk_2', [0.2] * 640, {'doc_id': 'doc_2'}),
            ('chunk_3', [0.3] * 640, None)
        ]
        
        result = manager.store_vectors_batch(chunk_vectors, 'test_collection')
        
        assert result == 3
        mock_client.upsert.assert_called_once()
        
        call_args = mock_client.upsert.call_args
        points = call_args[1]['points']
        assert len(points) == 3
    
    @patch('core.database.vector_manager.VectorManager.connect')
    def test_store_vectors_batch_empty(self, mock_connect, mock_config):
        """Test batch vector storage with empty list."""
        mock_client = Mock()
        manager = VectorManager(mock_config)
        manager.client = mock_client
        
        result = manager.store_vectors_batch([], 'test_collection')
        
        assert result == 0
        mock_client.upsert.assert_not_called()
    
    @patch('core.database.vector_manager.rate_limiter.acquire')
    @patch('core.database.vector_manager.VectorManager.connect')
    @pytest.mark.asyncio
    async def test_store_vector_async_success(self, mock_connect, mock_acquire, 
                                             mock_config, sample_vector):
        """Test successful asynchronous vector storage."""
        mock_client = Mock()
        manager = VectorManager(mock_config)
        manager.client = mock_client
        
        result = await manager.store_vector_async('chunk_123', sample_vector, 'test_collection')
        
        assert result is True
        mock_acquire.assert_called_once_with("qdrant")
        mock_client.upsert.assert_called_once()
    
    @patch('core.database.vector_manager.VectorManager.connect')
    def test_search_similar_success(self, mock_connect, mock_config, sample_vector):
        """Test successful similarity search."""
        mock_client = Mock()
        mock_results = [
            Mock(score=0.95, payload={'chunk_id': 'chunk_1', 'doc_id': 'doc_1'}),
            Mock(score=0.87, payload={'chunk_id': 'chunk_2', 'doc_id': 'doc_2'})
        ]
        mock_client.search.return_value = mock_results
        
        manager = VectorManager(mock_config)
        manager.client = mock_client
        
        results = manager.search_similar(sample_vector, 'test_collection', top_k=2)
        
        assert len(results) == 2
        assert results[0]['chunk_id'] == 'chunk_1'
        assert results[0]['score'] == 0.95
        assert results[1]['chunk_id'] == 'chunk_2'
        assert results[1]['score'] == 0.87
    
    @patch('core.database.vector_manager.VectorManager.connect')
    def test_search_with_filter_success(self, mock_connect, mock_config, sample_vector):
        """Test successful similarity search with filtering."""
        mock_client = Mock()
        mock_results = [
            Mock(score=0.92, payload={'chunk_id': 'chunk_1', 'doc_type': 'pdf'})
        ]
        mock_client.search.return_value = mock_results
        
        manager = VectorManager(mock_config)
        manager.client = mock_client
        
        filter_conditions = {'doc_type': 'pdf'}
        results = manager.search_with_filter(sample_vector, 'test_collection', filter_conditions)
        
        assert len(results) == 1
        assert results[0]['chunk_id'] == 'chunk_1'
        mock_client.search.assert_called_once()
        
        # Verify filter was applied
        call_args = mock_client.search.call_args
        assert call_args[1]['query_filter'] is not None
    
    @patch('core.database.vector_manager.VectorManager.connect')
    def test_delete_vector_success(self, mock_connect, mock_config):
        """Test successful vector deletion."""
        mock_client = Mock()
        manager = VectorManager(mock_config)
        manager.client = mock_client
        
        result = manager.delete_vector('chunk_123', 'test_collection')
        
        assert result is True
        mock_client.delete.assert_called_once()
        
        # Verify correct point ID is used
        call_args = mock_client.delete.call_args
        point_ids = call_args[1]['points_selector']
        expected_id = str(uuid_module.uuid5(uuid_module.NAMESPACE_DNS, 'chunk_123'))
        assert point_ids == [expected_id]
    
    @patch('core.database.vector_manager.VectorManager.connect')
    def test_count_vectors_success(self, mock_connect, mock_config):
        """Test successful vector counting."""
        mock_client = Mock()
        mock_collection_info = Mock()
        mock_collection_info.vectors_count = 1500
        mock_client.get_collection.return_value = mock_collection_info
        
        manager = VectorManager(mock_config)
        manager.client = mock_client
        
        result = manager.count_vectors('test_collection')
        
        assert result == 1500
        mock_client.get_collection.assert_called_once_with('test_collection')
    
    @patch('core.database.vector_manager.VectorManager.connect')
    def test_count_vectors_none(self, mock_connect, mock_config):
        """Test vector counting when count is None."""
        mock_client = Mock()
        mock_collection_info = Mock()
        mock_collection_info.vectors_count = None
        mock_client.get_collection.return_value = mock_collection_info
        
        manager = VectorManager(mock_config)
        manager.client = mock_client
        
        result = manager.count_vectors('test_collection')
        
        assert result == 0
    
    @patch('core.database.vector_manager.VectorManager.connect')
    def test_get_collection_info_success(self, mock_connect, mock_config):
        """Test successful collection info retrieval."""
        mock_client = Mock()
        mock_collection_info = Mock()
        mock_collection_info.vectors_count = 1000
        mock_collection_info.indexed_vectors_count = 950
        mock_collection_info.points_count = 1000
        mock_collection_info.segments_count = 2
        mock_collection_info.status = 'green'
        mock_collection_info.optimizer_status = 'ok'
        mock_collection_info.disk_data_size = 1024000
        mock_collection_info.ram_data_size = 512000
        mock_client.get_collection.return_value = mock_collection_info
        
        manager = VectorManager(mock_config)
        manager.client = mock_client
        
        result = manager.get_collection_info('test_collection')
        
        assert result is not None
        assert result['vectors_count'] == 1000
        assert result['indexed_vectors_count'] == 950
        assert result['points_count'] == 1000
        assert result['segments_count'] == 2
        assert result['status'] == 'green'
        assert result['optimizer_status'] == 'ok'
        assert result['disk_data_size'] == 1024000
        assert result['ram_data_size'] == 512000
    
    @patch('core.database.vector_manager.VectorManager.connect')
    def test_error_handling_with_logging(self, mock_connect, mock_config):
        """Test error handling and logging."""
        mock_client = Mock()
        mock_client.get_collection.side_effect = Exception("Database error")
        
        manager = VectorManager(mock_config)
        manager.client = mock_client
        
        with patch('core.database.vector_manager.logger') as mock_logger:
            result = manager.count_vectors('test_collection')
            
            assert result == 0
            mock_logger.error.assert_called_once()
    
    @patch('core.database.vector_manager.VectorManager.connect')
    def test_connection_failure_handling(self, mock_connect, mock_config):
        """Test handling when connection fails."""
        mock_connect.return_value = False
        
        manager = VectorManager(mock_config)
        result = manager.count_vectors('test_collection')
        
        assert result == 0
    
    @patch('asyncio.get_event_loop')
    @patch('core.database.vector_manager.VectorManager.connect')
    def test_rate_limiting_with_running_loop(self, mock_connect, mock_get_loop, 
                                           mock_config, sample_vector):
        """Test rate limiting with running event loop."""
        mock_client = Mock()
        mock_loop = Mock()
        mock_loop.is_running.return_value = True
        mock_get_loop.return_value = mock_loop
        
        manager = VectorManager(mock_config)
        manager.client = mock_client
        
        # Should not raise exception
        result = manager.store_vector('chunk_123', sample_vector, 'test_collection')
        assert result is True
    
    @patch('asyncio.get_event_loop')
    @patch('core.database.vector_manager.VectorManager.connect')
    def test_rate_limiting_no_loop(self, mock_connect, mock_get_loop, 
                                  mock_config, sample_vector):
        """Test rate limiting when no event loop exists."""
        mock_client = Mock()
        mock_get_loop.side_effect = RuntimeError("No event loop")
        
        manager = VectorManager(mock_config)
        manager.client = mock_client
        
        # Should not raise exception
        result = manager.store_vector('chunk_123', sample_vector, 'test_collection')
        assert result is True
    
    def test_uuid_generation_consistency(self):
        """Test that UUID generation is consistent for same chunk_id."""
        chunk_id = "test_chunk_123"
        
        # Generate UUID twice - should be identical
        uuid1 = str(uuid_module.uuid5(uuid_module.NAMESPACE_DNS, chunk_id))
        uuid2 = str(uuid_module.uuid5(uuid_module.NAMESPACE_DNS, chunk_id))
        
        assert uuid1 == uuid2
        
        # Different chunk_id should generate different UUID
        uuid3 = str(uuid_module.uuid5(uuid_module.NAMESPACE_DNS, "different_chunk"))
        assert uuid1 != uuid3


if __name__ == '__main__':
    pytest.main([__file__])