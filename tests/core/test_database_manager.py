"""
Comprehensive unit tests for the database manager module.

Tests cover PostgreSQL operations, connection management, error handling,
and data integrity for document and chunk storage.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import psycopg2

from core.database.postgresql_manager import PostgreSQLManager
from core.database.schemas import DocumentMetadata, ChunkMetadata


class TestPostgreSQLManager:
    """Test suite for PostgreSQL database manager."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock database configuration."""
        return {
            'host': 'localhost',
            'port': 5432,
            'database': 'test_db',
            'username': 'test_user',
            'password': 'test_pass',
            'use_connection_pool': True,
            'pool_min_conn': 1,
            'pool_max_conn': 5
        }
    
    @pytest.fixture
    def sample_document(self):
        """Sample document metadata for testing."""
        return DocumentMetadata(
            id="test_doc_123",
            file_path="/path/to/test.txt",
            file_name="test.txt",
            folder_path="/path/to",
            file_size=1024,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            total_chunks=3
        )
    
    @pytest.fixture
    def sample_chunk(self):
        """Sample chunk metadata for testing."""
        return ChunkMetadata(
            id="chunk_123",
            document_id="test_doc_123",
            chunk_number=1,
            chunk_text="This is a test chunk of text.",
            token_count=7,
            overlap_tokens=2,
            previous_chunk_id=None,
            next_chunk_id="chunk_124",
            created_at=datetime.now()
        )
    
    def test_init(self, mock_config):
        """Test PostgreSQLManager initialization."""
        manager = PostgreSQLManager(mock_config)
        
        assert manager.config == mock_config
        assert manager.connection is None
        assert manager.connection_pool is None
        assert manager._pool_min_conn == mock_config['pool_min_conn']
        assert manager._pool_max_conn == mock_config['pool_max_conn']
    
    def test_init_with_defaults(self):
        """Test initialization with default pool settings."""
        config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'test_db',
            'username': 'test_user',
            'password': 'test_pass'
        }
        
        manager = PostgreSQLManager(config)
        assert manager._pool_min_conn == 1
        assert manager._pool_max_conn == 10  # Default value
    
    @patch('core.database.postgresql_manager.SimpleConnectionPool')
    def test_connect_with_pool(self, mock_pool_class, mock_config):
        """Test connection establishment with connection pool."""
        mock_pool = Mock()
        mock_pool_class.return_value = mock_pool
        
        manager = PostgreSQLManager(mock_config)
        result = manager.connect()
        
        assert result is True
        assert manager.connection_pool == mock_pool
        mock_pool_class.assert_called_once_with(
            1, 5,
            host='localhost',
            port=5432,
            database='test_db',
            user='test_user',
            password='test_pass'
        )
    
    @patch('core.database.postgresql_manager.psycopg2.connect')
    def test_connect_without_pool(self, mock_connect, mock_config):
        """Test connection establishment without connection pool."""
        mock_config['use_connection_pool'] = False
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        manager = PostgreSQLManager(mock_config)
        result = manager.connect()
        
        assert result is True
        assert manager.connection == mock_conn
        mock_connect.assert_called_once_with(
            host='localhost',
            port=5432,
            database='test_db',
            user='test_user',
            password='test_pass'
        )
    
    @patch('core.database.postgresql_manager.SimpleConnectionPool')
    def test_connect_failure(self, mock_pool_class, mock_config):
        """Test connection failure handling."""
        mock_pool_class.side_effect = psycopg2.Error("Connection failed")
        
        manager = PostgreSQLManager(mock_config)
        result = manager.connect()
        
        assert result is False
        assert manager.connection_pool is None
    
    def test_disconnect_with_pool(self, mock_config):
        """Test disconnection with connection pool."""
        manager = PostgreSQLManager(mock_config)
        mock_pool = Mock()
        manager.connection_pool = mock_pool
        
        manager.disconnect()
        
        mock_pool.closeall.assert_called_once()
        assert manager.connection_pool is None
    
    def test_disconnect_without_pool(self, mock_config):
        """Test disconnection without connection pool."""
        manager = PostgreSQLManager(mock_config)
        mock_conn = Mock()
        manager.connection = mock_conn
        
        manager.disconnect()
        
        mock_conn.close.assert_called_once()
        assert manager.connection is None
    
    def test_get_connection_context_manager_with_pool(self, mock_config):
        """Test connection context manager with pool."""
        manager = PostgreSQLManager(mock_config)
        mock_pool = Mock()
        mock_conn = Mock()
        mock_pool.getconn.return_value = mock_conn
        manager.connection_pool = mock_pool
        
        with manager.get_connection() as conn:
            assert conn == mock_conn
        
        mock_pool.getconn.assert_called_once()
        mock_pool.putconn.assert_called_once_with(mock_conn)
    
    @patch('core.database.postgresql_manager.PostgreSQLManager.connect')
    def test_get_connection_context_manager_without_pool(self, mock_connect, mock_config):
        """Test connection context manager without pool."""
        mock_config['use_connection_pool'] = False
        manager = PostgreSQLManager(mock_config)
        mock_conn = Mock()
        manager.connection = mock_conn
        
        with manager.get_connection() as conn:
            assert conn == mock_conn
        
        # Connection should not be returned to pool
        assert manager.connection == mock_conn
    
    @patch('core.database.postgresql_manager.PostgreSQLManager.get_connection')
    def test_initialize_schema_success(self, mock_get_conn, mock_config):
        """Test successful schema initialization."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        
        manager = PostgreSQLManager(mock_config)
        result = manager.initialize_schema()
        
        assert result is True
        mock_conn.commit.assert_called_once()
        assert mock_cursor.execute.call_count >= 5  # Tables + indexes
    
    @patch('core.database.postgresql_manager.PostgreSQLManager.get_connection')
    def test_initialize_schema_failure(self, mock_get_conn, mock_config):
        """Test schema initialization failure."""
        mock_get_conn.side_effect = psycopg2.Error("Schema error")
        
        manager = PostgreSQLManager(mock_config)
        result = manager.initialize_schema()
        
        assert result is False
    
    @patch('core.database.postgresql_manager.rate_limiter.acquire')
    @patch('core.database.postgresql_manager.PostgreSQLManager.get_connection')
    @pytest.mark.asyncio
    async def test_store_document_async_success(self, mock_get_conn, mock_acquire, 
                                               mock_config, sample_document):
        """Test successful asynchronous document storage."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        
        manager = PostgreSQLManager(mock_config)
        result = await manager.store_document_async(sample_document)
        
        assert result is True
        mock_acquire.assert_called_once_with("postgresql")
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()
    
    @patch('core.database.postgresql_manager.rate_limiter.acquire')
    @patch('core.database.postgresql_manager.PostgreSQLManager.get_connection')
    @pytest.mark.asyncio
    async def test_store_document_async_failure(self, mock_get_conn, mock_acquire,
                                               mock_config, sample_document):
        """Test asynchronous document storage failure."""
        mock_get_conn.side_effect = psycopg2.Error("Storage error")
        
        manager = PostgreSQLManager(mock_config)
        result = await manager.store_document_async(sample_document)
        
        assert result is False
    
    @patch('core.database.postgresql_manager.PostgreSQLManager.get_connection')
    def test_store_document_success(self, mock_get_conn, mock_config, sample_document):
        """Test successful synchronous document storage."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        
        manager = PostgreSQLManager(mock_config)
        result = manager.store_document(sample_document)
        
        assert result is True
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()
    
    @patch('core.database.postgresql_manager.PostgreSQLManager.get_connection')
    def test_store_chunk_success(self, mock_get_conn, mock_config, sample_chunk):
        """Test successful chunk storage."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        
        manager = PostgreSQLManager(mock_config)
        result = manager.store_chunk(sample_chunk)
        
        assert result is True
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()
    
    @patch('core.database.postgresql_manager.PostgreSQLManager.get_connection')
    def test_store_chunks_batch_success(self, mock_get_conn, mock_config):
        """Test successful batch chunk storage."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        
        chunks = [
            ChunkMetadata(
                id=f"chunk_{i}",
                document_id="test_doc",
                chunk_number=i,
                chunk_text=f"Chunk {i} text",
                token_count=5,
                overlap_tokens=1,
                previous_chunk_id=f"chunk_{i-1}" if i > 1 else None,
                next_chunk_id=f"chunk_{i+1}" if i < 3 else None,
                created_at=datetime.now()
            ) for i in range(1, 4)
        ]
        
        manager = PostgreSQLManager(mock_config)
        result = manager.store_chunks_batch(chunks)
        
        assert result == 3
        assert mock_cursor.execute.call_count == 3
        mock_conn.commit.assert_called_once()
    
    @patch('core.database.postgresql_manager.PostgreSQLManager.get_connection')
    def test_get_document_success(self, mock_get_conn, mock_config):
        """Test successful document retrieval."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_row = {
            'id': 'test_doc',
            'file_path': '/path/test.txt',
            'file_name': 'test.txt',
            'folder_path': '/path',
            'file_size': 1024,
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'total_chunks': 2
        }
        mock_cursor.fetchone.return_value = mock_row
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        
        manager = PostgreSQLManager(mock_config)
        result = manager.get_document('test_doc')
        
        assert isinstance(result, DocumentMetadata)
        assert result.id == 'test_doc'
        mock_cursor.execute.assert_called_once()
    
    @patch('core.database.postgresql_manager.PostgreSQLManager.get_connection')
    def test_get_document_not_found(self, mock_get_conn, mock_config):
        """Test document retrieval when not found."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        
        manager = PostgreSQLManager(mock_config)
        result = manager.get_document('nonexistent')
        
        assert result is None
    
    @patch('core.database.postgresql_manager.PostgreSQLManager.get_connection')
    def test_count_documents_success(self, mock_get_conn, mock_config):
        """Test successful document counting."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (42,)
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        
        manager = PostgreSQLManager(mock_config)
        result = manager.count_documents()
        
        assert result == 42
        mock_cursor.execute.assert_called_once_with("SELECT COUNT(*) FROM documents")
    
    @patch('core.database.postgresql_manager.PostgreSQLManager.get_connection')
    def test_count_chunks_success(self, mock_get_conn, mock_config):
        """Test successful chunk counting."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (128,)
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        
        manager = PostgreSQLManager(mock_config)
        result = manager.count_chunks()
        
        assert result == 128
        mock_cursor.execute.assert_called_once_with("SELECT COUNT(*) FROM chunks")
    
    @patch('core.database.postgresql_manager.PostgreSQLManager.get_connection')
    def test_get_chunks_by_document_success(self, mock_get_conn, mock_config):
        """Test successful chunk retrieval by document."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_rows = [
            {
                'id': 'chunk_1',
                'document_id': 'test_doc',
                'chunk_number': 1,
                'chunk_text': 'First chunk',
                'token_count': 5,
                'overlap_tokens': 0,
                'previous_chunk_id': None,
                'next_chunk_id': 'chunk_2',
                'created_at': datetime.now()
            },
            {
                'id': 'chunk_2',
                'document_id': 'test_doc',
                'chunk_number': 2,
                'chunk_text': 'Second chunk',
                'token_count': 6,
                'overlap_tokens': 2,
                'previous_chunk_id': 'chunk_1',
                'next_chunk_id': None,
                'created_at': datetime.now()
            }
        ]
        mock_cursor.fetchall.return_value = mock_rows
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        
        manager = PostgreSQLManager(mock_config)
        result = manager.get_chunks_by_document('test_doc')
        
        assert len(result) == 2
        assert all(isinstance(chunk, ChunkMetadata) for chunk in result)
        assert result[0].id == 'chunk_1'
        assert result[1].id == 'chunk_2'
    
    @patch('core.database.postgresql_manager.PostgreSQLManager.get_connection')
    def test_error_handling_with_logging(self, mock_get_conn, mock_config):
        """Test error handling and logging."""
        mock_get_conn.side_effect = psycopg2.Error("Database error")
        
        manager = PostgreSQLManager(mock_config)
        
        with patch('core.database.postgresql_manager.logger') as mock_logger:
            result = manager.count_documents()
            
            assert result == 0
            mock_logger.error.assert_called_once()


class TestDatabaseSchemas:
    """Test suite for database schemas."""
    
    def test_document_metadata_creation(self):
        """Test DocumentMetadata creation."""
        now = datetime.now()
        doc = DocumentMetadata(
            id="test_123",
            file_path="/path/to/file.txt",
            file_name="file.txt",
            folder_path="/path/to",
            file_size=2048,
            created_at=now,
            updated_at=now,
            total_chunks=5
        )
        
        assert doc.id == "test_123"
        assert doc.file_path == "/path/to/file.txt"
        assert doc.file_name == "file.txt"
        assert doc.folder_path == "/path/to"
        assert doc.file_size == 2048
        assert doc.created_at == now
        assert doc.updated_at == now
        assert doc.total_chunks == 5
    
    def test_chunk_metadata_creation(self):
        """Test ChunkMetadata creation."""
        now = datetime.now()
        chunk = ChunkMetadata(
            id="chunk_456",
            document_id="doc_123",
            chunk_number=2,
            chunk_text="This is chunk text",
            token_count=10,
            overlap_tokens=3,
            previous_chunk_id="chunk_455",
            next_chunk_id="chunk_457",
            created_at=now
        )
        
        assert chunk.id == "chunk_456"
        assert chunk.document_id == "doc_123"
        assert chunk.chunk_number == 2
        assert chunk.chunk_text == "This is chunk text"
        assert chunk.token_count == 10
        assert chunk.overlap_tokens == 3
        assert chunk.previous_chunk_id == "chunk_455"
        assert chunk.next_chunk_id == "chunk_457"
        assert chunk.created_at == now
    
    def test_chunk_metadata_optional_fields(self):
        """Test ChunkMetadata with optional fields."""
        chunk = ChunkMetadata(
            id="chunk_first",
            document_id="doc_123",
            chunk_number=1,
            chunk_text="First chunk",
            token_count=5,
            overlap_tokens=0,
            previous_chunk_id=None,
            next_chunk_id=None,
            created_at=datetime.now()
        )
        
        assert chunk.previous_chunk_id is None
        assert chunk.next_chunk_id is None


if __name__ == '__main__':
    pytest.main([__file__])