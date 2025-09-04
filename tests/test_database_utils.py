#!/usr/bin/env python3
"""
Comprehensive unit tests for database utilities
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock
import uuid

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services', 'orchestrator'))

from database_utils import DatabaseManager, DocumentMetadata, ChunkMetadata

class TestDatabaseManager:
    """Test cases for DatabaseManager class"""

    @pytest.fixture
    def db_config(self):
        """Sample database configuration"""
        return {
            'postgresql': {
                'host': 'localhost',
                'port': 5432,
                'database': 'test_db',
                'username': 'test_user',
                'password': 'test_pass'
            },
            'qdrant': {
                'host': 'localhost',
                'port': 6333,
                'collection_name': 'test_embeddings'
            }
        }

    @pytest.fixture
    def db_manager(self, db_config):
        """Create DatabaseManager instance for testing"""
        return DatabaseManager(db_config['postgresql'], db_config['qdrant'])

    @pytest.fixture
    def sample_document(self):
        """Create sample document metadata"""
        return DocumentMetadata(
            id=str(uuid.uuid4()),
            file_path="/test/path/document.txt",
            file_name="document.txt",
            folder_path="/test/path",
            file_size=1024,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            total_chunks=5
        )

    @pytest.fixture
    def sample_chunk(self, sample_document):
        """Create sample chunk metadata"""
        return ChunkMetadata(
            id=str(uuid.uuid4()),
            document_id=sample_document.id,
            chunk_number=0,
            chunk_text="This is a sample chunk of text for testing purposes.",
            token_count=12,
            overlap_tokens=2,
            previous_chunk_id=None,
            next_chunk_id=str(uuid.uuid4()),
            created_at=datetime.utcnow()
        )

    def test_database_manager_initialization(self, db_manager):
        """Test DatabaseManager initialization"""
        assert db_manager.postgresql_config['host'] == 'localhost'
        assert db_manager.qdrant_config['port'] == 6333
        assert db_manager.pg_conn is None
        assert db_manager.qdrant_client is None

    @patch('psycopg2.connect')
    def test_connect_postgresql_success(self, mock_connect, db_manager):
        """Test successful PostgreSQL connection"""
        mock_connect.return_value = MagicMock()
        
        result = db_manager.connect_postgresql()
        
        assert result == True
        assert db_manager.pg_conn is not None
        mock_connect.assert_called_once()

    @patch('psycopg2.connect')
    def test_connect_postgresql_failure(self, mock_connect, db_manager):
        """Test failed PostgreSQL connection"""
        mock_connect.side_effect = Exception("Connection failed")
        
        result = db_manager.connect_postgresql()
        
        assert result == False
        assert db_manager.pg_conn is None

    @patch('qdrant_client.QdrantClient')
    def test_connect_qdrant_success(self, mock_client, db_manager):
        """Test successful Qdrant connection"""
        mock_client.return_value = MagicMock()
        
        result = db_manager.connect_qdrant()
        
        assert result == True
        assert db_manager.qdrant_client is not None
        mock_client.assert_called_once_with(host='localhost', port=6333)

    @patch('qdrant_client.QdrantClient')
    def test_connect_qdrant_failure(self, mock_client, db_manager):
        """Test failed Qdrant connection"""
        mock_client.side_effect = Exception("Connection failed")
        
        result = db_manager.connect_qdrant()
        
        assert result == False
        assert db_manager.qdrant_client is None

    @patch('psycopg2.connect')
    def test_initialize_schemas(self, mock_connect, db_manager):
        """Test database schema initialization"""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        db_manager.pg_conn = mock_conn
        
        result = db_manager.initialize_schemas()
        
        assert result == True
        assert mock_cursor.execute.call_count >= 4  # Should execute multiple schema creation queries

    @patch('qdrant_client.QdrantClient')
    def test_create_qdrant_collection(self, mock_client_class, db_manager):
        """Test Qdrant collection creation"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        db_manager.qdrant_client = mock_client
        
        result = db_manager.create_qdrant_collection("test_collection", 768)
        
        assert result == True
        mock_client.recreate_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_document_async(self, db_manager, sample_document):
        """Test asynchronous document storage"""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        db_manager.pg_conn = mock_conn
        
        with patch('rate_limiter.rate_limiter.acquire', new_callable=AsyncMock):
            result = await db_manager.store_document_async(sample_document)
            
            assert result == True
            mock_cursor.execute.assert_called_once()

    def test_store_chunk(self, db_manager, sample_chunk):
        """Test chunk storage"""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        db_manager.pg_conn = mock_conn
        
        result = db_manager.store_chunk(sample_chunk)
        
        assert result == True
        mock_cursor.execute.assert_called_once()

    def test_store_chunk_vector(self, db_manager):
        """Test chunk vector storage in Qdrant"""
        mock_client = MagicMock()
        db_manager.qdrant_client = mock_client
        
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.run_until_complete.return_value = None
            
            result = db_manager.store_chunk_vector(
                "test-chunk-id", 
                [0.1, 0.2, 0.3], 
                "test_collection"
            )
            
            assert result == True
            mock_client.upsert.assert_called_once()

    def test_get_document(self, db_manager, sample_document):
        """Test document retrieval"""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {
            'id': sample_document.id,
            'file_path': sample_document.file_path,
            'file_name': sample_document.file_name,
            'folder_path': sample_document.folder_path,
            'file_size': sample_document.file_size,
            'created_at': sample_document.created_at,
            'updated_at': sample_document.updated_at,
            'total_chunks': sample_document.total_chunks
        }
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        db_manager.pg_conn = mock_conn
        
        result = db_manager.get_document(sample_document.id)
        
        assert result is not None
        assert result.id == sample_document.id
        assert result.file_name == sample_document.file_name

    def test_get_document_not_found(self, db_manager):
        """Test document retrieval when document doesn't exist"""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        db_manager.pg_conn = mock_conn
        
        result = db_manager.get_document("non-existent-id")
        
        assert result is None

    def test_get_chunks_for_document(self, db_manager, sample_document, sample_chunk):
        """Test chunk retrieval for a document"""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [{
            'id': sample_chunk.id,
            'document_id': sample_chunk.document_id,
            'chunk_number': sample_chunk.chunk_number,
            'chunk_text': sample_chunk.chunk_text,
            'token_count': sample_chunk.token_count,
            'overlap_tokens': sample_chunk.overlap_tokens,
            'previous_chunk_id': sample_chunk.previous_chunk_id,
            'next_chunk_id': sample_chunk.next_chunk_id,
            'created_at': sample_chunk.created_at
        }]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        db_manager.pg_conn = mock_conn
        
        result = db_manager.get_chunks_for_document(sample_document.id)
        
        assert len(result) == 1
        assert result[0].id == sample_chunk.id
        assert result[0].chunk_text == sample_chunk.chunk_text

    def test_close_connections(self, db_manager):
        """Test connection cleanup"""
        mock_pg_conn = MagicMock()
        mock_qdrant_client = MagicMock()
        
        db_manager.pg_conn = mock_pg_conn
        db_manager.qdrant_client = mock_qdrant_client
        
        db_manager.close_connections()
        
        mock_pg_conn.close.assert_called_once()

class TestDocumentMetadata:
    """Test cases for DocumentMetadata dataclass"""

    def test_document_metadata_creation(self):
        """Test DocumentMetadata creation"""
        doc_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        doc = DocumentMetadata(
            id=doc_id,
            file_path="/path/to/file.txt",
            file_name="file.txt",
            folder_path="/path/to",
            file_size=2048,
            created_at=now,
            updated_at=now,
            total_chunks=10
        )
        
        assert doc.id == doc_id
        assert doc.file_name == "file.txt"
        assert doc.file_size == 2048
        assert doc.total_chunks == 10

class TestChunkMetadata:
    """Test cases for ChunkMetadata dataclass"""

    def test_chunk_metadata_creation(self):
        """Test ChunkMetadata creation"""
        chunk_id = str(uuid.uuid4())
        doc_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        chunk = ChunkMetadata(
            id=chunk_id,
            document_id=doc_id,
            chunk_number=1,
            chunk_text="Sample chunk text",
            token_count=15,
            overlap_tokens=3,
            previous_chunk_id=None,
            next_chunk_id=str(uuid.uuid4()),
            created_at=now
        )
        
        assert chunk.id == chunk_id
        assert chunk.document_id == doc_id
        assert chunk.chunk_number == 1
        assert chunk.token_count == 15

if __name__ == "__main__":
    pytest.main([__file__, "-v"])