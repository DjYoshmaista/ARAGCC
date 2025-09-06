"""
PostgreSQL database manager for the AgenticRAG system.

Provides thread-safe PostgreSQL operations with proper connection pooling,
error handling, and rate limiting for document and chunk storage.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import logging
import contextlib
from dataclasses import asdict

from shared.utils.rate_limiter import rate_limiter, RateLimit
from .schemas import DocumentMetadata, ChunkMetadata

logger = logging.getLogger(__name__)

class PostgreSQLManager:
    """
    Manages PostgreSQL database operations with connection pooling and rate limiting.
    
    Provides thread-safe database operations for document and chunk storage
    with automatic connection management, error handling, and performance optimization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PostgreSQL manager with configuration.
        
        Args:
            config: Database configuration containing host, port, database,
                   username, password, and optional connection pool settings
        """
        self.config = config
        self.connection = None
        self.connection_pool: Optional[SimpleConnectionPool] = None
        self._pool_min_conn = config.get('pool_min_conn', 1)
        self._pool_max_conn = config.get('pool_max_conn', 10)
        
        # Set rate limits - PostgreSQL: 100 requests per second, burst of 20
        rate_limiter.set_rate_limit("postgresql", RateLimit(requests_per_second=100.0, burst_limit=20))
    
    def connect(self) -> bool:
        """
        Establish connection to PostgreSQL with connection pooling support.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.config.get('use_connection_pool', True):
                # Initialize connection pool
                self.connection_pool = SimpleConnectionPool(
                    self._pool_min_conn,
                    self._pool_max_conn,
                    host=self.config['host'],
                    port=self.config['port'],
                    database=self.config['database'],
                    user=self.config['username'],
                    password=self.config['password']
                )
                logger.info(f"Successfully created PostgreSQL connection pool "
                           f"(min={self._pool_min_conn}, max={self._pool_max_conn})")
            else:
                # Single connection
                self.connection = psycopg2.connect(
                    host=self.config['host'],
                    port=self.config['port'],
                    database=self.config['database'],
                    user=self.config['username'],
                    password=self.config['password']
                )
                logger.info("Successfully connected to PostgreSQL")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            return False
    
    def disconnect(self):
        """Close PostgreSQL connections and clean up resources"""
        if self.connection_pool:
            self.connection_pool.closeall()
            self.connection_pool = None
            logger.info("Closed PostgreSQL connection pool")
        
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Disconnected from PostgreSQL")
    
    @contextlib.contextmanager
    def get_connection(self):
        """
        Context manager to get a database connection.
        
        Uses connection pool if available, otherwise uses single connection.
        Handles connection cleanup automatically.
        """
        connection = None
        try:
            if self.connection_pool:
                connection = self.connection_pool.getconn()
            else:
                if not self.connection:
                    if not self.connect():
                        raise psycopg2.Error("Failed to establish database connection")
                connection = self.connection
            
            yield connection
        finally:
            if self.connection_pool and connection:
                self.connection_pool.putconn(connection)
    
    def initialize_schema(self) -> bool:
        """
        Initialize database schema for document and chunk storage.
        
        Creates tables and indexes if they don't exist.
        
        Returns:
            True if schema initialization successful, False otherwise
        """
        try:
            with self.get_connection() as connection:
                with connection.cursor() as cursor:
                    # Create documents table
                    cursor.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id VARCHAR(255) PRIMARY KEY,
                        file_path TEXT NOT NULL,
                        file_name VARCHAR(255) NOT NULL,
                        folder_path TEXT NOT NULL,
                        file_size INTEGER NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL,
                        total_chunks INTEGER NOT NULL DEFAULT 0
                    )
                """)
                    
                    # Create chunks table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS chunks (
                            id VARCHAR(255) PRIMARY KEY,
                            document_id VARCHAR(255) NOT NULL REFERENCES documents(id),
                            chunk_number INTEGER NOT NULL,
                            chunk_text TEXT NOT NULL,
                            token_count INTEGER NOT NULL,
                            overlap_tokens INTEGER NOT NULL DEFAULT 0,
                            previous_chunk_id VARCHAR(255),
                            next_chunk_id VARCHAR(255),
                            created_at TIMESTAMP NOT NULL
                        )
                    """)
                    
                    # Create indexes for better performance
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_chunks_document_id 
                        ON chunks(document_id)
                    """)
                    
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_chunks_chunk_number 
                        ON chunks(chunk_number)
                    """)
                    
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_documents_file_path 
                        ON documents(file_path)
                    """)
                    
                    connection.commit()
                    logger.info("Database schema initialized successfully")
                    return True
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            return False
    
    async def store_document_async(self, document: DocumentMetadata) -> bool:
        """
        Store document metadata asynchronously with rate limiting.
        
        Args:
            document: Document metadata to store
            
        Returns:
            True if storage successful, False otherwise
        """
        try:
            # Apply rate limiting
            await rate_limiter.acquire("postgresql")
            
            with self.get_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO documents (
                            id, file_path, file_name, folder_path, 
                            file_size, created_at, updated_at, total_chunks
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            updated_at = EXCLUDED.updated_at,
                            total_chunks = EXCLUDED.total_chunks
                    """, (
                        document.id, document.file_path, document.file_name,
                        document.folder_path, document.file_size,
                        document.created_at, document.updated_at, document.total_chunks
                    ))
                    
                    connection.commit()
                    logger.debug(f"Stored document: {document.id}")
                    return True
        except Exception as e:
            logger.error(f"Failed to store document {document.id}: {e}")
            return False
    
    def store_document(self, document: DocumentMetadata) -> bool:
        """
        Store document metadata synchronously.
        
        Args:
            document: Document metadata to store
            
        Returns:
            True if storage successful, False otherwise
        """
        try:
            with self.get_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO documents (
                            id, file_path, file_name, folder_path, 
                            file_size, created_at, updated_at, total_chunks
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            updated_at = EXCLUDED.updated_at,
                            total_chunks = EXCLUDED.total_chunks
                    """, (
                        document.id, document.file_path, document.file_name,
                        document.folder_path, document.file_size,
                        document.created_at, document.updated_at, document.total_chunks
                    ))
                    
                    connection.commit()
                    logger.debug(f"Stored document: {document.id}")
                    return True
        except Exception as e:
            logger.error(f"Failed to store document {document.id}: {e}")
            return False
    
    def store_chunk(self, chunk: ChunkMetadata) -> bool:
        """
        Store chunk metadata.
        
        Args:
            chunk: Chunk metadata to store
            
        Returns:
            True if storage successful, False otherwise
        """
        try:
            with self.get_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO chunks (
                            id, document_id, chunk_number, chunk_text,
                            token_count, overlap_tokens, previous_chunk_id, next_chunk_id, created_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            chunk_text = EXCLUDED.chunk_text,
                            token_count = EXCLUDED.token_count
                    """, (
                        chunk.id, chunk.document_id, chunk.chunk_number, chunk.chunk_text,
                        chunk.token_count, chunk.overlap_tokens, 
                        chunk.previous_chunk_id, chunk.next_chunk_id, chunk.created_at
                    ))
                    
                    connection.commit()
                    logger.debug(f"Stored chunk: {chunk.id}")
                    return True
        except Exception as e:
            logger.error(f"Failed to store chunk {chunk.id}: {e}")
            return False
    
    def store_chunks_batch(self, chunks: List[ChunkMetadata]) -> int:
        """Store multiple chunks in a batch for better performance"""
        if not self.connection:
            if not self.connect():
                return 0
        
        stored_count = 0
        try:
            with self.connection.cursor() as cursor:
                for chunk in chunks:
                    try:
                        cursor.execute("""
                            INSERT INTO chunks (
                                id, document_id, chunk_number, chunk_text,
                                token_count, overlap_tokens, previous_chunk_id, next_chunk_id, created_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO UPDATE SET
                                chunk_text = EXCLUDED.chunk_text,
                                token_count = EXCLUDED.token_count
                        """, (
                            chunk.id, chunk.document_id, chunk.chunk_number, chunk.chunk_text,
                            chunk.token_count, chunk.overlap_tokens, 
                            chunk.previous_chunk_id, chunk.next_chunk_id, chunk.created_at
                        ))
                        stored_count += 1
                    except Exception as e:
                        logger.error(f"Failed to store chunk {chunk.id}: {e}")
                
                self.connection.commit()
                logger.debug(f"Stored {stored_count}/{len(chunks)} chunks in batch")
                return stored_count
        except Exception as e:
            logger.error(f"Batch chunk storage failed: {e}")
            if self.connection:
                self.connection.rollback()
            return stored_count
    
    def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        """Retrieve document metadata by ID"""
        if not self.connection:
            if not self.connect():
                return None
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM documents WHERE id = %s", (document_id,))
                row = cursor.fetchone()
                if row:
                    return DocumentMetadata(**dict(row))
                return None
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None
    
    def get_chunks_by_document(self, document_id: str) -> List[ChunkMetadata]:
        """Retrieve all chunks for a document"""
        if not self.connection:
            if not self.connect():
                return []
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM chunks 
                    WHERE document_id = %s 
                    ORDER BY chunk_number
                """, (document_id,))
                rows = cursor.fetchall()
                return [ChunkMetadata(**dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get chunks for document {document_id}: {e}")
            return []
    
    def count_documents(self) -> int:
        """Count total number of documents"""
        if not self.connection:
            if not self.connect():
                return 0
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM documents")
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to count documents: {e}")
            return 0
    
    def count_chunks(self) -> int:
        """Count total number of chunks"""
        if not self.connection:
            if not self.connect():
                return 0
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM chunks")
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to count chunks: {e}")
            return 0