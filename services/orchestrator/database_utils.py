"""
Database utilities for the AgenticRAG system
Supports both PostgreSQL and Qdrant databases
"""

import asyncio
import psycopg2
from psycopg2.extras import RealDictCursor
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any, Optional
import uuid
import os
from dataclasses import dataclass, asdict
from datetime import datetime

# Add the current directory to Python path for local imports
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Use absolute imports
from rate_limiter import rate_limiter, RateLimit

@dataclass
class DocumentMetadata:
    """Metadata for a document"""
    id: str
    file_path: str
    file_name: str
    folder_path: str
    file_size: int
    created_at: datetime
    updated_at: datetime
    total_chunks: int

@dataclass
class ChunkMetadata:
    """Metadata for a chunk"""
    id: str
    document_id: str
    chunk_number: int
    chunk_text: str
    token_count: int
    overlap_tokens: int
    previous_chunk_id: Optional[str]
    next_chunk_id: Optional[str]
    created_at: datetime

class DatabaseManager:
    """Manages connections to both PostgreSQL and Qdrant databases"""
    
    def __init__(self, postgresql_config: Dict[str, Any], qdrant_config: Dict[str, Any]):
        self.postgresql_config = postgresql_config
        self.qdrant_config = qdrant_config
        self.pg_conn = None
        self.qdrant_client = None
        
        # Set rate limits
        # PostgreSQL: 100 requests per second, burst of 20
        rate_limiter.set_rate_limit("postgresql", RateLimit(requests_per_second=100.0, burst_limit=20))
        
        # Qdrant: 200 requests per second, burst of 50
        rate_limiter.set_rate_limit("qdrant", RateLimit(requests_per_second=200.0, burst_limit=50))
    
    def connect_postgresql(self):
        """Establish connection to PostgreSQL"""
        try:
            self.pg_conn = psycopg2.connect(
                host=self.postgresql_config['host'],
                port=self.postgresql_config['port'],
                database=self.postgresql_config['database'],
                user=self.postgresql_config['username'],
                password=self.postgresql_config['password']
            )
            return True
        except Exception as e:
            print(f"Failed to connect to PostgreSQL: {e}")
            return False
    
    def connect_qdrant(self):
        """Establish connection to Qdrant"""
        try:
            self.qdrant_client = QdrantClient(
                host=self.qdrant_config['host'],
                port=self.qdrant_config['port']
            )
            return True
        except Exception as e:
            print(f"Failed to connect to Qdrant: {e}")
            return False
    
    def initialize_schemas(self):
        """Initialize database schemas for document and chunk storage"""
        if not self.pg_conn:
            if not self.connect_postgresql():
                return False
        
        try:
            with self.pg_conn.cursor() as cursor:
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
                
                # Create indexes
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_document_id 
                    ON chunks(document_id)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_chunk_number 
                    ON chunks(chunk_number)
                """)
                
                self.pg_conn.commit()
                return True
        except Exception as e:
            print(f"Failed to initialize schemas: {e}")
            self.pg_conn.rollback()
            return False
    
    def create_qdrant_collection(self, collection_name: str, vector_size: int):
        """Create Qdrant collection for vector storage"""
        if not self.qdrant_client:
            if not self.connect_qdrant():
                return False
        
        try:
            self.qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            return True
        except Exception as e:
            print(f"Failed to create Qdrant collection: {e}")
            return False
    
    def store_document(self, document: DocumentMetadata) -> bool:
        """Store document metadata in PostgreSQL synchronously"""
        if not self.pg_conn:
            if not self.connect_postgresql():
                return False
        
        try:
            with self.pg_conn.cursor() as cursor:
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
                
                self.pg_conn.commit()
                return True
        except Exception as e:
            print(f"Failed to store document: {e}")
            self.pg_conn.rollback()
            return False

    async def store_document_async(self, document: DocumentMetadata) -> bool:
        """Store document metadata in PostgreSQL asynchronously"""
        if not self.pg_conn:
            if not self.connect_postgresql():
                return False
        
        try:
            # Apply rate limiting
            await rate_limiter.acquire("postgresql")
            
            with self.pg_conn.cursor() as cursor:
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
                
                self.pg_conn.commit()
                return True
        except Exception as e:
            print(f"Failed to store document: {e}")
            self.pg_conn.rollback()
            return False
    
    def store_chunk(self, chunk: ChunkMetadata) -> bool:
        """Store chunk metadata in PostgreSQL"""
        if not self.pg_conn:
            if not self.connect_postgresql():
                return False
        
        try:
            with self.pg_conn.cursor() as cursor:
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
                
                self.pg_conn.commit()
                return True
        except Exception as e:
            print(f"Failed to store chunk: {e}")
            self.pg_conn.rollback()
            return False
    
    def store_chunk_vector(self, chunk_id: str, vector: List[float], collection_name: str) -> bool:
        """Store chunk vector embedding in Qdrant"""
        if not self.qdrant_client:
            if not self.connect_qdrant():
                return False
        
        try:
            # Apply rate limiting - handle case where no event loop exists
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, we can't use run_until_complete
                    # Skip rate limiting for now (could use asyncio.create_task instead)
                    pass
                else:
                    loop.run_until_complete(rate_limiter.acquire("qdrant"))
            except RuntimeError:
                # No event loop exists - skip rate limiting for this thread
                # This is acceptable for embedding result worker threads
                pass
            
            # Generate a valid UUID for Qdrant point ID
            import uuid as uuid_module
            point_id = str(uuid_module.uuid5(uuid_module.NAMESPACE_DNS, chunk_id))
            
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=[PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={"chunk_id": chunk_id}  # Keep original chunk_id in payload
                )]
            )
            return True
        except Exception as e:
            print(f"Failed to store chunk vector: {e}")
            return False
    
    def store_chunk_vectors_batch(self, chunk_vectors: List[tuple], collection_name: str) -> int:
        """
        Store multiple chunk vector embeddings in Qdrant using batch operations for better performance
        
        Args:
            chunk_vectors: List of (chunk_id, vector) tuples
            collection_name: Qdrant collection name
            
        Returns:
            Number of successfully stored vectors
        """
        if not self.qdrant_client:
            if not self.connect_qdrant():
                return 0
        
        if not chunk_vectors:
            return 0
        
        try:
            # Apply rate limiting for batch operation
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    pass
                else:
                    loop.run_until_complete(rate_limiter.acquire("qdrant"))
            except RuntimeError:
                pass
            
            # Prepare batch points
            points = []
            import uuid as uuid_module
            
            for chunk_id, vector in chunk_vectors:
                if not vector or len(vector) == 0:
                    continue  # Skip invalid vectors
                    
                point_id = str(uuid_module.uuid5(uuid_module.NAMESPACE_DNS, chunk_id))
                points.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={"chunk_id": chunk_id}
                ))
            
            if not points:
                return 0
            
            # Batch upsert - much more efficient than individual operations
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True  # Wait for completion to ensure data consistency
            )
            
            return len(points)
            
        except Exception as e:
            print(f"Failed to store chunk vectors batch: {e}")
            return 0
    
    def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        """Retrieve document metadata from PostgreSQL"""
        if not self.pg_conn:
            if not self.connect_postgresql():
                return None
        
        try:
            with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM documents WHERE id = %s
                """, (document_id,))
                
                result = cursor.fetchone()
                if result:
                    return DocumentMetadata(**result)
                return None
        except Exception as e:
            print(f"Failed to retrieve document: {e}")
            return None
    
    def get_chunks_for_document(self, document_id: str) -> List[ChunkMetadata]:
        """Retrieve all chunks for a document from PostgreSQL"""
        if not self.pg_conn:
            if not self.connect_postgresql():
                return []
        
        try:
            with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM chunks WHERE document_id = %s ORDER BY chunk_number
                """, (document_id,))
                
                results = cursor.fetchall()
                return [ChunkMetadata(**result) for result in results]
        except Exception as e:
            print(f"Failed to retrieve chunks: {e}")
            return []
    
    def close_connections(self):
        """Close database connections"""
        if self.pg_conn:
            self.pg_conn.close()
        if self.qdrant_client:
            # Qdrant client doesn't need explicit closing
            pass