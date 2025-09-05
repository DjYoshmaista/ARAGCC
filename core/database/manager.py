"""
Unified database manager that coordinates PostgreSQL and Qdrant operations.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging

from .postgresql_manager import PostgreSQLManager
from .vector_manager import VectorManager
from .schemas import DocumentMetadata, ChunkMetadata

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Unified database manager that provides a single interface to both
    PostgreSQL (for metadata) and Qdrant (for vectors) databases.
    """
    
    def __init__(self, postgresql_config: Dict[str, Any], qdrant_config: Dict[str, Any]):
        self.postgresql = PostgreSQLManager(postgresql_config)
        self.vector = VectorManager(qdrant_config)
        self.qdrant_config = qdrant_config
        
    def initialize_databases(self, collection_name: str = None, vector_dimensions: int = 384) -> bool:
        """Initialize both PostgreSQL schema and Qdrant collection"""
        collection_name = collection_name or self.qdrant_config.get('collection_name', 'embeddings')
        
        # Initialize PostgreSQL schema
        postgres_success = self.postgresql.initialize_schema()
        
        # Initialize Qdrant collection
        vector_success = self.vector.create_collection(collection_name, vector_dimensions)
        
        if postgres_success and vector_success:
            logger.info("Both databases initialized successfully")
            return True
        else:
            logger.error(f"Database initialization failed - PostgreSQL: {postgres_success}, Qdrant: {vector_success}")
            return False
    
    def connect(self) -> bool:
        """Connect to both databases"""
        postgres_connected = self.postgresql.connect()
        vector_connected = self.vector.connect()
        
        return postgres_connected and vector_connected
    
    def disconnect(self):
        """Disconnect from both databases"""
        self.postgresql.disconnect()
        self.vector.disconnect()
    
    # Document operations
    def store_document(self, document: DocumentMetadata) -> bool:
        """Store document metadata in PostgreSQL"""
        return self.postgresql.store_document(document)
    
    async def store_document_async(self, document: DocumentMetadata) -> bool:
        """Store document metadata asynchronously"""
        return await self.postgresql.store_document_async(document)
    
    def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        """Retrieve document metadata"""
        return self.postgresql.get_document(document_id)
    
    # Chunk operations
    def store_chunk(self, chunk: ChunkMetadata) -> bool:
        """Store chunk metadata in PostgreSQL"""
        return self.postgresql.store_chunk(chunk)
    
    def store_chunks_batch(self, chunks: List[ChunkMetadata]) -> int:
        """Store multiple chunks in batch"""
        return self.postgresql.store_chunks_batch(chunks)
    
    def get_chunks_by_document(self, document_id: str) -> List[ChunkMetadata]:
        """Get all chunks for a document"""
        return self.postgresql.get_chunks_by_document(document_id)
    
    # Vector operations
    def store_chunk_vector(self, chunk_id: str, vector: List[float], 
                          collection_name: str = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store chunk vector in Qdrant"""
        collection_name = collection_name or self.qdrant_config.get('collection_name', 'embeddings')
        return self.vector.store_vector(chunk_id, vector, collection_name, metadata)
    
    async def store_chunk_vector_async(self, chunk_id: str, vector: List[float], 
                                      collection_name: str = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store chunk vector asynchronously"""
        collection_name = collection_name or self.qdrant_config.get('collection_name', 'embeddings')
        return await self.vector.store_vector_async(chunk_id, vector, collection_name, metadata)
    
    def store_chunk_vectors_batch(self, chunk_vectors: List[Tuple[str, List[float], Optional[Dict[str, Any]]]], 
                                 collection_name: str = None) -> int:
        """Store multiple chunk vectors in batch"""
        collection_name = collection_name or self.qdrant_config.get('collection_name', 'embeddings')
        return self.vector.store_vectors_batch(chunk_vectors, collection_name)
    
    # Search operations
    def search_similar_chunks(self, query_vector: List[float], top_k: int = 5, 
                             collection_name: str = None, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity"""
        collection_name = collection_name or self.qdrant_config.get('collection_name', 'embeddings')
        return self.vector.search_similar(query_vector, collection_name, top_k, score_threshold)
    
    def search_chunks_with_filter(self, query_vector: List[float], filter_conditions: Dict[str, Any],
                                 top_k: int = 5, collection_name: str = None) -> List[Dict[str, Any]]:
        """Search for similar chunks with metadata filtering"""
        collection_name = collection_name or self.qdrant_config.get('collection_name', 'embeddings')
        return self.vector.search_with_filter(query_vector, collection_name, filter_conditions, top_k)
    
    # Statistics and info
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        collection_name = self.qdrant_config.get('collection_name', 'embeddings')
        
        return {
            'documents_count': self.postgresql.count_documents(),
            'chunks_count': self.postgresql.count_chunks(),
            'vectors_count': self.vector.count_vectors(collection_name),
            'collection_info': self.vector.get_collection_info(collection_name)
        }
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of both databases"""
        postgres_healthy = False
        vector_healthy = False
        
        try:
            # Test PostgreSQL connection
            if self.postgresql.connection or self.postgresql.connect():
                docs_count = self.postgresql.count_documents()
                postgres_healthy = docs_count >= 0
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
        
        try:
            # Test Qdrant connection
            if self.vector.client or self.vector.connect():
                collection_name = self.qdrant_config.get('collection_name', 'embeddings')
                vectors_count = self.vector.count_vectors(collection_name)
                vector_healthy = vectors_count >= 0
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
        
        return {
            'postgresql': postgres_healthy,
            'qdrant': vector_healthy,
            'overall': postgres_healthy and vector_healthy
        }