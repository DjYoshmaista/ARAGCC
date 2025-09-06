"""
Qdrant vector database manager for the AgenticRAG system.
"""

import asyncio
import uuid as uuid_module
from typing import List, Dict, Any, Optional, Tuple
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from shared.utils.rate_limiter import rate_limiter, RateLimit

logger = logging.getLogger(__name__)

class VectorManager:
    """Manages Qdrant vector database operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None
        
        # Set rate limits - Qdrant: 200 requests per second, burst of 50
        rate_limiter.set_rate_limit("qdrant", RateLimit(requests_per_second=200.0, burst_limit=50))
    
    def connect(self) -> bool:
        """Establish connection to Qdrant"""
        try:
            self.client = QdrantClient(
                host=self.config['host'],
                port=self.config['port'],
                timeout=60
            )
            logger.info("Successfully connected to Qdrant")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            return False
    
    def disconnect(self):
        """Close Qdrant connection"""
        if self.client:
            self.client.close()
            self.client = None
            logger.info("Disconnected from Qdrant")
    
    def create_collection(self, collection_name: str, vector_size: int) -> bool:
        """Create Qdrant collection for vector storage"""
        if not self.client:
            if not self.connect():
                return False
        
        try:
            # Check if collection already exists
            collections = self.client.get_collections()
            existing_names = [c.name for c in collections.collections]
            
            if collection_name in existing_names:
                logger.info(f"Collection '{collection_name}' already exists")
                return True
            
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            logger.info(f"Created Qdrant collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create Qdrant collection: {e}")
            return False
    
    def recreate_collection(self, collection_name: str, vector_size: int) -> bool:
        """Recreate Qdrant collection (deletes existing data)"""
        if not self.client:
            if not self.connect():
                return False
        
        try:
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            logger.info(f"Recreated Qdrant collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to recreate Qdrant collection: {e}")
            return False
    
    def store_vector(self, chunk_id: str, vector: List[float], collection_name: str, 
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store a single vector embedding in Qdrant"""
        if not self.client:
            if not self.connect():
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
            
            # Generate a deterministic UUID for Qdrant point ID from chunk_id
            point_id = str(uuid_module.uuid5(uuid_module.NAMESPACE_DNS, chunk_id))
            
            # Prepare payload
            payload = {"chunk_id": chunk_id}
            if metadata:
                payload.update(metadata)
            
            self.client.upsert(
                collection_name=collection_name,
                points=[PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )]
            )
            logger.debug(f"Stored vector for chunk: {chunk_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store chunk vector {chunk_id}: {e}")
            return False
    
    def store_vectors_batch(self, chunk_vectors: List[Tuple[str, List[float], Optional[Dict[str, Any]]]], 
                           collection_name: str) -> int:
        """
        Store multiple vector embeddings in batch for better performance
        
        Args:
            chunk_vectors: List of (chunk_id, vector, metadata) tuples
            collection_name: Qdrant collection name
            
        Returns:
            Number of vectors successfully stored
        """
        if not self.client:
            if not self.connect():
                return 0
        
        if not chunk_vectors:
            return 0
        
        try:
            # Apply rate limiting - handle case where no event loop exists
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    pass  # Skip rate limiting in running loop
                else:
                    loop.run_until_complete(rate_limiter.acquire("qdrant"))
            except RuntimeError:
                pass  # No event loop - skip rate limiting
            
            points = []
            for chunk_id, vector, metadata in chunk_vectors:
                # Generate deterministic UUID for Qdrant point ID
                point_id = str(uuid_module.uuid5(uuid_module.NAMESPACE_DNS, chunk_id))
                
                # Prepare payload
                payload = {"chunk_id": chunk_id}
                if metadata:
                    payload.update(metadata)
                
                points.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                ))
            
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            logger.debug(f"Stored {len(points)} vectors in batch")
            return len(points)
        except Exception as e:
            logger.error(f"Failed to store batch vectors: {e}")
            return 0
    
    async def store_vector_async(self, chunk_id: str, vector: List[float], 
                                collection_name: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store vector embedding asynchronously with rate limiting"""
        if not self.client:
            if not self.connect():
                return False
        
        try:
            # Apply rate limiting
            await rate_limiter.acquire("qdrant")
            
            # Generate deterministic UUID for Qdrant point ID
            point_id = str(uuid_module.uuid5(uuid_module.NAMESPACE_DNS, chunk_id))
            
            # Prepare payload
            payload = {"chunk_id": chunk_id}
            if metadata:
                payload.update(metadata)
            
            self.client.upsert(
                collection_name=collection_name,
                points=[PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )]
            )
            logger.debug(f"Stored vector for chunk: {chunk_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store chunk vector {chunk_id}: {e}")
            return False
    
    def search_similar(self, query_vector: List[float], collection_name: str, 
                      top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        if not self.client:
            if not self.connect():
                return []
        
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold
            )
            
            return [
                {
                    'chunk_id': result.payload.get('chunk_id'),
                    'score': result.score,
                    'metadata': result.payload
                }
                for result in results
            ]
        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            return []
    
    def search_with_filter(self, query_vector: List[float], collection_name: str,
                          filter_conditions: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors with metadata filtering"""
        if not self.client:
            if not self.connect():
                return []
        
        try:
            # Build filter from conditions
            must_conditions = []
            for field, value in filter_conditions.items():
                must_conditions.append(
                    FieldCondition(key=field, match=MatchValue(value=value))
                )
            
            filter_obj = Filter(must=must_conditions) if must_conditions else None
            
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=filter_obj,
                limit=top_k
            )
            
            return [
                {
                    'chunk_id': result.payload.get('chunk_id'),
                    'score': result.score,
                    'metadata': result.payload
                }
                for result in results
            ]
        except Exception as e:
            logger.error(f"Failed to search vectors with filter: {e}")
            return []
    
    def delete_vector(self, chunk_id: str, collection_name: str) -> bool:
        """Delete a vector by chunk ID"""
        if not self.client:
            if not self.connect():
                return False
        
        try:
            # Generate the same deterministic UUID that was used for storage
            point_id = str(uuid_module.uuid5(uuid_module.NAMESPACE_DNS, chunk_id))
            
            self.client.delete(
                collection_name=collection_name,
                points_selector=[point_id]
            )
            logger.debug(f"Deleted vector for chunk: {chunk_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete vector for chunk {chunk_id}: {e}")
            return False
    
    def count_vectors(self, collection_name: str) -> int:
        """Count total number of vectors in collection"""
        if not self.client:
            if not self.connect():
                return 0
        
        try:
            collection_info = self.client.get_collection(collection_name)
            return collection_info.vectors_count or 0
        except Exception as e:
            logger.error(f"Failed to count vectors: {e}")
            return 0
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a collection"""
        if not self.client:
            if not self.connect():
                return None
        
        try:
            collection_info = self.client.get_collection(collection_name)
            return {
                'vectors_count': collection_info.vectors_count,
                'indexed_vectors_count': collection_info.indexed_vectors_count,
                'points_count': collection_info.points_count,
                'segments_count': collection_info.segments_count,
                'status': str(collection_info.status),
                'optimizer_status': str(collection_info.optimizer_status),
                'disk_data_size': collection_info.disk_data_size,
                'ram_data_size': collection_info.ram_data_size,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None