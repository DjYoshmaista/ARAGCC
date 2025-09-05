"""
Embedding generator for the AgenticRAG system.
Handles embedding generation using various models.
"""

import asyncio
import httpx
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    High-level embedding generator that interfaces with model gateways
    to generate embeddings for text chunks.
    """
    
    def __init__(self, model_gateway_url: str, default_model: str = "granite-embedding:latest"):
        self.model_gateway_url = model_gateway_url
        self.default_model = default_model
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def close(self):
        """Close HTTP client"""
        if self.client:
            await self.client.aclose()
    
    async def generate_embedding(self, text: str, model: str = None) -> Optional[List[float]]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            model: Model to use (defaults to default_model)
            
        Returns:
            List of floats representing the embedding vector, or None on failure
        """
        model = model or self.default_model
        
        try:
            response = await self.client.post(
                f"{self.model_gateway_url}/embed",
                json={
                    "model": model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            data = response.json()
            
            embedding = data.get("embedding")
            if embedding:
                logger.debug(f"Generated embedding with {len(embedding)} dimensions for text length {len(text)}")
                return embedding
            else:
                logger.error("No embedding found in response")
                return None
                
        except httpx.RequestError as e:
            logger.error(f"Network error during embedding generation: {e}")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during embedding generation: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during embedding generation: {e}")
            return None
    
    async def generate_embeddings_batch(self, texts: List[str], model: str = None) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in parallel
        
        Args:
            texts: List of texts to embed
            model: Model to use (defaults to default_model)
            
        Returns:
            List of embedding vectors (or None for failed embeddings)
        """
        tasks = [
            self.generate_embedding(text, model)
            for text in texts
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to None
        embeddings = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Embedding generation failed: {result}")
                embeddings.append(None)
            else:
                embeddings.append(result)
        
        success_count = sum(1 for e in embeddings if e is not None)
        logger.info(f"Generated {success_count}/{len(texts)} embeddings successfully")
        
        return embeddings
    
    async def generate_embedding_with_retry(self, text: str, model: str = None, 
                                          max_retries: int = 3, base_delay: float = 1.0) -> Optional[List[float]]:
        """
        Generate embedding with exponential backoff retry
        
        Args:
            text: Text to embed
            model: Model to use
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries (exponentially increased)
            
        Returns:
            Embedding vector or None if all retries failed
        """
        for attempt in range(max_retries + 1):
            try:
                result = await self.generate_embedding(text, model)
                if result is not None:
                    if attempt > 0:
                        logger.info(f"Embedding generation succeeded on attempt {attempt + 1}")
                    return result
                    
            except Exception as e:
                logger.warning(f"Embedding generation attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.debug(f"Retrying after {delay}s delay...")
                await asyncio.sleep(delay)
        
        logger.error(f"All {max_retries + 1} embedding generation attempts failed")
        return None
    
    def estimate_embedding_time(self, text_length: int) -> float:
        """
        Estimate embedding generation time based on text length
        
        Args:
            text_length: Length of text in characters
            
        Returns:
            Estimated time in seconds
        """
        # Simple heuristic: ~0.01 seconds per 100 characters
        base_time = 0.5  # Base processing time
        variable_time = text_length / 10000  # Variable time based on length
        return base_time + variable_time
    
    async def health_check(self) -> bool:
        """
        Check if the embedding service is healthy
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            # Test with a small sample text
            test_text = "This is a test."
            embedding = await self.generate_embedding(test_text)
            
            if embedding and len(embedding) > 0:
                logger.debug("Embedding service health check passed")
                return True
            else:
                logger.warning("Embedding service health check failed - no embedding returned")
                return False
                
        except Exception as e:
            logger.error(f"Embedding service health check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current embedding model
        
        Returns:
            Dictionary with model information
        """
        return {
            'model': self.default_model,
            'gateway_url': self.model_gateway_url,
            'estimated_dimensions': 384  # Default for granite-embedding
        }