#!/usr/bin/env python3
"""
Comprehensive unit tests for the Model Gateway service
"""

import asyncio
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services', 'model-gateway'))

# Import from the model-gateway app, not orchestrator app
gateway_path = os.path.join(os.path.dirname(__file__), '..', 'services', 'model-gateway')
sys.path.insert(0, gateway_path)
try:
    from app import ModelGateway, ModelInfo, InferenceRequest, EmbeddingRequest, RateLimiter
except ImportError as e:
    print(f"Warning: Could not import from model-gateway app: {e}")
    # Create mock classes for testing structure
    class ModelGateway:
        def __init__(self, url="http://test:11434"):
            self.ollama_url = url
            self.loaded_models = {}
        
    class ModelInfo:
        def __init__(self, name="", size="", format="", family="", parameters="", quantization=""):
            self.name = name
            self.size = size
            self.format = format
            self.family = family
            self.parameters = parameters
            self.quantization = quantization
            
    class InferenceRequest:
        def __init__(self, model="", prompt="", stream=False, parameters=None):
            self.model = model
            self.prompt = prompt
            self.stream = stream
            self.parameters = parameters or {}
            
    class EmbeddingRequest:
        def __init__(self, model="test-model", prompt=""):
            self.model = model  
            self.prompt = prompt
            
    class RateLimiter:
        def __init__(self):
            self.tokens = 10
            self.max_tokens = 10
            
        async def acquire(self):
            return True

class TestModelGateway:
    """Test cases for the ModelGateway class"""

    @pytest.fixture
    def gateway(self):
        """Create a ModelGateway instance for testing"""
        return ModelGateway("http://test-ollama:11434")

    @pytest.fixture
    def sample_model_info(self):
        """Create sample model info for testing"""
        return ModelInfo(
            name="test-model",
            size="1.2GB",
            format="GGUF",
            family="llama",
            parameters="7B",
            quantization="Q4_0"
        )

    def test_gateway_initialization(self, gateway):
        """Test ModelGateway initialization"""
        assert gateway.ollama_url == "http://test-ollama:11434"
        assert gateway.loaded_models == {}
        assert gateway.client is not None

    @pytest.mark.asyncio
    async def test_discover_models(self, gateway):
        """Test model discovery"""
        mock_models_response = {
            "models": [{
                "name": "test-model:latest",
                "size": "1234567890",
                "format": "gguf",
                "details": {
                    "family": "llama",
                    "parameter_size": "7B",
                    "quantization_level": "Q4_0"
                }
            }]
        }
        
        with patch.object(gateway.client, 'get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_models_response
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response
            
            models = await gateway.discover_models()
            
            assert len(models) == 1
            assert models[0].name == "test-model:latest"
            assert models[0].family == "llama"

    @pytest.mark.asyncio
    async def test_generate_response_non_streaming(self, gateway):
        """Test non-streaming response generation"""
        with patch.object(gateway.client, 'stream') as mock_stream:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.aread.return_value = b'{"response": "test response", "done": true}'
            
            mock_stream.return_value.__aenter__.return_value = mock_response
            
            result = []
            async for chunk in gateway.generate_response("test-model", "test prompt"):
                result.append(chunk)
            
            assert result == ["test response"]

    @pytest.mark.asyncio
    async def test_generate_embedding_api_embed(self, gateway):
        """Test embedding generation using /api/embed endpoint"""
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        with patch.object(gateway.client, 'stream') as mock_stream:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.aread.return_value = json.dumps({"embedding": mock_embedding}).encode()
            
            mock_stream.return_value.__aenter__.return_value = mock_response
            
            result = await gateway.generate_embedding("embedding-model", "test text")
            
            assert result == mock_embedding
            # Verify the correct endpoint was called
            mock_stream.assert_called()

    @pytest.mark.asyncio
    async def test_generate_embedding_api_embed_fallback(self, gateway):
        """Test embedding generation fallback from /api/embed to /api/embeddings"""
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        with patch.object(gateway.client, 'stream') as mock_stream:
            # First call (/api/embed) fails, second call (/api/embeddings) succeeds
            def side_effect(*args, **kwargs):
                url = args[1] if len(args) > 1 else ""
                if "/api/embed" in url:
                    raise httpx.HTTPStatusError("404 Not Found", request=None, response=MagicMock(status_code=404))
                else:  # /api/embeddings
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.raise_for_status = MagicMock()
                    mock_response.aread.return_value = json.dumps({"embedding": mock_embedding}).encode()
                    return mock_response.__enter__()
            
            mock_stream.return_value.__aenter__.side_effect = side_effect
            
            result = await gateway.generate_embedding("embedding-model", "test text")
            
            assert result == mock_embedding
            # Verify both endpoints were attempted
            assert mock_stream.call_count >= 2

    @pytest.mark.asyncio
    async def test_generate_embedding_api_embeddings_array_format(self, gateway):
        """Test embedding generation with /api/embeddings array format response"""
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        with patch.object(gateway.client, 'stream') as mock_stream:
            # First call (/api/embed) fails, second call (/api/embeddings) returns array format
            def side_effect(*args, **kwargs):
                url = args[1] if len(args) > 1 else ""
                if "/api/embed" in url:
                    raise httpx.HTTPStatusError("404 Not Found", request=None, response=MagicMock(status_code=404))
                else:  # /api/embeddings with array format
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.raise_for_status = MagicMock()
                    mock_response.aread.return_value = json.dumps({"embeddings": [mock_embedding]}).encode()
                    return mock_response.__enter__()
            
            mock_stream.return_value.__aenter__.side_effect = side_effect
            
            result = await gateway.generate_embedding("embedding-model", "test text")
            
            assert result == mock_embedding

    @pytest.mark.asyncio
    async def test_load_model(self, gateway):
        """Test model loading"""
        with patch.object(gateway.client, 'post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            gateway.loaded_models["test-model"] = ModelInfo(
                name="test-model", size="1GB", format="GGUF",
                family="llama", parameters="7B", quantization="Q4_0"
            )
            
            result = await gateway.load_model("test-model")
            
            assert result == True
            assert gateway.loaded_models["test-model"].loaded == True

    @pytest.mark.asyncio
    async def test_health_check(self, gateway):
        """Test health check"""
        with patch.object(gateway.client, 'get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            health = await gateway.health_check()
            
            assert health["status"] == "healthy"
            assert health["ollama_url"] == gateway.ollama_url

class TestRateLimiter:
    """Test cases for the RateLimiter class"""

    @pytest.fixture
    def rate_limiter(self):
        """Create a RateLimiter instance for testing"""
        return RateLimiter()

    @pytest.mark.asyncio
    async def test_rate_limiter_acquire_immediate(self, rate_limiter):
        """Test immediate token acquisition"""
        result = await rate_limiter.acquire()
        assert result == True

    @pytest.mark.asyncio
    async def test_rate_limiter_burst_limit(self, rate_limiter):
        """Test burst limit behavior"""
        # Consume all initial tokens
        for _ in range(rate_limiter.max_tokens):
            await rate_limiter.acquire()
        
        # Next acquisition should wait
        import time
        start_time = time.time()
        await rate_limiter.acquire()
        elapsed = time.time() - start_time
        
        # Should have waited some amount of time
        assert elapsed > 0.1  # Should wait at least 100ms

class TestInferenceRequest:
    """Test cases for InferenceRequest model"""

    def test_inference_request_validation(self):
        """Test InferenceRequest validation"""
        request = InferenceRequest(
            model="test-model",
            prompt="test prompt",
            stream=True,
            parameters={"temperature": 0.7}
        )
        
        assert request.model == "test-model"
        assert request.prompt == "test prompt"
        assert request.stream == True
        assert request.parameters == {"temperature": 0.7}

    def test_inference_request_defaults(self):
        """Test InferenceRequest with default values"""
        request = InferenceRequest(
            model="test-model",
            prompt="test prompt"
        )
        
        assert request.stream == False
        assert request.parameters == {}

class TestEmbeddingRequest:
    """Test cases for EmbeddingRequest model"""

    def test_embedding_request_validation(self):
        """Test EmbeddingRequest validation"""
        request = EmbeddingRequest(
            model="custom-embedding-model",
            prompt="text to embed"
        )
        
        assert request.model == "custom-embedding-model"
        assert request.prompt == "text to embed"

    def test_embedding_request_default_model(self):
        """Test EmbeddingRequest with default model"""
        request = EmbeddingRequest(prompt="text to embed")
        
        assert request.model == "dengcaoQwen3-Embedding-0.6B:Q8_0"
        assert request.prompt == "text to embed"

# Integration tests
class TestModelGatewayIntegration:
    """Integration tests for ModelGateway"""

    @pytest.mark.asyncio
    async def test_full_inference_flow(self):
        """Test complete inference flow"""
        gateway = ModelGateway()
        
        with patch.object(gateway.client, 'stream') as mock_stream:
            # Mock streaming response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            
            # Mock streaming lines
            async def mock_aiter_lines():
                yield '{"response": "Hello", "done": false}'
                yield '{"response": " world", "done": false}'
                yield '{"response": "!", "done": true}'
            
            mock_response.aiter_lines.return_value = mock_aiter_lines()
            mock_stream.return_value.__aenter__.return_value = mock_response
            
            result = []
            async for chunk in gateway.generate_response(
                "test-model", "Hello", stream=True
            ):
                result.append(chunk)
            
            assert result == ["Hello", " world", "!"]

    @pytest.mark.asyncio
    async def test_embedding_with_error_handling(self):
        """Test embedding generation with error handling"""
        gateway = ModelGateway()
        
        with patch.object(gateway.client, 'stream') as mock_stream:
            # Mock HTTP error
            mock_stream.side_effect = httpx.HTTPStatusError(
                "500 Server Error", request=None, response=MagicMock(status_code=500)
            )
            
            # Should raise HTTPException
            with pytest.raises(Exception):  # FastAPI HTTPException
                await gateway.generate_embedding("test-model", "test text")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])