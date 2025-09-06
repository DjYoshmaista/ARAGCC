"""
Comprehensive unit tests for the models gateway module.

Tests cover request/response validation, model management, load balancing,
error handling, and API endpoints.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import httpx
import json
from datetime import datetime, timezone

from core.models.gateway import (
    ModelInfo, InferenceRequest, InferenceResponse, EmbeddingRequest, EmbeddingResponse,
    ModelGateway, OllamaInstancePool
)


class TestModelInfo:
    """Test suite for ModelInfo dataclass."""
    
    def test_model_info_creation(self):
        """Test ModelInfo creation with all fields."""
        model = ModelInfo(
            name="qwen3:8b",
            size="8.5GB",
            format="gguf",
            family="qwen",
            parameters="8B",
            quantization="Q4_0",
            loaded=True
        )
        
        assert model.name == "qwen3:8b"
        assert model.size == "8.5GB"
        assert model.format == "gguf"
        assert model.family == "qwen"
        assert model.parameters == "8B"
        assert model.quantization == "Q4_0"
        assert model.loaded is True
    
    def test_model_info_defaults(self):
        """Test ModelInfo creation with minimal fields."""
        model = ModelInfo(
            name="test-model",
            size="1GB",
            format="gguf",
            family="test",
            parameters="1B",
            quantization="Q8_0"
        )
        
        assert model.loaded is False  # Default value


class TestInferenceRequest:
    """Test suite for InferenceRequest validation."""
    
    def test_valid_inference_request(self):
        """Test valid inference request creation."""
        request = InferenceRequest(
            model="qwen3:8b",
            prompt="What is AI?",
            stream=True,
            parameters={"temperature": 0.7, "max_tokens": 1024}
        )
        
        assert request.model == "qwen3:8b"
        assert request.prompt == "What is AI?"
        assert request.stream is True
        assert request.parameters == {"temperature": 0.7, "max_tokens": 1024}
    
    def test_minimal_inference_request(self):
        """Test inference request with minimal fields."""
        request = InferenceRequest(
            model="qwen3:8b",
            prompt="Test prompt"
        )
        
        assert request.model == "qwen3:8b"
        assert request.prompt == "Test prompt"
        assert request.stream is False  # Default
        assert request.parameters == {}  # Default
    
    def test_inference_request_validation_empty_model(self):
        """Test inference request validation with empty model."""
        with pytest.raises(ValueError, match="Model name must be a non-empty string"):
            InferenceRequest(model="", prompt="Test prompt")
    
    def test_inference_request_validation_empty_prompt(self):
        """Test inference request validation with empty prompt."""
        with pytest.raises(ValueError, match="Prompt must be a non-empty string"):
            InferenceRequest(model="qwen3:8b", prompt="")
    
    def test_inference_request_validation_whitespace_trimming(self):
        """Test that whitespace is trimmed from model and prompt."""
        request = InferenceRequest(
            model="  qwen3:8b  ",
            prompt="  Test prompt  "
        )
        
        assert request.model == "qwen3:8b"
        assert request.prompt == "  Test prompt  "  # Prompt validation doesn't trim
    
    def test_inference_request_validation_invalid_types(self):
        """Test validation with invalid types."""
        with pytest.raises(ValueError):
            InferenceRequest(model=123, prompt="Test prompt")
        
        with pytest.raises(ValueError):
            InferenceRequest(model="qwen3:8b", prompt=123)


class TestInferenceResponse:
    """Test suite for InferenceResponse model."""
    
    def test_inference_response_creation(self):
        """Test InferenceResponse creation."""
        response = InferenceResponse(
            response="Generated text response",
            done=True,
            context=[1, 2, 3],
            total_duration=1000000,
            eval_count=50
        )
        
        assert response.response == "Generated text response"
        assert response.done is True
        assert response.context == [1, 2, 3]
        assert response.total_duration == 1000000
        assert response.eval_count == 50
    
    def test_inference_response_minimal(self):
        """Test InferenceResponse with minimal fields."""
        response = InferenceResponse(response="Test response")
        
        assert response.response == "Test response"
        assert response.done is True  # Default
        assert response.context is None  # Default


class TestEmbeddingRequest:
    """Test suite for EmbeddingRequest validation."""
    
    def test_valid_embedding_request(self):
        """Test valid embedding request creation."""
        request = EmbeddingRequest(
            model="custom-embedding-model",
            prompt="Text to embed"
        )
        
        assert request.model == "custom-embedding-model"
        assert request.prompt == "Text to embed"
    
    def test_embedding_request_default_model(self):
        """Test embedding request with default model."""
        request = EmbeddingRequest(prompt="Text to embed")
        
        assert request.model == "dengcao/Qwen3-Embedding-0.6B:Q8_0"  # Default
        assert request.prompt == "Text to embed"
    
    def test_embedding_request_validation_empty_prompt(self):
        """Test embedding request validation with empty prompt."""
        with pytest.raises(ValueError, match="Prompt must be a non-empty string"):
            EmbeddingRequest(prompt="")
    
    def test_embedding_request_whitespace_trimming(self):
        """Test that whitespace is trimmed from prompt."""
        request = EmbeddingRequest(prompt="  Text to embed  ")
        assert request.prompt == "Text to embed"


class TestEmbeddingResponse:
    """Test suite for EmbeddingResponse model."""
    
    def test_embedding_response_creation(self):
        """Test EmbeddingResponse creation."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        response = EmbeddingResponse(
            embedding=embedding,
            dimensions=len(embedding),
            total_duration=500000,
            model_info={"name": "test-model"}
        )
        
        assert response.embedding == embedding
        assert response.dimensions == 5
        assert response.total_duration == 500000
        assert response.model_info == {"name": "test-model"}


class TestOllamaInstancePool:
    """Test suite for OllamaInstancePool load balancing."""
    
    @pytest.fixture
    def pool(self):
        """Create fresh instance pool for testing."""
        return OllamaInstancePool()
    
    def test_pool_initialization(self, pool):
        """Test pool initialization with default instance."""
        assert len(pool.instances) == 1
        assert pool.instances[0] == "http://localhost:11434"
        assert pool.current_index == 0
    
    @pytest.mark.asyncio
    async def test_add_instance(self, pool):
        """Test adding new instance to pool."""
        new_url = "http://localhost:11435"
        
        await pool.add_instance(new_url)
        
        assert len(pool.instances) == 2
        assert new_url in pool.instances
    
    @pytest.mark.asyncio
    async def test_add_duplicate_instance(self, pool):
        """Test adding duplicate instance doesn't create duplicates."""
        existing_url = "http://localhost:11434"
        
        await pool.add_instance(existing_url)
        
        assert len(pool.instances) == 1  # No duplicate added
    
    @pytest.mark.asyncio
    async def test_get_next_instance_round_robin(self, pool):
        """Test round-robin instance selection."""
        url1 = "http://localhost:11435"
        url2 = "http://localhost:11436"
        
        await pool.add_instance(url1)
        await pool.add_instance(url2)
        
        # Should cycle through instances
        first = await pool.get_next_instance()
        second = await pool.get_next_instance()
        third = await pool.get_next_instance()
        fourth = await pool.get_next_instance()
        
        assert first in pool.instances
        assert second in pool.instances
        assert third in pool.instances
        assert fourth == first  # Should wrap around
    
    @pytest.mark.asyncio
    async def test_get_random_instance(self, pool):
        """Test random instance selection."""
        url1 = "http://localhost:11435"
        url2 = "http://localhost:11436"
        
        await pool.add_instance(url1)
        await pool.add_instance(url2)
        
        # Get multiple random instances
        instances = [await pool.get_random_instance() for _ in range(10)]
        
        # All should be valid instances
        for instance in instances:
            assert instance in pool.instances
    
    @pytest.mark.asyncio
    async def test_empty_pool_fallback(self):
        """Test fallback when pool is empty."""
        pool = OllamaInstancePool()
        pool.instances = []  # Empty the pool
        
        next_instance = await pool.get_next_instance()
        random_instance = await pool.get_random_instance()
        
        assert next_instance == "http://localhost:11434"
        assert random_instance == "http://localhost:11434"
    
    def test_get_instance_count(self, pool):
        """Test instance count retrieval."""
        assert pool.get_instance_count() == 1
        
        # Add instances and check count
        asyncio.run(pool.add_instance("http://localhost:11435"))
        assert pool.get_instance_count() == 2


class TestModelGateway:
    """Test suite for ModelGateway class."""
    
    @pytest.fixture
    def gateway(self):
        """Create ModelGateway instance for testing."""
        return ModelGateway("http://localhost:11434")
    
    def test_gateway_initialization(self, gateway):
        """Test gateway initialization."""
        assert gateway.ollama_url == "http://localhost:11434"
        assert isinstance(gateway.loaded_models, dict)
        assert len(gateway.loaded_models) == 0
        assert isinstance(gateway.client, httpx.AsyncClient)
    
    @pytest.mark.asyncio
    async def test_discover_models_success(self, gateway):
        """Test successful model discovery."""
        mock_response_data = {
            "models": [
                {
                    "name": "qwen3:8b",
                    "size": "8.5GB",
                    "format": "gguf",
                    "details": {
                        "family": "qwen",
                        "parameter_size": "8B",
                        "quantization_level": "Q4_0"
                    }
                },
                {
                    "name": "llama3:7b",
                    "size": "7.2GB",
                    "format": "gguf",
                    "details": {
                        "family": "llama",
                        "parameter_size": "7B",
                        "quantization_level": "Q4_0"
                    }
                }
            ]
        }
        
        with patch.object(gateway.client, 'get') as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = mock_response_data
            mock_get.return_value = mock_response
            
            models = await gateway.discover_models()
            
            assert len(models) == 2
            assert models[0].name == "qwen3:8b"
            assert models[1].name == "llama3:7b"
            assert len(gateway.loaded_models) == 2
            assert "qwen3:8b" in gateway.loaded_models
            assert "llama3:7b" in gateway.loaded_models
    
    @pytest.mark.asyncio
    async def test_discover_models_http_error(self, gateway):
        """Test model discovery with HTTP error."""
        with patch.object(gateway.client, 'get') as mock_get:
            mock_get.side_effect = httpx.HTTPStatusError(
                "Error", request=Mock(), response=Mock(status_code=500)
            )
            
            with pytest.raises(HTTPException) as exc_info:
                await gateway.discover_models()
            
            assert exc_info.value.status_code == 503
            assert "Cannot connect to Ollama" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_load_model_success(self, gateway):
        """Test successful model loading."""
        model_name = "qwen3:8b"
        
        with patch.object(gateway.client, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            # Add model to loaded_models first
            gateway.loaded_models[model_name] = ModelInfo(
                name=model_name, size="8GB", format="gguf",
                family="qwen", parameters="8B", quantization="Q4_0"
            )
            
            result = await gateway.load_model(model_name)
            
            assert result is True
            assert gateway.loaded_models[model_name].loaded is True
    
    @pytest.mark.asyncio
    async def test_load_model_failure(self, gateway):
        """Test model loading failure."""
        model_name = "qwen3:8b"
        
        with patch.object(gateway.client, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_post.return_value = mock_response
            
            result = await gateway.load_model(model_name)
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_unload_model(self, gateway):
        """Test model unloading."""
        model_name = "qwen3:8b"
        
        # Add loaded model
        gateway.loaded_models[model_name] = ModelInfo(
            name=model_name, size="8GB", format="gguf",
            family="qwen", parameters="8B", quantization="Q4_0",
            loaded=True
        )
        
        result = await gateway.unload_model(model_name)
        
        assert result is True
        assert gateway.loaded_models[model_name].loaded is False
    
    @pytest.mark.asyncio
    async def test_get_model_info_exists(self, gateway):
        """Test getting existing model info."""
        model_name = "qwen3:8b"
        model_info = ModelInfo(
            name=model_name, size="8GB", format="gguf",
            family="qwen", parameters="8B", quantization="Q4_0"
        )
        gateway.loaded_models[model_name] = model_info
        
        result = await gateway.get_model_info(model_name)
        
        assert result == model_info
    
    @pytest.mark.asyncio
    async def test_get_model_info_not_exists(self, gateway):
        """Test getting non-existent model info."""
        result = await gateway.get_model_info("nonexistent")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, gateway):
        """Test health check when service is healthy."""
        with patch.object(gateway.client, 'get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            # Add some loaded models
            gateway.loaded_models["test"] = ModelInfo(
                name="test", size="1GB", format="gguf",
                family="test", parameters="1B", quantization="Q8_0",
                loaded=True
            )
            
            result = await gateway.health_check()
            
            assert result["status"] == "healthy"
            assert result["ollama_url"] == gateway.ollama_url
            assert result["models_loaded"] == 1
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, gateway):
        """Test health check when service is unhealthy."""
        with patch.object(gateway.client, 'get') as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection failed")
            
            result = await gateway.health_check()
            
            assert result["status"] == "unhealthy"
            assert "error" in result
            assert result["ollama_url"] == gateway.ollama_url
    
    @pytest.mark.asyncio
    async def test_generate_embedding_success(self, gateway):
        """Test successful embedding generation."""
        model_name = "embedding-model"
        prompt = "Test text"
        expected_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Mock the instance pool
        with patch('core.models.gateway.ollama_instance_pool') as mock_pool:
            mock_pool.get_random_instance.return_value = "http://localhost:11434"
            
            with patch.object(gateway.client, 'stream') as mock_stream:
                mock_response = Mock()
                mock_response.aread.return_value = json.dumps({"embedding": expected_embedding}).encode()
                mock_response.raise_for_status = Mock()
                
                mock_context = AsyncMock()
                mock_context.__aenter__.return_value = mock_response
                mock_context.__aexit__.return_value = None
                mock_stream.return_value = mock_context
                
                result = await gateway.generate_embedding(model_name, prompt)
                
                assert result == expected_embedding
                mock_stream.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_embedding_fallback_to_embeddings_endpoint(self, gateway):
        """Test embedding generation fallback to /api/embeddings."""
        model_name = "embedding-model"
        prompt = "Test text"
        expected_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Mock the instance pool
        with patch('core.models.gateway.ollama_instance_pool') as mock_pool:
            mock_pool.get_random_instance.return_value = "http://localhost:11434"
            
            call_count = 0
            def mock_stream_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                
                if "/api/embed" in args[1]:  # First call to /api/embed fails
                    raise httpx.HTTPStatusError("Not found", request=Mock(), response=Mock())
                else:  # Second call to /api/embeddings succeeds
                    mock_response = Mock()
                    mock_response.aread.return_value = json.dumps({"embedding": expected_embedding}).encode()
                    mock_response.raise_for_status = Mock()
                    
                    mock_context = AsyncMock()
                    mock_context.__aenter__.return_value = mock_response
                    mock_context.__aexit__.return_value = None
                    return mock_context
            
            with patch.object(gateway.client, 'stream', side_effect=mock_stream_side_effect):
                result = await gateway.generate_embedding(model_name, prompt)
                
                assert result == expected_embedding
    
    @pytest.mark.asyncio
    async def test_generate_embedding_both_endpoints_fail(self, gateway):
        """Test embedding generation when both endpoints fail."""
        model_name = "embedding-model"
        prompt = "Test text"
        
        # Mock the instance pool
        with patch('core.models.gateway.ollama_instance_pool') as mock_pool:
            mock_pool.get_random_instance.return_value = "http://localhost:11434"
            
            with patch.object(gateway.client, 'stream') as mock_stream:
                mock_stream.side_effect = httpx.HTTPStatusError(
                    "Error", request=Mock(), response=Mock()
                )
                
                with pytest.raises(HTTPException) as exc_info:
                    await gateway.generate_embedding(model_name, prompt)
                
                assert exc_info.value.status_code == 500
                assert "Embedding generation error" in str(exc_info.value.detail)


if __name__ == '__main__':
    pytest.main([__file__])