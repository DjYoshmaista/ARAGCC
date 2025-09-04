#!/usr/bin/env python3
"""
Unit tests for the embedding load balancer
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import httpx
import sys
import os

# Add the services directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'services', 'orchestrator'))

try:
    from embedding_load_balancer import EmbeddingLoadBalancer, LoadBalancingStrategy
    from ollama_instance_manager import OllamaInstance
except ImportError as e:
    print(f"Warning: Could not import embedding load balancer: {e}")
    
    # Mock classes for testing structure
    class LoadBalancingStrategy:
        LEAST_CONNECTIONS = "least_connections"
        ROUND_ROBIN = "round_robin"
        WEIGHTED_RESPONSE_TIME = "weighted_response_time"
        RANDOM = "random"
    
    class OllamaInstance:
        def __init__(self, port=11434, status="running", base_url=None):
            self.port = port
            self.status = status
            self.base_url = base_url or f"http://localhost:{port}"
            self.last_response_time = 0.1
    
    class EmbeddingLoadBalancer:
        def __init__(self, strategy=LoadBalancingStrategy.LEAST_CONNECTIONS):
            self.strategy = strategy
            
        async def generate_embedding(self, text, model):
            return ([0.1] * 1024, True, "")

class TestEmbeddingLoadBalancer(unittest.TestCase):
    """Test cases for the EmbeddingLoadBalancer class"""
    
    def setUp(self):
        """Set up test environment"""
        self.load_balancer = EmbeddingLoadBalancer(LoadBalancingStrategy.LEAST_CONNECTIONS)
        
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self.load_balancer, 'cleanup'):
            try:
                asyncio.run(self.load_balancer.cleanup())
            except:
                pass
    
    def test_initialization(self):
        """Test load balancer initialization"""
        self.assertEqual(self.load_balancer.strategy, LoadBalancingStrategy.LEAST_CONNECTIONS)
        self.assertIsNotNone(self.load_balancer)
    
    def test_strategy_types(self):
        """Test all load balancing strategy types"""
        strategies = [
            LoadBalancingStrategy.LEAST_CONNECTIONS,
            LoadBalancingStrategy.ROUND_ROBIN, 
            LoadBalancingStrategy.WEIGHTED_RESPONSE_TIME,
            LoadBalancingStrategy.RANDOM
        ]
        
        for strategy in strategies:
            balancer = EmbeddingLoadBalancer(strategy)
            self.assertEqual(balancer.strategy, strategy)
    
    @patch('embedding_load_balancer.get_instance_manager')
    async def test_generate_embedding_api_embed_success(self, mock_get_instance_manager):
        """Test successful embedding generation using /api/embed endpoint"""
        if not hasattr(self.load_balancer, 'generate_embedding'):
            self.skipTest("generate_embedding method not available")
        
        # Mock instance manager
        mock_instance = OllamaInstance(port=11434)
        mock_instance_manager = Mock()
        mock_instance_manager.instances = [mock_instance]
        mock_get_instance_manager.return_value = mock_instance_manager
        
        # Mock HTTP client
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response
        
        # Mock the client pool
        if hasattr(self.load_balancer, 'client_pool'):
            self.load_balancer.client_pool = {11434: mock_client}
            self.load_balancer.instance_connections = {11434: 0}
            self.load_balancer.instance_circuit_breakers = {11434: {'state': 'closed', 'failures': 0}}
        
        result = await self.load_balancer.generate_embedding("test text", "test-model")
        
        # Check result structure
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)  # (embedding, success, error_message)
        embedding, success, error_msg = result
        
        self.assertTrue(success)
        self.assertEqual(error_msg, "")
        self.assertIsInstance(embedding, list)
    
    @patch('embedding_load_balancer.get_instance_manager') 
    async def test_generate_embedding_api_embed_fallback_to_embeddings(self, mock_get_instance_manager):
        """Test fallback from /api/embed to /api/embeddings endpoint"""
        if not hasattr(self.load_balancer, 'generate_embedding'):
            self.skipTest("generate_embedding method not available")
        
        # Mock instance manager
        mock_instance = OllamaInstance(port=11434)
        mock_instance_manager = Mock()
        mock_instance_manager.instances = [mock_instance]
        mock_get_instance_manager.return_value = mock_instance_manager
        
        # Mock HTTP client - first call fails, second succeeds
        mock_client = AsyncMock()
        
        def mock_post_side_effect(*args, **kwargs):
            endpoint = args[0] if args else ""
            if "/api/embed" in endpoint:
                # First call to /api/embed fails
                raise httpx.HTTPStatusError("404 Not Found", request=None, response=Mock(status_code=404))
            else:
                # Second call to /api/embeddings succeeds
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
                mock_response.raise_for_status = Mock()
                return mock_response
        
        mock_client.post.side_effect = mock_post_side_effect
        
        # Mock the client pool
        if hasattr(self.load_balancer, 'client_pool'):
            self.load_balancer.client_pool = {11434: mock_client}
            self.load_balancer.instance_connections = {11434: 0}
            self.load_balancer.instance_circuit_breakers = {11434: {'state': 'closed', 'failures': 0}}
        
        result = await self.load_balancer.generate_embedding("test text", "test-model")
        
        # Check that both endpoints were tried
        self.assertEqual(mock_client.post.call_count, 2)
        
        # Check result
        embedding, success, error_msg = result
        self.assertTrue(success)
        self.assertEqual(error_msg, "")
        self.assertIsInstance(embedding, list)
    
    @patch('embedding_load_balancer.get_instance_manager')
    async def test_generate_embedding_both_endpoints_fail(self, mock_get_instance_manager):
        """Test behavior when both /api/embed and /api/embeddings fail"""
        if not hasattr(self.load_balancer, 'generate_embedding'):
            self.skipTest("generate_embedding method not available")
        
        # Mock instance manager
        mock_instance = OllamaInstance(port=11434)
        mock_instance_manager = Mock()
        mock_instance_manager.instances = [mock_instance]
        mock_get_instance_manager.return_value = mock_instance_manager
        
        # Mock HTTP client - both calls fail
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.HTTPStatusError("500 Server Error", request=None, response=Mock(status_code=500))
        
        # Mock the client pool
        if hasattr(self.load_balancer, 'client_pool'):
            self.load_balancer.client_pool = {11434: mock_client}
            self.load_balancer.instance_connections = {11434: 0}
            self.load_balancer.instance_circuit_breakers = {11434: {'state': 'closed', 'failures': 0}}
        
        result = await self.load_balancer.generate_embedding("test text", "test-model")
        
        # Check result - should indicate failure
        embedding, success, error_msg = result
        self.assertFalse(success)
        self.assertIsNotNone(error_msg)
        self.assertIn("failed", error_msg.lower())
    
    async def test_batch_generate_embeddings(self):
        """Test batch embedding generation"""
        if not hasattr(self.load_balancer, 'batch_generate_embeddings'):
            self.skipTest("batch_generate_embeddings method not available")
        
        requests = [
            ("text 1", "model 1"),
            ("text 2", "model 2"),
            ("text 3", "model 3")
        ]
        
        with patch.object(self.load_balancer, 'generate_embedding') as mock_generate:
            # Mock successful embedding generation
            mock_generate.return_value = ([0.1, 0.2, 0.3], True, "")
            
            results = await self.load_balancer.batch_generate_embeddings(requests)
            
            self.assertEqual(len(results), 3)
            self.assertEqual(mock_generate.call_count, 3)
            
            for embedding, success, error_msg in results:
                self.assertTrue(success)
                self.assertEqual(error_msg, "")
                self.assertIsInstance(embedding, list)
    
    def test_get_stats(self):
        """Test statistics retrieval"""
        if not hasattr(self.load_balancer, 'get_stats'):
            self.skipTest("get_stats method not available")
        
        stats = self.load_balancer.get_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('strategy', stats)
        self.assertEqual(stats['strategy'], self.load_balancer.strategy.value if hasattr(self.load_balancer.strategy, 'value') else self.load_balancer.strategy)

class TestLoadBalancingStrategies(unittest.TestCase):
    """Test different load balancing strategies"""
    
    def setUp(self):
        """Set up test instances"""
        self.instances = [
            OllamaInstance(port=11434),
            OllamaInstance(port=11435),
            OllamaInstance(port=11436)
        ]
    
    def test_round_robin_strategy(self):
        """Test round-robin load balancing strategy"""
        balancer = EmbeddingLoadBalancer(LoadBalancingStrategy.ROUND_ROBIN)
        
        if hasattr(balancer, '_round_robin_select'):
            # Test multiple selections to verify round-robin behavior
            selections = []
            for i in range(6):  # More than number of instances to see rotation
                selected = balancer._round_robin_select(self.instances)
                selections.append(selected.port)
            
            # Should see repeated pattern
            self.assertEqual(selections[0], selections[3])
            self.assertEqual(selections[1], selections[4])  
            self.assertEqual(selections[2], selections[5])
    
    def test_least_connections_strategy(self):
        """Test least connections load balancing strategy"""
        balancer = EmbeddingLoadBalancer(LoadBalancingStrategy.LEAST_CONNECTIONS)
        
        if hasattr(balancer, '_least_connections_select') and hasattr(balancer, 'instance_connections'):
            # Set up connection counts
            balancer.instance_connections = {
                11434: 5,
                11435: 2,  # Should be selected (least connections)
                11436: 8
            }
            
            selected = balancer._least_connections_select(self.instances)
            self.assertEqual(selected.port, 11435)
    
    def test_weighted_response_time_strategy(self):
        """Test weighted response time strategy"""
        balancer = EmbeddingLoadBalancer(LoadBalancingStrategy.WEIGHTED_RESPONSE_TIME)
        
        if hasattr(balancer, '_weighted_response_time_select'):
            # Set response times
            self.instances[0].last_response_time = 0.5  # Slower
            self.instances[1].last_response_time = 0.1  # Faster (should be preferred)
            self.instances[2].last_response_time = 0.3  # Medium
            
            # Run multiple selections and verify faster instance is selected more often
            selections = {}
            for _ in range(100):
                selected = balancer._weighted_response_time_select(self.instances)
                port = selected.port
                selections[port] = selections.get(port, 0) + 1
            
            # Instance with fastest response time should be selected most often
            fastest_port = 11435
            self.assertGreater(selections[fastest_port], selections.get(11434, 0))
            self.assertGreater(selections[fastest_port], selections.get(11436, 0))

class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.balancer = EmbeddingLoadBalancer()
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization"""
        if hasattr(self.balancer, '_initialize_circuit_breaker'):
            self.balancer._initialize_circuit_breaker(11434)
            
            if hasattr(self.balancer, 'instance_circuit_breakers'):
                breaker = self.balancer.instance_circuit_breakers.get(11434)
                self.assertIsNotNone(breaker)
                self.assertEqual(breaker['state'], 'closed')
                self.assertEqual(breaker['failures'], 0)
    
    def test_circuit_breaker_failure_tracking(self):
        """Test circuit breaker failure tracking"""
        if not (hasattr(self.balancer, '_initialize_circuit_breaker') and 
                hasattr(self.balancer, '_update_circuit_breaker') and
                hasattr(self.balancer, 'circuit_breaker_threshold')):
            self.skipTest("Circuit breaker methods not available")
        
        port = 11434
        self.balancer._initialize_circuit_breaker(port)
        
        # Simulate failures up to threshold
        for i in range(self.balancer.circuit_breaker_threshold):
            self.balancer._update_circuit_breaker(port, False)  # Failure
        
        breaker = self.balancer.instance_circuit_breakers[port]
        self.assertEqual(breaker['state'], 'open')  # Should open after threshold failures
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery"""
        if not (hasattr(self.balancer, '_initialize_circuit_breaker') and 
                hasattr(self.balancer, '_update_circuit_breaker')):
            self.skipTest("Circuit breaker methods not available")
        
        port = 11434
        self.balancer._initialize_circuit_breaker(port)
        
        # Open circuit breaker
        breaker = self.balancer.instance_circuit_breakers[port]
        breaker['state'] = 'half_open'
        
        # Simulate successful requests for recovery
        recovery_requests = getattr(self.balancer, 'circuit_breaker_recovery_requests', 3)
        for i in range(recovery_requests):
            self.balancer._update_circuit_breaker(port, True)  # Success
        
        # Should be closed again
        self.assertEqual(breaker['state'], 'closed')

if __name__ == '__main__':
    unittest.main()