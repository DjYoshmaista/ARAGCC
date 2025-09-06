"""
Comprehensive unit tests for the CLI application.

Tests cover configuration management, command parsing, API client functionality,
error handling, and service integrations.
"""

import pytest
import asyncio
import tempfile
import os
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import httpx

from client.app import (
    CLIConfig, CLIManager, CommandParser, APIClient, 
    CONFIG_DIR, CONFIG_FILE, LOG_DIR
)


class TestCLIConfig:
    """Test suite for CLI configuration."""
    
    def test_cli_config_defaults(self):
        """Test CLI configuration with default values."""
        config = CLIConfig()
        
        assert config.orchestrator_url == "http://localhost:8001"
        assert config.model_gateway_url == "http://localhost:8070"
        assert config.vector_engine_url == "http://localhost:8080"
        assert config.postgresql_host == "localhost"
        assert config.postgresql_port == 5432
        assert config.postgresql_db == "agentic_system"
        assert config.rag_top_k == 5
        assert config.rag_weight == 0.0
        assert config.max_agents == 5
        assert config.default_temperature == 0.7
        assert config.timeout_seconds == 300
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
    
    def test_cli_config_post_init(self):
        """Test CLI configuration post-initialization."""
        config = CLIConfig()
        
        assert config.models is not None
        assert "orchestration" in config.models
        assert "inference" in config.models
        assert "embedding" in config.models
        assert "rag" in config.models
        assert config.enabled_plugins == []
        assert config.plugin_dir is not None
    
    def test_cli_config_validation_valid(self):
        """Test CLI configuration validation with valid values."""
        config = CLIConfig(
            postgresql_port=5432,
            rag_top_k=10,
            rag_weight=5.0,
            max_agents=3,
            default_temperature=0.8,
            timeout_seconds=600,
            max_retries=5
        )
        # Should not raise any exceptions
        assert config.postgresql_port == 5432
    
    def test_cli_config_validation_invalid_port(self):
        """Test CLI configuration validation with invalid port."""
        with pytest.raises(ValueError, match="Invalid PostgreSQL port"):
            CLIConfig(postgresql_port=0)
        
        with pytest.raises(ValueError, match="Invalid PostgreSQL port"):
            CLIConfig(postgresql_port=70000)
    
    def test_cli_config_validation_invalid_rag_top_k(self):
        """Test CLI configuration validation with invalid rag_top_k."""
        with pytest.raises(ValueError, match="Invalid rag_top_k"):
            CLIConfig(rag_top_k=0)
        
        with pytest.raises(ValueError, match="Invalid rag_top_k"):
            CLIConfig(rag_top_k=101)
    
    def test_cli_config_validation_invalid_rag_weight(self):
        """Test CLI configuration validation with invalid rag_weight."""
        with pytest.raises(ValueError, match="Invalid rag_weight"):
            CLIConfig(rag_weight=-11.0)
        
        with pytest.raises(ValueError, match="Invalid rag_weight"):
            CLIConfig(rag_weight=11.0)
    
    def test_cli_config_validation_invalid_max_agents(self):
        """Test CLI configuration validation with invalid max_agents."""
        with pytest.raises(ValueError, match="Invalid max_agents"):
            CLIConfig(max_agents=0)
        
        with pytest.raises(ValueError, match="Invalid max_agents"):
            CLIConfig(max_agents=51)
    
    def test_cli_config_validation_invalid_temperature(self):
        """Test CLI configuration validation with invalid temperature."""
        with pytest.raises(ValueError, match="Invalid default_temperature"):
            CLIConfig(default_temperature=-0.1)
        
        with pytest.raises(ValueError, match="Invalid default_temperature"):
            CLIConfig(default_temperature=2.1)
    
    def test_cli_config_validation_invalid_timeout(self):
        """Test CLI configuration validation with invalid timeout."""
        with pytest.raises(ValueError, match="Invalid timeout_seconds"):
            CLIConfig(timeout_seconds=5)
        
        with pytest.raises(ValueError, match="Invalid timeout_seconds"):
            CLIConfig(timeout_seconds=3601)
    
    def test_cli_config_validation_invalid_retries(self):
        """Test CLI configuration validation with invalid retries."""
        with pytest.raises(ValueError, match="Invalid max_retries"):
            CLIConfig(max_retries=0)
        
        with pytest.raises(ValueError, match="Invalid max_retries"):
            CLIConfig(max_retries=11)
    
    def test_cli_config_to_dict(self):
        """Test CLI configuration serialization to dictionary."""
        config = CLIConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "orchestrator_url" in config_dict
        assert "models" in config_dict
        assert "plugin_dir" in config_dict
        # Plugin dir should be converted to string
        assert isinstance(config_dict["plugin_dir"], str)


class TestCLIManager:
    """Test suite for CLI manager."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Mock the config directories
            with patch('client.app.CONFIG_DIR', temp_path / "config"):
                with patch('client.app.CONFIG_FILE', temp_path / "config" / "config.yaml"):
                    with patch('client.app.LOG_DIR', temp_path / "logs"):
                        yield temp_path
    
    def test_cli_manager_initialization(self, temp_config_dir):
        """Test CLI manager initialization."""
        manager = CLIManager()
        
        assert isinstance(manager.config, CLIConfig)
        assert manager.http_client is None
        assert manager.logger is not None
    
    def test_cli_manager_ensure_directories(self, temp_config_dir):
        """Test CLI manager directory creation."""
        manager = CLIManager()
        
        # Directories should be created
        config_dir = temp_config_dir / "config"
        log_dir = temp_config_dir / "logs"
        assert config_dir.exists()
        assert log_dir.exists()
    
    @patch('builtins.open')
    @patch('yaml.safe_load')
    def test_cli_manager_load_config_success(self, mock_yaml_load, mock_open, temp_config_dir):
        """Test successful configuration loading."""
        mock_config_data = {
            'orchestrator_url': 'http://custom:8001',
            'verbose': True,
            'max_agents': 10
        }
        mock_yaml_load.return_value = mock_config_data
        mock_open.return_value.__enter__.return_value = Mock()
        
        with patch('client.app.CONFIG_FILE') as mock_config_file:
            mock_config_file.exists.return_value = True
            manager = CLIManager()
            
            assert manager.config.orchestrator_url == 'http://custom:8001'
            assert manager.config.verbose is True
            assert manager.config.max_agents == 10
    
    @patch('builtins.open')
    def test_cli_manager_load_config_file_not_found(self, mock_open, temp_config_dir):
        """Test configuration loading when file doesn't exist."""
        with patch('client.app.CONFIG_FILE') as mock_config_file:
            mock_config_file.exists.return_value = False
            manager = CLIManager()
            
            # Should use default config
            assert manager.config.orchestrator_url == "http://localhost:8001"
    
    @patch('builtins.open')
    @patch('yaml.safe_load')
    def test_cli_manager_load_config_error(self, mock_yaml_load, mock_open, temp_config_dir):
        """Test configuration loading with error."""
        mock_yaml_load.side_effect = yaml.YAMLError("Invalid YAML")
        mock_open.return_value.__enter__.return_value = Mock()
        
        with patch('client.app.CONFIG_FILE') as mock_config_file:
            mock_config_file.exists.return_value = True
            with patch('builtins.print') as mock_print:
                manager = CLIManager()
                
                # Should use default config and print warning
                assert manager.config.orchestrator_url == "http://localhost:8001"
                mock_print.assert_called()
    
    @patch('builtins.open')
    @patch('yaml.dump')
    def test_cli_manager_save_config_success(self, mock_yaml_dump, mock_open, temp_config_dir):
        """Test successful configuration saving."""
        manager = CLIManager()
        
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        manager._save_config()
        
        mock_open.assert_called()
        mock_yaml_dump.assert_called()
    
    @patch('builtins.open')
    def test_cli_manager_save_config_error(self, mock_open, temp_config_dir):
        """Test configuration saving with error."""
        mock_open.side_effect = IOError("Permission denied")
        manager = CLIManager()
        
        with patch('builtins.print') as mock_print:
            manager._save_config()
            mock_print.assert_called()
    
    @pytest.mark.asyncio
    async def test_cli_manager_get_http_client(self, temp_config_dir):
        """Test HTTP client creation."""
        manager = CLIManager()
        
        client = await manager._get_http_client()
        
        assert isinstance(client, httpx.AsyncClient)
        assert manager.http_client is not None
        
        # Getting client again should return same instance
        client2 = await manager._get_http_client()
        assert client2 is client
        
        await manager.close()
    
    @pytest.mark.asyncio
    async def test_cli_manager_close(self, temp_config_dir):
        """Test CLI manager cleanup."""
        manager = CLIManager()
        
        # Create HTTP client
        await manager._get_http_client()
        assert manager.http_client is not None
        
        # Close should cleanup client
        await manager.close()


class TestCommandParser:
    """Test suite for command parser."""
    
    def test_command_parser_initialization(self):
        """Test command parser initialization."""
        with patch('client.app.CLIManager') as mock_manager:
            mock_cli_manager = Mock()
            parser = CommandParser(mock_cli_manager)
            
            assert parser.cli_manager is mock_cli_manager
            assert parser.parser is not None
    
    def test_command_parser_no_args(self):
        """Test command parser with no arguments."""
        with patch('client.app.CLIManager') as mock_manager:
            mock_cli_manager = Mock()
            parser = CommandParser(mock_cli_manager)
            
            args = parser.parse_args([])
            assert args.command is None
    
    def test_command_parser_query_command(self):
        """Test command parser with query command."""
        with patch('client.app.CLIManager') as mock_manager:
            mock_cli_manager = Mock()
            parser = CommandParser(mock_cli_manager)
            
            args = parser.parse_args(['query', 'test prompt'])
            assert args.command == 'query'
            assert args.prompt == 'test prompt'
            assert args.model is None
            assert args.temperature is None
            assert args.rag_weight is None
    
    def test_command_parser_query_command_with_options(self):
        """Test command parser with query command and options."""
        with patch('client.app.CLIManager') as mock_manager:
            mock_cli_manager = Mock()
            parser = CommandParser(mock_cli_manager)
            
            args = parser.parse_args([
                'query', 'test prompt', 
                '--model', 'qwen3:8b',
                '--temperature', '0.8',
                '--rag-weight', '5.0',
                '--top-k', '10'
            ])
            assert args.command == 'query'
            assert args.prompt == 'test prompt'
            assert args.model == 'qwen3:8b'
            assert args.temperature == 0.8
            assert args.rag_weight == 5.0
            assert args.top_k == 10
    
    def test_command_parser_ingest_command(self):
        """Test command parser with ingest command."""
        with patch('client.app.CLIManager') as mock_manager:
            mock_cli_manager = Mock()
            parser = CommandParser(mock_cli_manager)
            
            args = parser.parse_args(['ingest', 'path1', 'path2', '--recursive'])
            assert args.command == 'ingest'
            assert args.paths == ['path1', 'path2']
            assert args.recursive is True
    
    def test_command_parser_config_command(self):
        """Test command parser with config command."""
        with patch('client.app.CLIManager') as mock_manager:
            mock_cli_manager = Mock()
            parser = CommandParser(mock_cli_manager)
            
            args = parser.parse_args(['config', '--show', '--set', 'key=value'])
            assert args.command == 'config'
            assert args.show is True
            assert args.set == ['key=value']
    
    def test_command_parser_status_command(self):
        """Test command parser with status command."""
        with patch('client.app.CLIManager') as mock_manager:
            mock_cli_manager = Mock()
            parser = CommandParser(mock_cli_manager)
            
            args = parser.parse_args(['status', '--health', '--services'])
            assert args.command == 'status'
            assert args.health is True
            assert args.services is True
    
    def test_command_parser_verbose_flag(self):
        """Test command parser with verbose flag."""
        with patch('client.app.CLIManager') as mock_manager:
            mock_cli_manager = Mock()
            parser = CommandParser(mock_cli_manager)
            
            args = parser.parse_args(['--verbose', 'query', 'test'])
            assert args.verbose is True
    
    def test_command_parser_no_color_flag(self):
        """Test command parser with no-color flag."""
        with patch('client.app.CLIManager') as mock_manager:
            mock_cli_manager = Mock()
            parser = CommandParser(mock_cli_manager)
            
            args = parser.parse_args(['--no-color', 'status'])
            assert args.no_color is True


class TestAPIClient:
    """Test suite for API client."""
    
    @pytest.fixture
    def mock_cli_manager(self):
        """Create mock CLI manager for testing."""
        manager = Mock()
        manager.config = CLIConfig()
        manager.logger = Mock()
        return manager
    
    @pytest.fixture
    def api_client(self, mock_cli_manager):
        """Create API client for testing."""
        return APIClient(mock_cli_manager)
    
    @pytest.mark.asyncio
    async def test_api_client_health_check_success(self, api_client, mock_cli_manager):
        """Test successful health check."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.elapsed.total_seconds.return_value = 0.123
        mock_client.get.return_value = mock_response
        
        mock_cli_manager._get_http_client.return_value = mock_client
        
        result = await api_client.health_check()
        
        assert "orchestrator" in result
        assert "model_gateway" in result
        assert "vector_engine" in result
        assert result["orchestrator"]["status"] == "healthy"
        assert result["orchestrator"]["status_code"] == 200
        assert result["orchestrator"]["response_time"] == 0.123
    
    @pytest.mark.asyncio
    async def test_api_client_health_check_connection_error(self, api_client, mock_cli_manager):
        """Test health check with connection error."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")
        
        mock_cli_manager._get_http_client.return_value = mock_client
        
        result = await api_client.health_check()
        
        assert result["orchestrator"]["status"] == "error"
        assert "Connection error" in result["orchestrator"]["error"]
    
    @pytest.mark.asyncio
    async def test_api_client_health_check_timeout(self, api_client, mock_cli_manager):
        """Test health check with timeout."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.TimeoutException("Request timeout")
        
        mock_cli_manager._get_http_client.return_value = mock_client
        
        result = await api_client.health_check()
        
        assert result["orchestrator"]["status"] == "error"
        assert "Timeout error" in result["orchestrator"]["error"]
    
    @pytest.mark.asyncio
    async def test_api_client_health_check_http_error(self, api_client, mock_cli_manager):
        """Test health check with HTTP error."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 500
        mock_client.get.side_effect = httpx.HTTPStatusError(
            "Server Error", request=Mock(), response=mock_response
        )
        
        mock_cli_manager._get_http_client.return_value = mock_client
        
        result = await api_client.health_check()
        
        assert result["orchestrator"]["status"] == "unhealthy"
        assert result["orchestrator"]["status_code"] == 500
    
    @pytest.mark.asyncio
    async def test_api_client_submit_task_success(self, api_client, mock_cli_manager):
        """Test successful task submission."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"task_id": "task123", "status": "created"}
        mock_client.post.return_value = mock_response
        
        mock_cli_manager._get_http_client.return_value = mock_client
        
        result = await api_client.submit_task(
            "test_task", 
            "Test task description",
            {"param1": "value1"}
        )
        
        assert result["task_id"] == "task123"
        assert result["status"] == "created"
        mock_client.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_api_client_submit_task_http_error(self, api_client, mock_cli_manager):
        """Test task submission with HTTP error."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "Bad Request", request=Mock(), response=mock_response
        )
        
        mock_cli_manager._get_http_client.return_value = mock_client
        
        with pytest.raises(httpx.HTTPStatusError):
            await api_client.submit_task("test_task", "Test description", {})
    
    @pytest.mark.asyncio
    async def test_api_client_get_task_status_success(self, api_client, mock_cli_manager):
        """Test successful task status retrieval."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "id": "task123",
            "status": "completed",
            "result": "Task completed successfully"
        }
        mock_client.get.return_value = mock_response
        
        mock_cli_manager._get_http_client.return_value = mock_client
        
        result = await api_client.get_task_status("task123")
        
        assert result["id"] == "task123"
        assert result["status"] == "completed"
        assert result["result"] == "Task completed successfully"
    
    @pytest.mark.asyncio
    async def test_api_client_get_task_status_not_found(self, api_client, mock_cli_manager):
        """Test task status retrieval with not found error."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Task not found"
        mock_client.get.side_effect = httpx.HTTPStatusError(
            "Not Found", request=Mock(), response=mock_response
        )
        
        mock_cli_manager._get_http_client.return_value = mock_client
        
        with pytest.raises(httpx.HTTPStatusError):
            await api_client.get_task_status("nonexistent")
    
    @pytest.mark.asyncio
    async def test_api_client_generate_response_success(self, api_client, mock_cli_manager):
        """Test successful response generation."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "response": "Generated response text",
            "done": True
        }
        mock_client.post.return_value = mock_response
        
        mock_cli_manager._get_http_client.return_value = mock_client
        
        result = await api_client.generate_response(
            "qwen3:8b",
            "Test prompt",
            {"temperature": 0.7}
        )
        
        assert result["response"] == "Generated response text"
        assert result["done"] is True
    
    @pytest.mark.asyncio
    async def test_api_client_generate_response_error(self, api_client, mock_cli_manager):
        """Test response generation with error."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Model error"
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "Server Error", request=Mock(), response=mock_response
        )
        
        mock_cli_manager._get_http_client.return_value = mock_client
        
        with pytest.raises(httpx.HTTPStatusError):
            await api_client.generate_response("qwen3:8b", "Test prompt", {})


if __name__ == '__main__':
    pytest.main([__file__])