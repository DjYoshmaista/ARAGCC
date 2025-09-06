
import unittest
from unittest.mock import patch, MagicMock
import asyncio
import sys
import os
import logging

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from client.app import main

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('test_cli.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

class TestCLI(unittest.TestCase):

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    @patch('client.app.APIClient')
    def test_query_command(self, MockAPIClient):
        """Test the query command."""
        logger.info("--- Running test_query_command ---")
        logger.info("Mocking APIClient...")
        # Mock the API client
        mock_api_client = MockAPIClient.return_value
        mock_api_client.submit_task.return_value = asyncio.Future()
        mock_api_client.submit_task.return_value.set_result({'task_id': '123'})
        logger.info("Mocked submit_task to return task_id: 123")
        mock_api_client.get_task_status.return_value = asyncio.Future()
        mock_api_client.get_task_status.return_value.set_result({'status': 'completed', 'result': 'Test response'})
        logger.info("Mocked get_task_status to return status: completed, result: Test response")

        # Run the CLI with the query command
        logger.info("Running CLI with query command: 'agentic-rag query \'What is the meaning of life?\''")
        with patch('sys.argv', ['agentic-rag', 'query', 'What is the meaning of life?']):
            self.loop.run_until_complete(main(['query', 'What is the meaning of life?']))
        logger.info("CLI command executed.")

        # Assert that the API client was called with the correct arguments
        logger.info("Asserting that submit_task was called with the correct arguments...")
        logger.info(f"submit_task call_args: {mock_api_client.submit_task.call_args}")
        mock_api_client.submit_task.assert_called_with(
            task_type='query',
            description='Query: What is the meaning of life?',
            parameters={
                'prompt': 'What is the meaning of life?',
                'model': 'qwen3:8b',
                'temperature': 0.7,
                'rag_weight': 0.0,
                'top_k': 5,
                'context_window': 4096,
            }
        )
        logger.info("submit_task assertion passed.")
        logger.info("--- Finished test_query_command ---")

    @patch('client.app.APIClient')
    def test_ingest_command(self, MockAPIClient):
        """Test the ingest command."""
        logger.info("--- Running test_ingest_command ---")
        logger.info("Mocking APIClient...")
        # Mock the API client
        mock_api_client = MockAPIClient.return_value
        mock_api_client.submit_task.return_value = asyncio.Future()
        mock_api_client.submit_task.return_value.set_result({'task_id': '456'})
        logger.info("Mocked submit_task to return task_id: 456")
        mock_api_client.get_task_status.return_value = asyncio.Future()
        mock_api_client.get_task_status.return_value.set_result({'status': 'completed'})
        logger.info("Mocked get_task_status to return status: completed")
        mock_api_client._get_http_client.return_value.get.return_value.json.return_value = {}

        # Run the CLI with the ingest command
        logger.info("Running CLI with ingest command: 'agentic-rag ingest ./test_data --recursive'")
        with patch('sys.argv', ['agentic-rag', 'ingest', './test_data', '--recursive']):
            self.loop.run_until_complete(main(['ingest', './test_data', '--recursive']))
        logger.info("CLI command executed.")

        # Assert that the API client was called with the correct arguments
        logger.info("Asserting that submit_task was called with the correct arguments...")
        logger.info(f"submit_task call_args: {mock_api_client.submit_task.call_args}")
        mock_api_client.submit_task.assert_called_with(
            task_type='folder_ingestion',
            description='Ingest folders/files: ./test_data',
            parameters={
                'paths': ['./test_data'],
                'recursive': True,
                'chunk_size': 5000,
                'overlap_size': 32,
                'embedding_model': 'dengcaoQwen3-Embedding-0.6B:Q8_0',
                'max_concurrent_files': 5,
                'max_concurrent_chunks': 10,
                'heartbeat_timeout': 300,
            }
        )
        logger.info("submit_task assertion passed.")
        logger.info("--- Finished test_ingest_command ---")

    @patch('httpx.AsyncClient')
    def test_rag_command(self, MockAsyncClient):
        """Test the rag command."""
        logger.info("--- Running test_rag_command ---")
        logger.info("Mocking httpx.AsyncClient...")
        # Mock the HTTP client
        mock_http_client = MockAsyncClient.return_value
        future_get = asyncio.Future()
        future_get.set_result(MagicMock(status_code=200, json=lambda: [{'id': 'doc1', 'file_name': 'doc1.txt'}]))
        mock_http_client.get.return_value = future_get
        logger.info("Mocked get method to return a document list.")
        future_delete = asyncio.Future()
        future_delete.set_result(MagicMock(status_code=200))
        mock_http_client.delete.return_value = future_delete
        logger.info("Mocked delete method to return status_code 200.")

        mock_http_client.aclose.return_value = asyncio.Future()
        mock_http_client.aclose.return_value.set_result(None)

        # Run the CLI with the rag --list command
        logger.info("Running CLI with rag --list command: 'agentic-rag rag --list'")
        with patch('sys.argv', ['agentic-rag', 'rag', '--list']):
            self.loop.run_until_complete(main(['rag', '--list']))
        logger.info("CLI command executed.")

        # Assert that the HTTP client was called with the correct arguments
        logger.info("Asserting that get was called with the correct arguments...")
        logger.info(f"get call_args: {mock_http_client.get.call_args}")
        mock_http_client.get.assert_called_with('http://localhost:8001/rag/documents')
        logger.info("get assertion passed.")

        # Run the CLI with the rag --delete command
        logger.info("Running CLI with rag --delete command: 'agentic-rag rag --delete doc1'")
        with patch('sys.argv', ['agentic-rag', 'rag', '--delete', 'doc1']):
            self.loop.run_until_complete(main(['rag', '--delete', 'doc1']))
        logger.info("CLI command executed.")

        # Assert that the HTTP client was called with the correct arguments
        logger.info("Asserting that delete was called with the correct arguments...")
        logger.info(f"delete call_args: {mock_http_client.delete.call_args}")
        mock_http_client.delete.assert_called_with('http://localhost:8001/rag/documents/doc1')
        logger.info("delete assertion passed.")
        logger.info("--- Finished test_rag_command ---")

    @patch('client.app.APIClient')
    def test_status_command(self, MockAPIClient):
        """Test the status command."""
        logger.info("--- Running test_status_command ---")
        logger.info("Mocking APIClient...")
        # Mock the API client
        mock_api_client = MockAPIClient.return_value
        future = asyncio.Future()
        future.set_result({'orchestrator': {'status': 'healthy'}})
        mock_api_client.health_check.return_value = future
        logger.info("Mocked health_check to return healthy orchestrator")

        # Run the CLI with the status command
        logger.info("Running CLI with status command: 'agentic-rag status'")
        with patch('sys.argv', ['agentic-rag', 'status']):
            self.loop.run_until_complete(main(['status']))
        logger.info("CLI command executed.")

        # Assert that the API client was called
        logger.info("Asserting that health_check was called...")
        mock_api_client.health_check.assert_called_with()
        logger.info("health_check assertion passed.")
        logger.info("--- Finished test_status_command ---")

    @patch('client.app.CLIManager._load_config')
    @patch('client.app.CLIManager._save_config')
    @patch('client.app.APIClient')
    def test_config_command(self, MockAPIClient, mock_save_config, mock_load_config):
        """Test the config command."""
        logger.info("--- Running test_config_command ---")
        logger.info("Mocking APIClient, _save_config, and _load_config...")
        # Mock the API client
        mock_api_client = MockAPIClient.return_value
        future = asyncio.Future()
        future.set_result({'orchestrator': {'status': 'healthy'}})
        mock_api_client.health_check.return_value = future
        logger.info("Mocked health_check to return healthy orchestrator")

        # Run the CLI with the config --show command
        logger.info("Running CLI with config command: 'agentic-rag config --show'")
        with patch('sys.argv', ['agentic-rag', 'config', '--show']):
            self.loop.run_until_complete(main(['config', '--show']))
        logger.info("CLI command executed.")

        # Run the CLI with the config --set command
        logger.info("Running CLI with config command: 'agentic-rag config --set default_model=test_model'")
        with patch('sys.argv', ['agentic-rag', 'config', '--set', 'default_model=test_model']):
            self.loop.run_until_complete(main(['config', '--set', 'default_model=test_model']))
        logger.info("CLI command executed.")
        mock_save_config.assert_called()

        # Run the CLI with the config --validate command
        logger.info("Running CLI with config command: 'agentic-rag config --validate'")
        with patch('sys.argv', ['agentic-rag', 'config', '--validate']):
            self.loop.run_until_complete(main(['config', '--validate']))
        logger.info("CLI command executed.")
        mock_api_client.health_check.assert_called_with()

        logger.info("--- Finished test_config_command ---")

if __name__ == '__main__':
    unittest.main()
