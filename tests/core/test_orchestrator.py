"""
Comprehensive unit tests for the orchestrator module.

Tests cover task management, dependency resolution, execution flow,
and API endpoints with proper error handling and edge cases.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import uuid
import json
import httpx

from core.orchestrator.app import (
    Task, TaskStatus, TaskPriority, Agent, Orchestrator, TaskCreateRequest,
    load_config
)


class TestTaskModel:
    """Test suite for Task data model."""
    
    def test_task_creation_defaults(self):
        """Test task creation with default values."""
        task = Task(type="test", description="Test task")
        
        assert task.type == "test"
        assert task.description == "Test task"
        assert task.status == TaskStatus.PENDING
        assert task.priority == TaskPriority.NORMAL
        assert isinstance(task.id, str)
        assert len(task.id) > 0
        assert task.retry_count == 0
        assert task.max_retries == 3
        assert task.timeout_seconds is None
    
    def test_task_creation_with_parameters(self):
        """Test task creation with custom parameters."""
        now = datetime.now(timezone.utc)
        deadline = now + timedelta(hours=1)
        
        task = Task(
            type="llm_inference",
            description="Generate response",
            parameters={"model": "qwen3:8b", "temperature": 0.7},
            dependencies=["task1", "task2"],
            priority=TaskPriority.HIGH,
            deadline=deadline,
            timeout_seconds=300
        )
        
        assert task.type == "llm_inference"
        assert task.description == "Generate response"
        assert task.parameters == {"model": "qwen3:8b", "temperature": 0.7}
        assert task.dependencies == ["task1", "task2"]
        assert task.priority == TaskPriority.HIGH
        assert task.deadline == deadline
        assert task.timeout_seconds == 300
    
    def test_task_to_dict(self):
        """Test task serialization to dictionary."""
        task = Task(
            type="test",
            description="Test task",
            priority=TaskPriority.HIGH,
            status=TaskStatus.RUNNING
        )
        
        task_dict = task.to_dict()
        
        assert task_dict['type'] == "test"
        assert task_dict['description'] == "Test task"
        assert task_dict['priority'] == TaskPriority.HIGH.value
        assert task_dict['status'] == TaskStatus.RUNNING.value
        assert isinstance(task_dict['created_at'], str)  # Should be ISO format
    
    def test_task_is_overdue(self):
        """Test overdue detection."""
        # Task with deadline in future
        future_deadline = datetime.now(timezone.utc) + timedelta(hours=1)
        task_future = Task(type="test", description="Future task", deadline=future_deadline)
        assert not task_future.is_overdue()
        
        # Task with deadline in past
        past_deadline = datetime.now(timezone.utc) - timedelta(hours=1)
        task_past = Task(type="test", description="Past task", deadline=past_deadline)
        assert task_past.is_overdue()
        
        # Task with no deadline
        task_no_deadline = Task(type="test", description="No deadline task")
        assert not task_no_deadline.is_overdue()
    
    def test_task_execution_time(self):
        """Test execution time calculation."""
        task = Task(type="test", description="Test task")
        
        # No execution time yet
        assert task.get_execution_time() is None
        
        # Set start time
        start_time = datetime.now(timezone.utc)
        task.started_at = start_time
        assert task.get_execution_time() is None  # Still no end time
        
        # Set completion time
        end_time = start_time + timedelta(seconds=30)
        task.completed_at = end_time
        execution_time = task.get_execution_time()
        
        assert execution_time is not None
        assert execution_time.total_seconds() == 30


class TestAgentModel:
    """Test suite for Agent data model."""
    
    def test_agent_creation_defaults(self):
        """Test agent creation with default values."""
        agent = Agent(type="test_agent")
        
        assert agent.type == "test_agent"
        assert agent.status == "idle"
        assert agent.capabilities == []
        assert agent.current_task is None
        assert agent.performance_score == 1.0
        assert isinstance(agent.id, str)
        assert len(agent.id) > 0
    
    def test_agent_creation_with_parameters(self):
        """Test agent creation with custom parameters."""
        agent = Agent(
            type="llm_agent",
            status="busy",
            capabilities=["text_generation", "summarization"],
            current_task="task123",
            performance_score=0.95
        )
        
        assert agent.type == "llm_agent"
        assert agent.status == "busy"
        assert agent.capabilities == ["text_generation", "summarization"]
        assert agent.current_task == "task123"
        assert agent.performance_score == 0.95


class TestTaskCreateRequest:
    """Test suite for TaskCreateRequest validation."""
    
    def test_valid_request(self):
        """Test valid task creation request."""
        request = TaskCreateRequest(
            task_type="llm_inference",
            description="Generate response",
            parameters={"model": "qwen3:8b"},
            dependencies=["task1"],
            priority="high"
        )
        
        assert request.task_type == "llm_inference"
        assert request.description == "Generate response"
        assert request.parameters == {"model": "qwen3:8b"}
        assert request.dependencies == ["task1"]
        assert request.priority == "high"
    
    def test_minimal_request(self):
        """Test task creation request with minimal fields."""
        request = TaskCreateRequest(
            task_type="test",
            description="Test task"
        )
        
        assert request.task_type == "test"
        assert request.description == "Test task"
        assert request.parameters == {}
        assert request.dependencies == []
        assert request.priority == "normal"
    
    def test_request_with_deadline(self):
        """Test task creation request with deadline."""
        deadline_str = "2024-12-31T23:59:59Z"
        request = TaskCreateRequest(
            task_type="test",
            description="Test task",
            deadline=deadline_str
        )
        
        assert request.deadline == deadline_str


class TestLoadConfig:
    """Test suite for configuration loading."""
    
    @patch('builtins.open')
    @patch('yaml.safe_load')
    def test_load_config_success(self, mock_yaml_load, mock_open):
        """Test successful configuration loading."""
        mock_config = {
            'services': {'model_gateway': {'host': 'localhost', 'port': 8070}},
            'databases': {'postgresql': {'host': 'localhost'}}
        }
        mock_yaml_load.return_value = mock_config
        mock_open.return_value.__enter__.return_value = Mock()
        
        config = load_config()
        
        assert config == mock_config
        mock_open.assert_called_once()
        mock_yaml_load.assert_called_once()
    
    @patch('builtins.open')
    def test_load_config_file_not_found(self, mock_open):
        """Test configuration loading when file doesn't exist."""
        mock_open.side_effect = FileNotFoundError("Config file not found")
        
        with pytest.raises(FileNotFoundError):
            load_config()
    
    @patch('builtins.open')
    @patch('yaml.safe_load')
    def test_load_config_invalid_yaml(self, mock_yaml_load, mock_open):
        """Test configuration loading with invalid YAML."""
        mock_yaml_load.side_effect = yaml.YAMLError("Invalid YAML")
        mock_open.return_value.__enter__.return_value = Mock()
        
        with pytest.raises(yaml.YAMLError):
            load_config()
    
    @patch('builtins.open')
    @patch('yaml.safe_load')
    def test_load_config_invalid_format(self, mock_yaml_load, mock_open):
        """Test configuration loading with non-dict result."""
        mock_yaml_load.return_value = "not a dictionary"
        mock_open.return_value.__enter__.return_value = Mock()
        
        with pytest.raises(ValueError, match="Configuration file must contain a YAML dictionary"):
            load_config()


class TestOrchestrator:
    """Test suite for Orchestrator class."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance for testing."""
        with patch('core.orchestrator.app.load_config') as mock_config:
            mock_config.return_value = {
                'services': {
                    'model_gateway': {'host': 'localhost', 'port': 8070}
                }
            }
            orchestrator = Orchestrator()
            return orchestrator
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert isinstance(orchestrator.tasks, dict)
        assert isinstance(orchestrator.agents, dict)
        assert isinstance(orchestrator.task_queue, list)
        assert not orchestrator.running
        assert orchestrator.model_gateway_url.startswith("http://")
    
    def test_add_task_without_dependencies(self, orchestrator):
        """Test adding task without dependencies."""
        task = Task(type="test", description="Test task")
        
        task_id = orchestrator.add_task(task)
        
        assert task_id == task.id
        assert task_id in orchestrator.tasks
        assert orchestrator.tasks[task_id] == task
        assert task in orchestrator.task_queue
    
    def test_add_task_with_dependencies(self, orchestrator):
        """Test adding task with dependencies."""
        # Add first task
        task1 = Task(type="test", description="First task")
        task1_id = orchestrator.add_task(task1)
        
        # Add second task depending on first
        task2 = Task(type="test", description="Second task", dependencies=[task1_id])
        task2_id = orchestrator.add_task(task2)
        
        assert task2_id in orchestrator.tasks
        assert task2 not in orchestrator.task_queue  # Should not be queued yet
        
        # Complete first task
        task1.status = TaskStatus.COMPLETED
        
        # Check dependencies again
        assert orchestrator.check_dependencies(task2)
    
    def test_check_dependencies_success(self, orchestrator):
        """Test dependency checking with completed dependencies."""
        # Create and complete dependency task
        dep_task = Task(type="test", description="Dependency task")
        dep_task.status = TaskStatus.COMPLETED
        orchestrator.tasks[dep_task.id] = dep_task
        
        # Create task with dependency
        task = Task(type="test", description="Test task", dependencies=[dep_task.id])
        
        assert orchestrator.check_dependencies(task)
    
    def test_check_dependencies_failure(self, orchestrator):
        """Test dependency checking with incomplete dependencies."""
        # Create incomplete dependency task
        dep_task = Task(type="test", description="Dependency task")
        dep_task.status = TaskStatus.RUNNING
        orchestrator.tasks[dep_task.id] = dep_task
        
        # Create task with dependency
        task = Task(type="test", description="Test task", dependencies=[dep_task.id])
        
        assert not orchestrator.check_dependencies(task)
    
    def test_check_dependencies_missing_task(self, orchestrator):
        """Test dependency checking with missing dependency task."""
        task = Task(type="test", description="Test task", dependencies=["nonexistent"])
        
        assert not orchestrator.check_dependencies(task)
    
    def test_get_task_exists(self, orchestrator):
        """Test getting existing task."""
        task = Task(type="test", description="Test task")
        orchestrator.tasks[task.id] = task
        
        retrieved_task = orchestrator.get_task(task.id)
        
        assert retrieved_task == task
    
    def test_get_task_not_exists(self, orchestrator):
        """Test getting non-existent task."""
        retrieved_task = orchestrator.get_task("nonexistent")
        
        assert retrieved_task is None
    
    def test_get_system_status(self, orchestrator):
        """Test system status retrieval."""
        # Add some tasks
        task1 = Task(type="test", description="Pending task")
        task2 = Task(type="test", description="Running task")
        task2.status = TaskStatus.RUNNING
        task3 = Task(type="test", description="Completed task")
        task3.status = TaskStatus.COMPLETED
        
        orchestrator.tasks[task1.id] = task1
        orchestrator.tasks[task2.id] = task2
        orchestrator.tasks[task3.id] = task3
        orchestrator.task_queue = [task1]
        
        # Add an agent
        agent = Agent(type="test_agent")
        orchestrator.agents[agent.id] = agent
        
        status = orchestrator.get_system_status()
        
        assert status['total_tasks'] == 3
        assert status['queue_length'] == 1
        assert status['status_breakdown']['pending'] == 1
        assert status['status_breakdown']['running'] == 1
        assert status['status_breakdown']['completed'] == 1
        assert status['agents_count'] == 1
        assert not status['running']
    
    @pytest.mark.asyncio
    async def test_start_stop(self, orchestrator):
        """Test orchestrator start and stop."""
        assert not orchestrator.running
        
        await orchestrator.start()
        assert orchestrator.running
        
        await orchestrator.stop()
        assert not orchestrator.running
    
    @pytest.mark.asyncio
    async def test_execute_task_generic(self, orchestrator):
        """Test generic task execution."""
        task = Task(type="generic", description="Generic test task")
        orchestrator.tasks[task.id] = task
        
        await orchestrator.execute_task(task)
        
        assert task.status == TaskStatus.COMPLETED
        assert task.started_at is not None
        assert task.completed_at is not None
        assert task.result is not None
        assert "Completed generic task" in task.result
    
    @pytest.mark.asyncio
    async def test_execute_task_failure(self, orchestrator):
        """Test task execution with failure."""
        task = Task(type="nonexistent_type", description="This should fail")
        orchestrator.tasks[task.id] = task
        
        # Mock handle_generic_task to raise exception
        with patch.object(orchestrator, 'handle_generic_task', side_effect=Exception("Test error")):
            await orchestrator.execute_task(task)
        
        assert task.status == TaskStatus.FAILED
        assert task.error == "Test error"
        assert task.completed_at is not None
    
    @pytest.mark.asyncio
    async def test_handle_llm_task_success(self, orchestrator):
        """Test successful LLM task handling."""
        task = Task(
            type="llm_inference",
            description="LLM task",
            parameters={
                "model": "qwen3:8b",
                "prompt": "Test prompt",
                "model_params": {"temperature": 0.7}
            }
        )
        
        mock_response = Mock()
        mock_response.json.return_value = {"response": "Generated response"}
        mock_response.raise_for_status = Mock()
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            result = await orchestrator.handle_llm_task(task)
            
            assert result == "Generated response"
            mock_client.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_llm_task_http_error(self, orchestrator):
        """Test LLM task handling with HTTP error."""
        task = Task(
            type="llm_inference",
            description="LLM task",
            parameters={"model": "qwen3:8b", "prompt": "Test prompt"}
        )
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post.side_effect = httpx.HTTPStatusError(
                "Error", request=Mock(), response=Mock(status_code=500, text="Server Error")
            )
            mock_client_class.return_value = mock_client
            
            with pytest.raises(Exception, match="Model gateway returned error"):
                await orchestrator.handle_llm_task(task)
    
    @pytest.mark.asyncio
    async def test_handle_llm_task_network_error(self, orchestrator):
        """Test LLM task handling with network error."""
        task = Task(
            type="llm_inference",
            description="LLM task",
            parameters={"model": "qwen3:8b", "prompt": "Test prompt"}
        )
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post.side_effect = httpx.ConnectError("Network error")
            mock_client_class.return_value = mock_client
            
            with pytest.raises(Exception, match="Network error calling model gateway"):
                await orchestrator.handle_llm_task(task)
    
    @pytest.mark.asyncio
    async def test_check_dependent_tasks(self, orchestrator):
        """Test checking for dependent tasks after completion."""
        # Create dependency task
        dep_task = Task(type="test", description="Dependency task")
        dep_task.status = TaskStatus.COMPLETED
        orchestrator.tasks[dep_task.id] = dep_task
        
        # Create dependent task
        dependent_task = Task(
            type="test",
            description="Dependent task",
            dependencies=[dep_task.id]
        )
        orchestrator.tasks[dependent_task.id] = dependent_task
        
        assert len(orchestrator.task_queue) == 0
        
        await orchestrator.check_dependent_tasks(dep_task.id)
        
        assert dependent_task in orchestrator.task_queue


class TestOrchestratorTaskRouting:
    """Test suite for task routing in orchestrator."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance for testing."""
        with patch('core.orchestrator.app.load_config') as mock_config:
            mock_config.return_value = {
                'services': {
                    'model_gateway': {'host': 'localhost', 'port': 8070}
                }
            }
            orchestrator = Orchestrator()
            return orchestrator
    
    @pytest.mark.asyncio
    async def test_task_routing_llm_inference(self, orchestrator):
        """Test task routing for LLM inference."""
        task = Task(type="llm_inference", description="LLM task")
        
        with patch.object(orchestrator, 'handle_llm_task', return_value="LLM result") as mock_handler:
            await orchestrator.execute_task(task)
            mock_handler.assert_called_once_with(task)
        
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "LLM result"
    
    @pytest.mark.asyncio
    async def test_task_routing_vector_search(self, orchestrator):
        """Test task routing for vector search."""
        task = Task(type="vector_search", description="Vector task")
        
        with patch.object(orchestrator, 'handle_vector_task', return_value="Vector result") as mock_handler:
            await orchestrator.execute_task(task)
            mock_handler.assert_called_once_with(task)
        
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "Vector result"
    
    @pytest.mark.asyncio
    async def test_task_routing_rag_ingest(self, orchestrator):
        """Test task routing for RAG ingestion."""
        task = Task(type="rag_ingest", description="RAG ingest task")
        
        with patch.object(orchestrator, 'handle_rag_ingest_task', return_value="RAG ingest result") as mock_handler:
            await orchestrator.execute_task(task)
            mock_handler.assert_called_once_with(task)
        
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "RAG ingest result"
    
    @pytest.mark.asyncio
    async def test_task_routing_folder_ingestion(self, orchestrator):
        """Test task routing for folder ingestion."""
        task = Task(type="folder_ingestion", description="Folder ingestion task")
        
        with patch.object(orchestrator, 'handle_folder_ingestion_task', return_value="Folder result") as mock_handler:
            await orchestrator.execute_task(task)
            mock_handler.assert_called_once_with(task)
        
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "Folder result"
    
    @pytest.mark.asyncio
    async def test_task_routing_generic(self, orchestrator):
        """Test task routing for unknown task type."""
        task = Task(type="unknown_type", description="Unknown task")
        
        with patch.object(orchestrator, 'handle_generic_task', return_value="Generic result") as mock_handler:
            await orchestrator.execute_task(task)
            mock_handler.assert_called_once_with(task)
        
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "Generic result"


if __name__ == '__main__':
    pytest.main([__file__])