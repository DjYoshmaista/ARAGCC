#!/usr/bin/env python3
"""
Comprehensive unit tests for the Orchestrator service
"""

import asyncio
import pytest
import json
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services', 'orchestrator'))

from app import Orchestrator, Task, TaskStatus, TaskPriority, TaskCreateRequest

class TestOrchestrator:
    """Test cases for the Orchestrator class"""

    @pytest.fixture
    def orchestrator(self):
        """Create an Orchestrator instance for testing"""
        return Orchestrator()

    @pytest.fixture
    def sample_task(self):
        """Create a sample task for testing"""
        return Task(
            type="test_task",
            description="Test task description",
            parameters={"param1": "value1"},
            dependencies=[],
            priority=TaskPriority.NORMAL
        )

    def test_orchestrator_initialization(self, orchestrator):
        """Test Orchestrator initialization"""
        assert orchestrator.tasks == {}
        assert orchestrator.agents == {}
        assert orchestrator.task_queue == []
        assert orchestrator.running == False

    def test_add_task(self, orchestrator, sample_task):
        """Test adding a task to the orchestrator"""
        task_id = orchestrator.add_task(sample_task)
        
        assert task_id == sample_task.id
        assert sample_task.id in orchestrator.tasks
        assert sample_task in orchestrator.task_queue

    def test_check_dependencies_no_deps(self, orchestrator, sample_task):
        """Test dependency check with no dependencies"""
        result = orchestrator.check_dependencies(sample_task)
        assert result == True

    def test_check_dependencies_with_deps(self, orchestrator):
        """Test dependency check with dependencies"""
        # Create dependency task
        dep_task = Task(type="dep", description="Dependency task")
        orchestrator.tasks[dep_task.id] = dep_task
        
        # Create task with dependency
        task_with_dep = Task(
            type="main",
            description="Main task",
            dependencies=[dep_task.id]
        )
        
        # Dependency not completed
        assert orchestrator.check_dependencies(task_with_dep) == False
        
        # Mark dependency as completed
        dep_task.status = TaskStatus.COMPLETED
        assert orchestrator.check_dependencies(task_with_dep) == True

    @pytest.mark.asyncio
    async def test_handle_llm_task(self, orchestrator):
        """Test LLM task handling"""
        task = Task(
            type="llm_inference",
            description="Test LLM task",
            parameters={
                "model": "test-model",
                "prompt": "test prompt",
                "model_params": {"temperature": 0.7}
            }
        )
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": "test response"}
            mock_response.raise_for_status = MagicMock()
            
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
            
            result = await orchestrator.handle_llm_task(task)
            assert result == "test response"

    @pytest.mark.asyncio
    async def test_handle_rag_ingest_task(self, orchestrator):
        """Test RAG ingestion task handling"""
        task = Task(
            type="rag_ingest",
            description="Test RAG ingest",
            parameters={
                "document_id": "test-doc",
                "content": "test content",
                "embedding_model": "test-embed-model"
            }
        )
        
        with patch('httpx.AsyncClient') as mock_client:
            # Mock embedding response
            mock_embed_response = MagicMock()
            mock_embed_response.status_code = 200
            mock_embed_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
            mock_embed_response.raise_for_status = MagicMock()
            
            # Mock vector storage response
            mock_vector_response = MagicMock()
            mock_vector_response.status_code = 200
            mock_vector_response.raise_for_status = MagicMock()
            
            mock_client_instance = mock_client.return_value.__aenter__.return_value
            mock_client_instance.post.side_effect = [mock_embed_response, mock_vector_response]
            
            result = await orchestrator.handle_rag_ingest_task(task)
            assert "test-doc" in result
            assert "successfully ingested" in result

    @pytest.mark.asyncio
    async def test_execute_task_success(self, orchestrator, sample_task):
        """Test successful task execution"""
        with patch.object(orchestrator, 'handle_generic_task', return_value="success"):
            await orchestrator.execute_task(sample_task)
            
            assert sample_task.status == TaskStatus.COMPLETED
            assert sample_task.result == "success"
            assert sample_task.started_at is not None
            assert sample_task.completed_at is not None

    @pytest.mark.asyncio
    async def test_execute_task_failure(self, orchestrator, sample_task):
        """Test task execution failure"""
        with patch.object(orchestrator, 'handle_generic_task', side_effect=Exception("Test error")):
            await orchestrator.execute_task(sample_task)
            
            assert sample_task.status == TaskStatus.FAILED
            assert sample_task.error == "Test error"
            assert sample_task.completed_at is not None

    def test_get_task(self, orchestrator, sample_task):
        """Test getting a task by ID"""
        orchestrator.tasks[sample_task.id] = sample_task
        
        retrieved_task = orchestrator.get_task(sample_task.id)
        assert retrieved_task == sample_task
        
        non_existent = orchestrator.get_task("non-existent-id")
        assert non_existent is None

    def test_get_system_status(self, orchestrator, sample_task):
        """Test getting system status"""
        orchestrator.tasks[sample_task.id] = sample_task
        orchestrator.task_queue.append(sample_task)
        
        status = orchestrator.get_system_status()
        
        assert status["total_tasks"] == 1
        assert status["queue_length"] == 1
        assert status["agents_count"] == 0
        assert status["running"] == False
        assert "status_breakdown" in status

class TestTaskCreateRequest:
    """Test cases for TaskCreateRequest model"""

    def test_task_create_request_validation(self):
        """Test TaskCreateRequest validation"""
        request = TaskCreateRequest(
            task_type="test",
            description="test description",
            parameters={"key": "value"},
            dependencies=["dep1"],
            priority="high"
        )
        
        assert request.task_type == "test"
        assert request.description == "test description"
        assert request.parameters == {"key": "value"}
        assert request.dependencies == ["dep1"]
        assert request.priority == "high"

    def test_task_create_request_defaults(self):
        """Test TaskCreateRequest with default values"""
        request = TaskCreateRequest(
            task_type="test",
            description="test description"
        )
        
        assert request.parameters == {}
        assert request.dependencies == []
        assert request.priority == "normal"

class TestTask:
    """Test cases for Task dataclass"""

    def test_task_initialization(self):
        """Test Task initialization"""
        task = Task()
        
        assert task.id != ""
        assert task.type == ""
        assert task.description == ""
        assert task.parameters == {}
        assert task.dependencies == []
        assert task.priority == TaskPriority.NORMAL
        assert task.status == TaskStatus.PENDING
        assert isinstance(task.created_at, datetime)

    def test_task_with_parameters(self):
        """Test Task with custom parameters"""
        task = Task(
            type="custom",
            description="Custom task",
            parameters={"param": "value"},
            priority=TaskPriority.HIGH
        )
        
        assert task.type == "custom"
        assert task.description == "Custom task"
        assert task.parameters == {"param": "value"}
        assert task.priority == TaskPriority.HIGH

# Integration test for task flow
class TestTaskFlow:
    """Integration tests for complete task flows"""

    @pytest.mark.asyncio
    async def test_decomposition_task_flow(self):
        """Test task decomposition flow"""
        orchestrator = Orchestrator()
        
        task = Task(
            type="decomposition",
            description="Complex task to decompose",
            parameters={"original_task": "Build a web application"}
        )
        
        with patch.object(orchestrator, 'decompose_task_simple', 
                         return_value=["subtask1", "subtask2", "subtask3"]):
            result = await orchestrator.handle_decomposition_task(task)
            
            result_data = json.loads(result)
            assert "subtask_ids" in result_data
            assert len(result_data["subtask_ids"]) == 3

if __name__ == "__main__":
    pytest.main([__file__, "-v"])