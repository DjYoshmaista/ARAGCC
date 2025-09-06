"""
Comprehensive unit tests for shared validation utilities.

Tests cover path validation, URL validation, email validation,
JSON structure validation, and error handling.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

from shared.utils.validators import (
    validate_path, validate_file_path, validate_directory_path,
    validate_url, validate_port, validate_email, validate_json_structure,
    _is_valid_hostname
)


class TestPathValidation:
    """Test suite for path validation functions."""
    
    def test_validate_path_existing_file(self):
        """Test path validation with existing file."""
        with tempfile.NamedTemporaryFile() as tmp_file:
            assert validate_path(tmp_file.name) is True
    
    def test_validate_path_existing_directory(self):
        """Test path validation with existing directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            assert validate_path(tmp_dir) is True
    
    def test_validate_path_nonexistent(self):
        """Test path validation with non-existent path."""
        assert validate_path("/nonexistent/path") is False
    
    def test_validate_path_invalid_input(self):
        """Test path validation with invalid input."""
        assert validate_path("") is False
        assert validate_path(None) is False
    
    def test_validate_file_path_existing_file(self):
        """Test file path validation with existing file."""
        with tempfile.NamedTemporaryFile() as tmp_file:
            assert validate_file_path(tmp_file.name) is True
    
    def test_validate_file_path_directory(self):
        """Test file path validation with directory (should fail)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            assert validate_file_path(tmp_dir) is False
    
    def test_validate_file_path_nonexistent(self):
        """Test file path validation with non-existent file."""
        assert validate_file_path("/nonexistent/file.txt") is False
    
    def test_validate_directory_path_existing_directory(self):
        """Test directory path validation with existing directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            assert validate_directory_path(tmp_dir) is True
    
    def test_validate_directory_path_file(self):
        """Test directory path validation with file (should fail)."""
        with tempfile.NamedTemporaryFile() as tmp_file:
            assert validate_directory_path(tmp_file.name) is False
    
    def test_validate_directory_path_nonexistent(self):
        """Test directory path validation with non-existent directory."""
        assert validate_directory_path("/nonexistent/directory") is False
    
    @patch('os.access')
    def test_validate_path_permission_denied(self, mock_access):
        """Test path validation when access is denied."""
        mock_access.return_value = False
        
        with tempfile.NamedTemporaryFile() as tmp_file:
            # File exists but access is denied
            assert validate_path(tmp_file.name) is False


class TestURLValidation:
    """Test suite for URL validation functions."""
    
    def test_validate_url_valid_http(self):
        """Test URL validation with valid HTTP URLs."""
        valid_urls = [
            "http://localhost",
            "http://localhost:8080",
            "http://example.com",
            "http://example.com:3000",
            "http://192.168.1.1",
            "http://192.168.1.1:8080",
            "http://example.com/path",
            "http://example.com/path?query=value"
        ]
        
        for url in valid_urls:
            is_valid, error = validate_url(url)
            assert is_valid is True, f"URL {url} should be valid, got error: {error}"
            assert error is None
    
    def test_validate_url_valid_https(self):
        """Test URL validation with valid HTTPS URLs."""
        valid_urls = [
            "https://localhost",
            "https://example.com",
            "https://api.example.com:443",
            "https://subdomain.example.com/api/v1"
        ]
        
        for url in valid_urls:
            is_valid, error = validate_url(url)
            assert is_valid is True, f"URL {url} should be valid, got error: {error}"
            assert error is None
    
    def test_validate_url_invalid_scheme(self):
        """Test URL validation with invalid schemes."""
        invalid_urls = [
            "ftp://example.com",
            "file:///path/to/file",
            "example.com",
            "//example.com"
        ]
        
        for url in invalid_urls:
            is_valid, error = validate_url(url)
            assert is_valid is False, f"URL {url} should be invalid"
            assert "scheme" in error.lower()
    
    def test_validate_url_missing_hostname(self):
        """Test URL validation with missing hostname."""
        invalid_urls = [
            "http://",
            "https://",
            "http://:8080"
        ]
        
        for url in invalid_urls:
            is_valid, error = validate_url(url)
            assert is_valid is False, f"URL {url} should be invalid"
            assert "hostname" in error.lower()
    
    def test_validate_url_invalid_port(self):
        """Test URL validation with invalid ports."""
        invalid_urls = [
            "http://example.com:0",
            "http://example.com:70000",
            "http://example.com:abc"
        ]
        
        for url in invalid_urls:
            is_valid, error = validate_url(url)
            assert is_valid is False, f"URL {url} should be invalid"
            assert "port" in error.lower()
    
    def test_validate_url_empty_or_none(self):
        """Test URL validation with empty or None input."""
        invalid_inputs = ["", None, 123, [], {}]
        
        for input_val in invalid_inputs:
            is_valid, error = validate_url(input_val)
            assert is_valid is False
            assert "non-empty string" in error
    
    def test_validate_url_whitespace_trimming(self):
        """Test URL validation trims whitespace."""
        url_with_whitespace = "  http://example.com  "
        is_valid, error = validate_url(url_with_whitespace)
        assert is_valid is True
        assert error is None


class TestHostnameValidation:
    """Test suite for hostname validation."""
    
    def test_is_valid_hostname_valid(self):
        """Test hostname validation with valid hostnames."""
        valid_hostnames = [
            "localhost",
            "example.com",
            "api.example.com",
            "sub-domain.example.com",
            "test123.example-site.com"
        ]
        
        for hostname in valid_hostnames:
            assert _is_valid_hostname(hostname) is True, f"Hostname {hostname} should be valid"
    
    def test_is_valid_hostname_invalid(self):
        """Test hostname validation with invalid hostnames."""
        invalid_hostnames = [
            "",
            ".",
            "example..com",
            "-example.com",
            "example-.com",
            "a" * 64 + ".com",  # Label too long
            "a" * 254  # Hostname too long
        ]
        
        for hostname in invalid_hostnames:
            assert _is_valid_hostname(hostname) is False, f"Hostname {hostname} should be invalid"
    
    def test_is_valid_hostname_localhost(self):
        """Test hostname validation with localhost variations."""
        assert _is_valid_hostname("localhost") is True
        assert _is_valid_hostname("LOCALHOST") is True
        assert _is_valid_hostname("LocalHost") is True


class TestPortValidation:
    """Test suite for port validation."""
    
    def test_validate_port_valid_integers(self):
        """Test port validation with valid integer ports."""
        valid_ports = [1, 80, 443, 8080, 65535]
        
        for port in valid_ports:
            is_valid, error = validate_port(port)
            assert is_valid is True, f"Port {port} should be valid, got error: {error}"
            assert error is None
    
    def test_validate_port_valid_strings(self):
        """Test port validation with valid string ports."""
        valid_ports = ["1", "80", "443", "8080", "65535"]
        
        for port in valid_ports:
            is_valid, error = validate_port(port)
            assert is_valid is True, f"Port {port} should be valid, got error: {error}"
            assert error is None
    
    def test_validate_port_invalid_range(self):
        """Test port validation with out-of-range ports."""
        invalid_ports = [0, -1, 65536, 70000]
        
        for port in invalid_ports:
            is_valid, error = validate_port(port)
            assert is_valid is False, f"Port {port} should be invalid"
            assert "Port must be" in error
    
    def test_validate_port_invalid_types(self):
        """Test port validation with invalid types."""
        invalid_ports = ["abc", 3.14, [], {}, None]
        
        for port in invalid_ports:
            is_valid, error = validate_port(port)
            assert is_valid is False, f"Port {port} should be invalid"
            assert "valid integer" in error


class TestEmailValidation:
    """Test suite for email validation."""
    
    def test_validate_email_valid(self):
        """Test email validation with valid emails."""
        valid_emails = [
            "test@example.com",
            "user.name@example.com",
            "user+tag@example.com",
            "user_name@example-site.com",
            "123@example.com",
            "test@subdomain.example.com"
        ]
        
        for email in valid_emails:
            is_valid, error = validate_email(email)
            assert is_valid is True, f"Email {email} should be valid, got error: {error}"
            assert error is None
    
    def test_validate_email_invalid_format(self):
        """Test email validation with invalid formats."""
        invalid_emails = [
            "invalid",
            "@example.com",
            "test@",
            "test..email@example.com",
            "test@example",
            "test@.com",
            "test@example..com"
        ]
        
        for email in invalid_emails:
            is_valid, error = validate_email(email)
            assert is_valid is False, f"Email {email} should be invalid"
            assert "Invalid email format" in error
    
    def test_validate_email_too_long(self):
        """Test email validation with too long local or domain parts."""
        # Local part too long (>64 chars)
        long_local = "a" * 65 + "@example.com"
        is_valid, error = validate_email(long_local)
        assert is_valid is False
        assert "local part too long" in error
        
        # Domain part too long (>253 chars)
        long_domain = "test@" + "a" * 250 + ".com"
        is_valid, error = validate_email(long_domain)
        assert is_valid is False
        assert "domain too long" in error
    
    def test_validate_email_empty_or_none(self):
        """Test email validation with empty or None input."""
        invalid_inputs = ["", None, 123, [], {}]
        
        for input_val in invalid_inputs:
            is_valid, error = validate_email(input_val)
            assert is_valid is False
            assert "non-empty string" in error
    
    def test_validate_email_whitespace_trimming(self):
        """Test email validation trims whitespace."""
        email_with_whitespace = "  test@example.com  "
        is_valid, error = validate_email(email_with_whitespace)
        assert is_valid is True
        assert error is None


class TestJSONStructureValidation:
    """Test suite for JSON structure validation."""
    
    def test_validate_json_structure_valid(self):
        """Test JSON structure validation with valid data."""
        data = {
            "name": "test",
            "value": 123,
            "nested": {"key": "value"}
        }
        required_fields = ["name", "value"]
        
        is_valid, error = validate_json_structure(data, required_fields)
        assert is_valid is True
        assert error is None
    
    def test_validate_json_structure_missing_fields(self):
        """Test JSON structure validation with missing required fields."""
        data = {"name": "test"}
        required_fields = ["name", "value", "type"]
        
        is_valid, error = validate_json_structure(data, required_fields)
        assert is_valid is False
        assert "Missing required fields" in error
        assert "value" in error
        assert "type" in error
    
    def test_validate_json_structure_not_dict(self):
        """Test JSON structure validation with non-dictionary data."""
        invalid_data = ["list", "data"]
        required_fields = ["name"]
        
        is_valid, error = validate_json_structure(invalid_data, required_fields)
        assert is_valid is False
        assert "Data must be a dictionary" in error
    
    def test_validate_json_structure_empty_requirements(self):
        """Test JSON structure validation with no required fields."""
        data = {"anything": "goes"}
        required_fields = []
        
        is_valid, error = validate_json_structure(data, required_fields)
        assert is_valid is True
        assert error is None
    
    def test_validate_json_structure_extra_fields_ok(self):
        """Test JSON structure validation allows extra fields."""
        data = {
            "required_field": "value",
            "extra_field": "also ok",
            "another_extra": 123
        }
        required_fields = ["required_field"]
        
        is_valid, error = validate_json_structure(data, required_fields)
        assert is_valid is True
        assert error is None


if __name__ == '__main__':
    pytest.main([__file__])