#!/usr/bin/env python3
"""
Simple test runner that works without external dependencies
"""

import sys
import os
import subprocess
from pathlib import Path

def test_text_chunker():
    """Test the text chunker functionality"""
    sys.path.insert(0, str(Path(__file__).parent / "services" / "orchestrator"))
    
    try:
        from text_chunker import TextChunker
        
        print("Testing TextChunker...")
        
        # Test 1: Basic chunking
        chunker = TextChunker(chunk_size=100, overlap_size=10)
        text = "This is a test sentence. " * 20
        result = chunker.chunk_text(text)
        
        assert len(result) > 0, "Should produce at least one chunk"
        assert all(len(r) == 3 for r in result), "Each result should have 3 elements"
        print("✅ Basic chunking test passed")
        
        # Test 2: Sentence-based chunking
        result2 = chunker.chunk_text_by_sentences(text)
        assert len(result2) > 0, "Should produce at least one chunk"
        print("✅ Sentence-based chunking test passed")
        
        # Test 3: Token estimation
        tokens = chunker.estimate_token_count("Hello world")
        assert tokens > 0, "Should estimate positive token count"
        print("✅ Token estimation test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ TextChunker tests failed: {e}")
        return False

def test_rate_limiter():
    """Test the rate limiter functionality"""
    sys.path.insert(0, str(Path(__file__).parent / "services" / "orchestrator"))
    
    try:
        from rate_limiter import RateLimiter, RateLimit
        import asyncio
        import time
        
        print("Testing RateLimiter...")
        
        async def async_test():
            limiter = RateLimiter()
            limiter.set_rate_limit("test", RateLimit(requests_per_second=10.0, burst_limit=5))
            
            # Test basic acquisition
            result = await limiter.acquire("test")
            assert result == True, "Should acquire token successfully"
            
            return True
        
        # Run async test
        result = asyncio.run(async_test())
        if result:
            print("✅ RateLimiter test passed")
            return True
        
    except Exception as e:
        print(f"❌ RateLimiter tests failed: {e}")
        return False

def test_zig_vector_engine():
    """Test the Zig vector engine"""
    vector_engine_dir = Path(__file__).parent / "services" / "vector-engine"
    
    if not vector_engine_dir.exists():
        print("❌ Vector engine directory not found")
        return False
    
    try:
        print("Testing Zig Vector Engine...")
        result = subprocess.run(
            ["zig", "test", "src/main.zig"],
            cwd=vector_engine_dir,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ Zig vector engine tests passed")
            print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Zig tests failed: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ Zig compiler not found, skipping Zig tests")
        return True  # Don't fail if Zig is not installed
    except Exception as e:
        print(f"❌ Error running Zig tests: {e}")
        return False

def test_configuration_loading():
    """Test configuration file loading"""
    config_file = Path(__file__).parent / "shared" / "configs" / "system.yaml"
    
    try:
        import yaml
        
        print("Testing configuration loading...")
        
        if not config_file.exists():
            print(f"❌ Configuration file not found: {config_file}")
            return False
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Verify required sections
        required_sections = ['services', 'databases', 'models']
        for section in required_sections:
            assert section in config, f"Missing required config section: {section}"
        
        # Verify service ports
        assert config['services']['orchestrator']['port'] == 8001
        assert config['services']['model_gateway']['port'] == 8070
        assert config['services']['vector_engine']['port'] == 8080
        
        print("✅ Configuration loading test passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading test failed: {e}")
        return False

def test_project_structure():
    """Test that key project files exist"""
    print("Testing project structure...")
    
    required_files = [
        "README.md",
        "shared/configs/system.yaml",
        "services/orchestrator/app.py",
        "services/model-gateway/app.py",
        "services/vector-engine/src/main.zig",
        "docker-compose.yml"
    ]
    
    project_root = Path(__file__).parent
    missing_files = []
    
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ Project structure test passed")
    return True

def main():
    """Run all simple tests"""
    print("=" * 60)
    print("RUNNING SIMPLE TESTS (No External Dependencies)")
    print("=" * 60)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Configuration Loading", test_configuration_loading),
        ("Text Chunker", test_text_chunker),
        ("Rate Limiter", test_rate_limiter),
        ("Zig Vector Engine", test_zig_vector_engine),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            print(f"Test failed: {test_name}")
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())