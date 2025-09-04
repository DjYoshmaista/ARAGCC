#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'services', 'orchestrator'))

from file_format_handlers import process_csv_file, process_json_file, process_md_file, process_py_file

def test_csv_handler():
    """Test CSV file handler"""
    print("Testing CSV handler...")
    test_file = "/home/yosh/repos/AgenticRAGQCode/test_files/sample.csv"
    chunks = list(process_csv_file(test_file))
    print(f"CSV handler returned {len(chunks)} chunk(s)")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk}")

def test_json_handler():
    """Test JSON file handler"""
    print("\nTesting JSON handler...")
    test_file = "/home/yosh/repos/AgenticRAGQCode/test_files/sample.json"
    chunks = list(process_json_file(test_file))
    print(f"JSON handler returned {len(chunks)} chunk(s)")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk}")

def test_md_handler():
    """Test Markdown file handler"""
    print("\nTesting Markdown handler...")
    test_file = "/home/yosh/repos/AgenticRAGQCode/test_files/sample.md"
    chunks = list(process_md_file(test_file))
    print(f"Markdown handler returned {len(chunks)} chunk(s)")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk}")

def test_py_handler():
    """Test Python file handler"""
    print("\nTesting Python handler...")
    test_file = "/home/yosh/repos/AgenticRAGQCode/test_files/sample.py"
    chunks = list(process_py_file(test_file))
    print(f"Python handler returned {len(chunks)} chunk(s)")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk}")

if __name__ == "__main__":
    test_csv_handler()
    test_json_handler()
    test_md_handler()
    test_py_handler()