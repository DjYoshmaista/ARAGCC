#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Add the orchestrator directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'services', 'orchestrator'))

def test_file_discovery(folder_path, recursive=True):
    """Test file discovery for CSV files"""
    print(f"Testing file discovery in: {folder_path}")
    
    supported_extensions = {
        '.txt', '.md', '.py', '.js', '.java', '.cpp', '.c', '.h', '.json', 
        '.xml', '.html', '.css', '.sql', '.csv', '.tsv', '.xls', '.pdf', 
        '.docx', '.doc', '.jsonl'
    }
    
    files = []
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Folder does not exist: {folder_path}")
        return []
    
    if not folder.is_dir():
        print(f"Path is not a directory: {folder_path}")
        return []
    
    print(f"Folder exists and is a directory: {folder_path}")
    
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"
    
    print(f"Searching with pattern: {pattern}")
    
    # Count total files first
    total_files = 0
    csv_files = 0
    
    try:
        for file_path in folder.glob(pattern):
            if file_path.is_file():
                total_files += 1
                if file_path.suffix.lower() in supported_extensions:
                    if not any(part.startswith('.') for part in file_path.parts):
                        if not any(exclude in str(file_path) for exclude in ['node_modules', '__pycache__', '.git', 'build', 'dist']):
                            files.append(str(file_path))
                            if file_path.suffix.lower() == '.csv':
                                csv_files += 1
                                
            # Print progress for large directories
            if total_files % 10000 == 0:
                print(f"Scanned {total_files} files so far...")
                
    except Exception as e:
        print(f"Error during file discovery: {e}")
        return []
    
    print(f"Total files found: {total_files}")
    print(f"Supported files found: {len(files)}")
    print(f"CSV files found: {csv_files}")
    
    # Show first few files as examples
    print("\nFirst 10 supported files:")
    for i, file_path in enumerate(files[:10]):
        print(f"  {i+1}. {file_path}")
    
    return files

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_file_discovery.py <folder_path>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    test_file_discovery(folder_path)