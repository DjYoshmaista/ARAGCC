#!/usr/bin/env python3
"""
Demonstration script for folder ingestion functionality
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the orchestrator directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services', 'orchestrator'))

from text_chunker import TextChunker

def demonstrate_chunking():
    """Demonstrate the text chunking functionality"""
    
    # Create a text chunker
    chunker = TextChunker(chunk_size=1000, overlap_size=50)
    
    # Read a sample document
    sample_file = Path("txt/sample_document.txt")
    if sample_file.exists():
        with open(sample_file, 'r') as f:
            content = f.read()
        
        print("Sample document content:")
        print(content[:200] + "..." if len(content) > 200 else content)
        print()
        
        # Chunk the text
        chunks = chunker.chunk_text_by_sentences(content)
        
        print(f"Document chunked into {len(chunks)} chunks:")
        for i, (chunk_text, token_count, overlap_tokens) in enumerate(chunks):
            print(f"Chunk {i+1}:")
            print(f"  Token count: {token_count}")
            print(f"  Overlap tokens: {overlap_tokens}")
            print(f"  Content: {chunk_text[:100]}..." if len(chunk_text) > 100 else chunk_text)
            print()
    else:
        print("Sample document not found. Creating a sample text for demonstration.")
        
        # Create a sample text
        sample_text = """
        This is a sample document for demonstrating the text chunking functionality. 
        The text chunker divides documents into smaller segments for processing.
        Each chunk has a specified size and overlap with adjacent chunks.
        This ensures that context is preserved across chunk boundaries.
        The chunking algorithm tries to split at sentence boundaries when possible.
        This helps maintain the meaning and coherence of each chunk.
        Overlapping chunks help ensure that information is not lost at boundaries.
        The system can process multiple files concurrently for efficiency.
        Parallel processing is controlled by semaphores to prevent resource exhaustion.
        Progress is tracked with dual progress bars for files and chunks.
        The system maintains metadata about each document and its chunks.
        This metadata includes file paths, folder structure, and chunk relationships.
        Documents are stored in a hierarchical structure with chunk linking.
        Vector embeddings are generated for each chunk using GPU acceleration.
        The embeddings are stored in a vector database for semantic search.
        The system supports recursive folder processing with file type filtering.
        Configuration parameters allow customization of chunk size and overlap.
        Error handling ensures robust processing even with problematic files.
        Statistics are collected to monitor processing performance and issues.
        """.strip()
        
        # Chunk the text
        chunks = chunker.chunk_text_by_sentences(sample_text)
        
        print("Sample text chunked into chunks:")
        for i, (chunk_text, token_count, overlap_tokens) in enumerate(chunks):
            print(f"Chunk {i+1}:")
            print(f"  Token count: {token_count}")
            print(f"  Overlap tokens: {overlap_tokens}")
            print(f"  Content: {chunk_text}")
            print()

def demonstrate_concurrent_processing():
    """Demonstrate the concept of concurrent file processing"""
    
    print("Concurrent Processing Demonstration:")
    print("The system uses semaphores to control concurrent file and chunk processing.")
    print("This prevents resource exhaustion while maintaining processing efficiency.")
    print()
    print("File-level concurrency: Process multiple files simultaneously")
    print("Chunk-level concurrency: Process chunks within files simultaneously")
    print()
    print("Semaphore-controlled concurrency ensures:")
    print("1. Resource limits are respected")
    print("2. System stability is maintained")
    print("3. Processing efficiency is optimized")
    print()

def demonstrate_progress_tracking():
    """Demonstrate the concept of progress tracking"""
    
    print("Progress Tracking Demonstration:")
    print("The system uses dual progress bars to track processing:")
    print("1. File progress bar: Shows processed files")
    print("2. Chunk progress bar: Shows processed chunks")
    print()
    print("Progress bars are updated every 100ms by a dedicated controller thread.")
    print("This provides real-time feedback on processing status.")
    print("Statistics are collected throughout processing for monitoring.")
    print()

def main():
    """Main demonstration function"""
    
    print("AgenticRAG Folder Ingestion System Demonstration")
    print("=" * 50)
    print()
    
    demonstrate_chunking()
    demonstrate_concurrent_processing()
    demonstrate_progress_tracking()
    
    print("Database Schema:")
    print("The system uses a hierarchical document-chunk schema:")
    print("1. Documents Table: Stores document metadata (file paths, folder structure)")
    print("2. Chunks Table: Stores chunk content and relationships with overlap tracking")
    print("3. Qdrant Collection: Stores vector embeddings for semantic search")
    print()
    print("This structure allows efficient querying and maintains document context.")

if __name__ == "__main__":
    main()