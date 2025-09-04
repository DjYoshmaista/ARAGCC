"""
Text chunking utilities for the AgenticRAG system
"""

import re
from typing import List, Tuple

class TextChunker:
    """Handles chunking of text documents with overlap"""
    
    def __init__(self, chunk_size: int = 5000, overlap_size: int = 32):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
    
    def estimate_token_count(self, text: str) -> int:
        """
        Roughly estimate token count (simplified approach)
        In a production system, you'd use a proper tokenizer
        """
        # Average token length is roughly 4 characters
        return len(text) // 4
    
    def chunk_text(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Chunk text into overlapping segments
        Returns list of (chunk_text, token_count, overlap_tokens)
        """
        chunks = []
        text_length = len(text)
        
        if text_length == 0:
            return chunks
        
        # Convert chunk size and overlap to character counts (rough approximation)
        char_chunk_size = self.chunk_size * 4
        char_overlap_size = self.overlap_size * 4
        
        start_pos = 0
        chunk_number = 0
        
        while start_pos < text_length:
            # Calculate end position
            end_pos = min(start_pos + char_chunk_size, text_length)
            
            # Extract chunk
            chunk_text = text[start_pos:end_pos]
            token_count = self.estimate_token_count(chunk_text)
            
            # Calculate overlap for this chunk
            overlap_tokens = 0
            if chunk_number > 0:  # Not the first chunk
                overlap_tokens = min(self.overlap_size, token_count)
            
            chunks.append((chunk_text, token_count, overlap_tokens))
            
            # Move to next chunk position
            if end_pos >= text_length:
                break
                
            # Move start position, accounting for overlap
            next_start = start_pos + char_chunk_size - char_overlap_size
            if next_start >= text_length:
                break
                
            start_pos = next_start
            chunk_number += 1
        
        return chunks
    
    def chunk_text_by_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Chunk text by sentences to preserve meaning
        Returns list of (chunk_text, token_count, overlap_tokens)
        """
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_token_count = 0
        chunk_number = 0
        
        for sentence in sentences:
            sentence_token_count = self.estimate_token_count(sentence)
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_token_count + sentence_token_count > self.chunk_size and current_chunk:
                chunk_text = '. '.join(current_chunk) + '.'
                overlap_tokens = 0
                if chunk_number > 0:
                    overlap_tokens = min(self.overlap_size, current_token_count)
                
                chunks.append((chunk_text, current_token_count, overlap_tokens))
                
                # Start new chunk with overlap from previous chunk
                # For simplicity, we'll just start fresh, but in a real implementation
                # you'd want to include some overlap
                current_chunk = [sentence]
                current_token_count = sentence_token_count
                chunk_number += 1
            else:
                current_chunk.append(sentence)
                current_token_count += sentence_token_count
        
        # Add final chunk
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            overlap_tokens = 0
            if chunk_number > 0:
                overlap_tokens = min(self.overlap_size, current_token_count)
            
            chunks.append((chunk_text, current_token_count, overlap_tokens))
        
        return chunks