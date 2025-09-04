#!/usr/bin/env python3
"""
Unit tests for the TextChunker class
"""

import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services', 'orchestrator'))

from text_chunker import TextChunker

class TestTextChunker:
    """Test cases for the TextChunker class"""

    @pytest.fixture
    def chunker(self):
        """Create a TextChunker instance for testing"""
        return TextChunker(chunk_size=1000, overlap_size=50)

    @pytest.fixture
    def sample_text(self):
        """Sample text for testing"""
        return (
            "This is the first sentence. This is the second sentence. "
            "This is the third sentence. This is the fourth sentence. "
            "This is the fifth sentence. This is the sixth sentence. "
            "This is the seventh sentence. This is the eighth sentence."
        )

    def test_chunker_initialization(self, chunker):
        """Test TextChunker initialization"""
        assert chunker.chunk_size == 1000
        assert chunker.overlap_size == 50

    def test_estimate_token_count(self, chunker):
        """Test token count estimation"""
        text = "This is a test sentence."
        estimated_tokens = chunker.estimate_token_count(text)
        
        # Should be roughly text length / 4
        expected_tokens = len(text) // 4
        assert abs(estimated_tokens - expected_tokens) <= 1

    def test_chunk_text_empty_input(self, chunker):
        """Test chunking with empty input"""
        result = chunker.chunk_text("")
        assert result == []

    def test_chunk_text_small_input(self, chunker):
        """Test chunking with small input that fits in one chunk"""
        text = "This is a small text."
        result = chunker.chunk_text(text)
        
        assert len(result) == 1
        chunk_text, token_count, overlap_tokens = result[0]
        assert chunk_text == text
        assert token_count > 0
        assert overlap_tokens == 0  # First chunk has no overlap

    def test_chunk_text_large_input(self):
        """Test chunking with large input that requires multiple chunks"""
        # Create a chunker with very small chunk size for testing
        chunker = TextChunker(chunk_size=10, overlap_size=2)
        
        # Create text that will definitely need multiple chunks
        text = "A" * 200  # 200 characters should create multiple chunks
        result = chunker.chunk_text(text)
        
        assert len(result) > 1
        
        # Check that first chunk has no overlap
        _, _, overlap_tokens = result[0]
        assert overlap_tokens == 0
        
        # Check that subsequent chunks have overlap
        for i in range(1, len(result)):
            _, _, overlap_tokens = result[i]
            assert overlap_tokens > 0

    def test_chunk_text_by_sentences_empty(self, chunker):
        """Test sentence-based chunking with empty input"""
        result = chunker.chunk_text_by_sentences("")
        assert result == []

    def test_chunk_text_by_sentences_single_sentence(self, chunker):
        """Test sentence-based chunking with single sentence"""
        text = "This is a single sentence."
        result = chunker.chunk_text_by_sentences(text)
        
        assert len(result) == 1
        chunk_text, token_count, overlap_tokens = result[0]
        assert "This is a single sentence" in chunk_text
        assert overlap_tokens == 0

    def test_chunk_text_by_sentences_multiple_sentences(self, chunker, sample_text):
        """Test sentence-based chunking with multiple sentences"""
        result = chunker.chunk_text_by_sentences(sample_text)
        
        assert len(result) >= 1
        
        # Check that each chunk contains complete sentences
        for chunk_text, token_count, overlap_tokens in result:
            assert chunk_text.endswith('.')
            assert token_count > 0

    def test_chunk_text_by_sentences_overflow(self):
        """Test sentence-based chunking when sentences exceed chunk size"""
        # Create a chunker with very small chunk size
        chunker = TextChunker(chunk_size=20, overlap_size=5)
        
        # Create text with sentences that will cause chunking
        sentences = []
        for i in range(20):
            sentences.append(f"This is sentence number {i} in our test.")
        
        text = " ".join(sentences)
        result = chunker.chunk_text_by_sentences(text)
        
        assert len(result) > 1
        
        # Verify first chunk has no overlap, others do
        assert result[0][2] == 0  # No overlap for first chunk
        for i in range(1, len(result)):
            assert result[i][2] >= 0  # Overlap tokens >= 0

    def test_chunk_consistency(self, chunker):
        """Test that chunking is consistent across multiple calls"""
        text = "Consistency test. " * 100
        
        result1 = chunker.chunk_text(text)
        result2 = chunker.chunk_text(text)
        
        assert len(result1) == len(result2)
        for i in range(len(result1)):
            assert result1[i][0] == result2[i][0]  # Same chunk text
            assert result1[i][1] == result2[i][1]  # Same token count

    def test_sentence_chunking_consistency(self, chunker, sample_text):
        """Test that sentence-based chunking is consistent"""
        result1 = chunker.chunk_text_by_sentences(sample_text)
        result2 = chunker.chunk_text_by_sentences(sample_text)
        
        assert len(result1) == len(result2)
        for i in range(len(result1)):
            assert result1[i][0] == result2[i][0]  # Same chunk text

    def test_overlap_calculation(self):
        """Test overlap calculation in chunking"""
        chunker = TextChunker(chunk_size=50, overlap_size=10)
        
        # Create text that will result in multiple chunks
        text = "Word " * 200  # Should create multiple chunks
        result = chunker.chunk_text(text)
        
        if len(result) > 1:
            # Check overlap tokens are calculated correctly
            for i in range(1, len(result)):
                _, token_count, overlap_tokens = result[i]
                assert overlap_tokens <= min(chunker.overlap_size, token_count)
                assert overlap_tokens >= 0

    def test_chunker_with_different_parameters(self):
        """Test TextChunker with different chunk sizes and overlaps"""
        test_cases = [
            (100, 10),
            (500, 25),
            (2000, 100),
            (50, 5)
        ]
        
        text = "Testing different parameters. " * 50
        
        for chunk_size, overlap_size in test_cases:
            chunker = TextChunker(chunk_size=chunk_size, overlap_size=overlap_size)
            result = chunker.chunk_text(text)
            
            assert len(result) >= 1
            
            # Verify all chunks have reasonable token counts
            for chunk_text, token_count, overlap_tokens in result:
                assert token_count > 0
                assert len(chunk_text) > 0
                assert overlap_tokens >= 0

class TestTextChunkerEdgeCases:
    """Test edge cases for TextChunker"""

    def test_zero_chunk_size(self):
        """Test behavior with zero chunk size"""
        with pytest.raises(Exception):
            # This should either raise an exception or handle gracefully
            chunker = TextChunker(chunk_size=0, overlap_size=0)
            chunker.chunk_text("test")

    def test_overlap_larger_than_chunk(self):
        """Test behavior when overlap is larger than chunk size"""
        chunker = TextChunker(chunk_size=10, overlap_size=20)
        text = "This is a test to see what happens with large overlap."
        
        result = chunker.chunk_text(text)
        
        # Should still produce valid results
        assert len(result) >= 1
        for chunk_text, token_count, overlap_tokens in result:
            assert token_count > 0
            # Overlap should not exceed token count or chunk size
            assert overlap_tokens <= max(token_count, chunker.chunk_size)

    def test_special_characters(self):
        """Test chunking text with special characters"""
        chunker = TextChunker(chunk_size=100, overlap_size=10)
        text = "Text with émojis 😀 and spécial châractërs! @#$%^&*()"
        
        result = chunker.chunk_text(text)
        
        assert len(result) >= 1
        chunk_text, token_count, overlap_tokens = result[0]
        assert len(chunk_text) > 0
        assert token_count > 0

    def test_unicode_text(self):
        """Test chunking Unicode text"""
        chunker = TextChunker(chunk_size=100, overlap_size=10)
        text = "Unicode text: 你好世界 Здравствуй мир مرحبا بالعالم"
        
        result = chunker.chunk_text(text)
        
        assert len(result) >= 1
        chunk_text, token_count, overlap_tokens = result[0]
        assert len(chunk_text) > 0
        assert token_count > 0

    def test_very_long_sentences(self):
        """Test chunking with very long sentences"""
        chunker = TextChunker(chunk_size=50, overlap_size=5)
        
        # Create one very long sentence
        long_sentence = "This is a very long sentence " * 20 + "."
        
        result = chunker.chunk_text_by_sentences(long_sentence)
        
        assert len(result) >= 1
        # Should handle the long sentence appropriately

if __name__ == "__main__":
    pytest.main([__file__, "-v"])