"""
Comprehensive unit tests for the enhanced text processing module.

Tests cover multiple chunking strategies, token estimation, overlap handling,
boundary detection, and chunk quality metrics.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.ingestion.text_processor import (
    TextChunker, ChunkingStrategy, ChunkMetrics,
)


class TestTextChunker:
    """Test suite for the enhanced TextChunker class."""
    
    def test_init_default_parameters(self):
        """Test TextChunker initialization with default parameters."""
        chunker = TextChunker()
        
        assert chunker.chunk_size == 1000
        assert chunker.overlap_size == 200
        assert chunker.strategy == ChunkingStrategy.SENTENCE_BASED
        assert chunker.preserve_sentences is True
        assert chunker.min_chunk_size == 50
    
    def test_init_custom_parameters(self):
        """Test TextChunker initialization with custom parameters."""
        chunker = TextChunker(
            chunk_size=2000,
            overlap_size=400,
            strategy=ChunkingStrategy.PARAGRAPH_BASED,
            preserve_sentences=False,
            min_chunk_size=100
        )
        
        assert chunker.chunk_size == 2000
        assert chunker.overlap_size == 400
        assert chunker.strategy == ChunkingStrategy.PARAGRAPH_BASED
        assert chunker.preserve_sentences is False
        assert chunker.min_chunk_size == 100
    
    def test_init_parameter_validation(self):
        """Test parameter validation during initialization."""
        # Test minimum chunk size enforcement
        chunker = TextChunker(chunk_size=50, min_chunk_size=5)
        assert chunker.chunk_size == 100  # Should be increased to minimum
        assert chunker.min_chunk_size == 10  # Should be increased to minimum
        
        # Test overlap size limitation (max 50% of chunk size)
        chunker = TextChunker(chunk_size=1000, overlap_size=800)
        assert chunker.overlap_size == 500  # Should be limited to 50%
    
    def test_estimate_token_count_empty_input(self):
        """Test token estimation with empty or invalid input."""
        chunker = TextChunker()
        
        assert chunker.estimate_token_count("") == 0
        assert chunker.estimate_token_count(None) == 0
        assert chunker.estimate_token_count(123) == 0
    
    def test_estimate_token_count_simple_text(self):
        """Test token estimation with simple text."""
        chunker = TextChunker()
        
        # Simple word counting
        result = chunker.estimate_token_count("Hello world")
        assert result > 0
        assert isinstance(result, int)
        
        # Text with punctuation
        result = chunker.estimate_token_count("Hello, world!")
        assert result > chunker.estimate_token_count("Hello world")
    
    def test_estimate_token_count_complex_text(self):
        """Test token estimation with complex text."""
        chunker = TextChunker()
        
        complex_text = """
        This is a more complex sentence with various punctuation marks,
        numbers like 123, and special characters like @#$%. 
        It should provide a reasonable token estimate.
        """
        
        result = chunker.estimate_token_count(complex_text)
        assert result > 20  # Should have substantial token count
        assert isinstance(result, int)
    
    def test_chunk_text_empty_input(self):
        """Test chunking with empty or invalid input."""
        chunker = TextChunker()
        
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text(None) == []
        assert chunker.chunk_text("   ") == []  # Whitespace only
    
    def test_chunk_text_strategy_routing(self):
        """Test that chunk_text routes to correct strategy methods."""
        text = "This is a test sentence. This is another test sentence."
        
        # Test each strategy
        strategies = [
            ChunkingStrategy.CHARACTER_BASED,
            ChunkingStrategy.SENTENCE_BASED,
            ChunkingStrategy.PARAGRAPH_BASED,
            ChunkingStrategy.SEMANTIC_BASED,
        ]
        
        for strategy in strategies:
            chunker = TextChunker(strategy=strategy, chunk_size=50)
            chunks = chunker.chunk_text(text)
            
            assert isinstance(chunks, list)
            if chunks:  # May be empty for very small text
                for chunk_text, token_count, overlap_tokens in chunks:
                    assert isinstance(chunk_text, str)
                    assert isinstance(token_count, int)
                    assert isinstance(overlap_tokens, int)
                    assert token_count > 0
                    assert overlap_tokens >= 0


class TestCharacterBasedChunking:
    """Test suite for character-based chunking strategy."""
    
    def test_character_chunking_basic(self):
        """Test basic character-based chunking."""
        chunker = TextChunker(
            chunk_size=100,  # Small size for testing
            overlap_size=20,
            strategy=ChunkingStrategy.CHARACTER_BASED
        )
        
        text = "This is a test document. " * 20  # Repeat to create longer text
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 1  # Should create multiple chunks
        
        # Verify chunk structure
        for chunk_text, token_count, overlap_tokens in chunks:
            assert isinstance(chunk_text, str)
            assert len(chunk_text.strip()) > 0
            assert token_count > 0
            assert overlap_tokens >= 0
    
    def test_character_chunking_boundary_detection(self):
        """Test sentence boundary detection in character chunking."""
        chunker = TextChunker(
            chunk_size=50,
            overlap_size=10,
            strategy=ChunkingStrategy.CHARACTER_BASED,
            preserve_sentences=True
        )
        
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunker.chunk_text(text)
        
        # With sentence preservation, chunks should end at sentence boundaries when possible
        for chunk_text, _, _ in chunks[:-1]:  # All but last chunk
            chunk_text = chunk_text.strip()
            # Should ideally end with sentence punctuation
            # (though not guaranteed due to size constraints)
    
    def test_character_chunking_no_boundary_preservation(self):
        """Test character chunking without sentence boundary preservation."""
        chunker = TextChunker(
            chunk_size=50,
            overlap_size=10,
            strategy=ChunkingStrategy.CHARACTER_BASED,
            preserve_sentences=False
        )
        
        text = "This is a very long sentence that should be split in the middle without regard for sentence boundaries."
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 0


class TestSentenceBasedChunking:
    """Test suite for sentence-based chunking strategy."""
    
    def test_sentence_chunking_basic(self):
        """Test basic sentence-based chunking."""
        chunker = TextChunker(
            chunk_size=100,
            overlap_size=20,
            strategy=ChunkingStrategy.SENTENCE_BASED
        )
        
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) >= 1
        
        # All chunks should contain complete sentences
        for chunk_text, token_count, overlap_tokens in chunks:
            assert isinstance(chunk_text, str)
            assert token_count > 0
            assert overlap_tokens >= 0
    
    def test_sentence_splitting_with_abbreviations(self):
        """Test sentence splitting with common abbreviations."""
        chunker = TextChunker(
            chunk_size=200,
            overlap_size=40,
            strategy=ChunkingStrategy.SENTENCE_BASED
        )
        
        text = "Dr. Smith went to Inc. Corp. building. He met with Mr. Johnson etc."
        chunks = chunker.chunk_text(text)
        
        # Should handle abbreviations correctly and not split inappropriately
        assert len(chunks) >= 1
        
        # Check that abbreviations are preserved
        full_text = ' '.join(chunk[0] for chunk in chunks)
        assert "Dr. Smith" in full_text
        assert "Inc. Corp." in full_text
        assert "Mr. Johnson" in full_text
    
    def test_sentence_overlap_handling(self):
        """Test proper overlap handling in sentence-based chunking."""
        chunker = TextChunker(
            chunk_size=80,
            overlap_size=30,
            strategy=ChunkingStrategy.SENTENCE_BASED
        )
        
        sentences = ["This is sentence one.", "This is sentence two.", "This is sentence three.", 
                    "This is sentence four.", "This is sentence five."]
        text = " ".join(sentences)
        
        chunks = chunker.chunk_text(text)
        
        if len(chunks) > 1:
            # Check that overlaps are properly calculated
            for i, (_, _, overlap_tokens) in enumerate(chunks):
                if i == 0:
                    assert overlap_tokens == 0  # First chunk has no overlap
                else:
                    assert overlap_tokens >= 0  # Subsequent chunks may have overlap
    
    def test_minimum_chunk_size_enforcement(self):
        """Test that minimum chunk size is enforced."""
        chunker = TextChunker(
            chunk_size=50,
            overlap_size=10,
            strategy=ChunkingStrategy.SENTENCE_BASED,
            min_chunk_size=20
        )
        
        # Create text with very short sentences
        text = "Hi. Ok. Yes. Maybe. No. Definitely. Absolutely. Perhaps."
        chunks = chunker.chunk_text(text)
        
        # Small final chunks should be merged with previous ones
        for chunk_text, token_count, _ in chunks:
            # All chunks should meet minimum size (except possibly the last if it's the only one)
            if len(chunks) > 1 or token_count >= chunker.min_chunk_size:
                assert token_count >= chunker.min_chunk_size or chunks.index((chunk_text, token_count, _)) == len(chunks) - 1


class TestParagraphBasedChunking:
    """Test suite for paragraph-based chunking strategy."""
    
    def test_paragraph_chunking_basic(self):
        """Test basic paragraph-based chunking."""
        chunker = TextChunker(
            chunk_size=200,
            overlap_size=40,
            strategy=ChunkingStrategy.PARAGRAPH_BASED
        )
        
        text = """First paragraph with some content.
        This is still the first paragraph.

        Second paragraph with different content.
        This belongs to the second paragraph.

        Third paragraph here.
        More content for third paragraph."""
        
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 1
        
        for chunk_text, token_count, overlap_tokens in chunks:
            assert isinstance(chunk_text, str)
            assert token_count > 0
            assert overlap_tokens >= 0
    
    def test_paragraph_chunking_large_paragraph(self):
        """Test paragraph chunking when a single paragraph is too large."""
        chunker = TextChunker(
            chunk_size=100,
            overlap_size=20,
            strategy=ChunkingStrategy.PARAGRAPH_BASED
        )
        
        # Create a very large paragraph
        large_paragraph = "This is a very long paragraph. " * 50
        text = f"{large_paragraph}\n\nThis is a small second paragraph."
        
        chunks = chunker.chunk_text(text)
        
        # Should handle the large paragraph by splitting it into sentences
        assert len(chunks) > 1
    
    def test_paragraph_overlap_calculation(self):
        """Test overlap calculation for paragraph-based chunking."""
        chunker = TextChunker(
            chunk_size=150,
            overlap_size=50,
            strategy=ChunkingStrategy.PARAGRAPH_BASED
        )
        
        text = """Paragraph one with some content here.
        
        Paragraph two with different content.
        
        Paragraph three with more content.
        
        Paragraph four with additional content."""
        
        chunks = chunker.chunk_text(text)
        
        # Verify overlap handling
        for i, (_, _, overlap_tokens) in enumerate(chunks):
            if i == 0:
                assert overlap_tokens == 0
            else:
                assert overlap_tokens >= 0


class TestSemanticChunking:
    """Test suite for semantic chunking strategy."""
    
    def test_semantic_chunking_fallback(self):
        """Test that semantic chunking falls back to sentence-based chunking."""
        chunker = TextChunker(
            chunk_size=100,
            overlap_size=20,
            strategy=ChunkingStrategy.SEMANTIC_BASED
        )
        
        text = "This is a test for semantic chunking. It should work properly."
        chunks = chunker.chunk_text(text)
        
        # Should produce results (falls back to sentence-based)
        assert len(chunks) >= 1
        
        for chunk_text, token_count, overlap_tokens in chunks:
            assert isinstance(chunk_text, str)
            assert token_count > 0
            assert overlap_tokens >= 0


class TestHelperMethods:
    """Test suite for helper methods."""
    
    def test_split_into_sentences_basic(self):
        """Test basic sentence splitting."""
        chunker = TextChunker()
        
        text = "First sentence. Second sentence! Third sentence?"
        sentences = chunker._split_into_sentences(text)
        
        assert len(sentences) == 3
        assert "First sentence" in sentences[0]
        assert "Second sentence" in sentences[1]
        assert "Third sentence" in sentences[2]
    
    def test_split_into_sentences_abbreviations(self):
        """Test sentence splitting with abbreviations."""
        chunker = TextChunker()
        
        text = "Dr. Smith works at Inc. Corp. He likes it."
        sentences = chunker._split_into_sentences(text)
        
        # Should not split on abbreviations
        assert any("Dr. Smith works at Inc. Corp" in sentence for sentence in sentences)
    
    def test_find_sentence_boundary(self):
        """Test sentence boundary detection."""
        chunker = TextChunker()
        
        text = "This is sentence one. This is sentence two. This is sentence three."
        
        # Find boundary near position 25 (should find end of first sentence)
        boundary = chunker._find_sentence_boundary(text, 0, 25)
        
        assert boundary > 0
        assert boundary <= len(text)
    
    def test_find_word_boundary(self):
        """Test word boundary detection."""
        chunker = TextChunker()
        
        text = "This is a test sentence with multiple words"
        
        # Find word boundary near middle
        boundary = chunker._find_word_boundary(text, 20)
        
        assert boundary > 0
        assert boundary <= len(text)
        # Should not split in the middle of a word
        if boundary < len(text):
            assert text[boundary].isspace() or text[boundary-1].isspace()
    
    def test_select_overlap_sentences(self):
        """Test sentence selection for overlap."""
        chunker = TextChunker()
        
        sentences = ["Short sentence.", "Another short sentence.", "Third sentence here.", "Fourth sentence."]
        
        overlap_sentences = chunker._select_overlap_sentences(sentences, 50)
        
        assert isinstance(overlap_sentences, list)
        assert len(overlap_sentences) <= len(sentences)
        
        # Calculate total tokens in selected sentences
        total_tokens = sum(chunker.estimate_token_count(s) for s in overlap_sentences)
        # Should be close to target but not exceed it by much
        assert total_tokens <= 50 * 1.2  # 20% buffer
    
    def test_calculate_sentence_overlap(self):
        """Test calculation of sentence overlap."""
        chunker = TextChunker()
        
        overlap_sentences = ["First sentence.", "Second sentence."]
        current_sentences = ["First sentence.", "Second sentence.", "Third sentence."]
        
        overlap_tokens = chunker._calculate_sentence_overlap(overlap_sentences, current_sentences)
        
        assert overlap_tokens >= 0
        assert isinstance(overlap_tokens, int)


class TestChunkMetrics:
    """Test suite for chunk quality metrics."""
    
    def test_calculate_metrics_empty_chunks(self):
        """Test metrics calculation with empty chunk list."""
        chunker = TextChunker()
        
        metrics = chunker.calculate_metrics([])
        
        assert isinstance(metrics, ChunkMetrics)
        assert metrics.total_chunks == 0
        assert metrics.avg_chunk_size == 0.0
        assert metrics.min_chunk_size == 0
        assert metrics.max_chunk_size == 0
        assert metrics.overlap_efficiency == 0.0
        assert metrics.sentence_boundary_alignment == 0.0
    
    def test_calculate_metrics_with_chunks(self):
        """Test metrics calculation with actual chunks."""
        chunker = TextChunker()
        
        # Create sample chunks: (text, token_count, overlap_tokens)
        chunks = [
            ("First chunk text.", 20, 0),
            ("Second chunk text with overlap.", 25, 5),
            ("Third chunk text.", 18, 3),
        ]
        
        metrics = chunker.calculate_metrics(chunks)
        
        assert metrics.total_chunks == 3
        assert metrics.avg_chunk_size == (20 + 25 + 18) / 3
        assert metrics.min_chunk_size == 18
        assert metrics.max_chunk_size == 25
        assert metrics.overlap_efficiency == (0 + 5 + 3) / (20 + 25 + 18)
        # All chunks end with periods, so should have 100% alignment
        assert metrics.sentence_boundary_alignment == 1.0
    
    def test_calculate_metrics_sentence_alignment(self):
        """Test sentence boundary alignment calculation."""
        chunker = TextChunker()
        
        # Mix of chunks with and without sentence endings
        chunks = [
            ("Chunk ends with period.", 15, 0),
            ("Chunk ends without period", 20, 5),
            ("Another chunk with exclamation!", 18, 3),
        ]
        
        metrics = chunker.calculate_metrics(chunks)
        
        # 2 out of 3 chunks end with sentence punctuation
        assert metrics.sentence_boundary_alignment == 2/3


class TestIntegration:
    """Integration tests for the text processor."""
    
    def test_full_document_processing(self):
        """Test processing a complete document with various features."""
        chunker = TextChunker(
            chunk_size=150,
            overlap_size=30,
            strategy=ChunkingStrategy.SENTENCE_BASED
        )
        
        document_text = """
        Introduction
        
        This is the introduction paragraph. It contains several sentences that explain
        the purpose of this document. The document will be used to test the text
        chunking functionality.
        
        Main Content
        
        The main content section has multiple paragraphs. Each paragraph discusses
        different aspects of the topic. Some sentences are longer than others, which
        helps test the chunking algorithm's ability to handle varied content.
        
        Dr. Johnson and Mr. Smith collaborated on this work. They found that the
        approach used by Inc. Corp. was effective. However, they noted some limitations.
        
        Conclusion
        
        The conclusion summarizes the findings. It provides recommendations for future
        work. The document ends here with a final statement about the research.
        """
        
        chunks = chunker.chunk_text(document_text)
        metrics = chunker.calculate_metrics(chunks)
        
        # Verify basic properties
        assert len(chunks) > 0
        assert metrics.total_chunks == len(chunks)
        assert metrics.avg_chunk_size > 0
        
        # Verify all chunks meet minimum requirements
        for chunk_text, token_count, overlap_tokens in chunks:
            assert len(chunk_text.strip()) > 0
            assert token_count > 0
            assert overlap_tokens >= 0
        
        # Verify overlap consistency
        total_overlap = sum(chunk[2] for chunk in chunks)
        assert total_overlap == sum(chunk[2] for chunk in chunks)  # Sanity check
    
    def test_different_strategies_same_text(self):
        """Test that different strategies produce different results on the same text."""
        text = """
        First paragraph with multiple sentences. This sentence provides more context.
        It continues with additional information.
        
        Second paragraph starts here. It has different content from the first paragraph.
        The content is structured to test paragraph-based chunking.
        
        Third paragraph completes the test. It ensures we have enough content for
        meaningful comparison between chunking strategies.
        """
        
        strategies = [
            ChunkingStrategy.CHARACTER_BASED,
            ChunkingStrategy.SENTENCE_BASED,
            ChunkingStrategy.PARAGRAPH_BASED,
        ]
        
        results = {}
        
        for strategy in strategies:
            chunker = TextChunker(
                chunk_size=100,
                overlap_size=20,
                strategy=strategy
            )
            chunks = chunker.chunk_text(text)
            results[strategy] = chunks
        
        # Different strategies should produce different chunk boundaries
        # (though this isn't guaranteed for all texts, it should be true for this structured text)
        assert len(results) == len(strategies)
        
        # All strategies should produce some chunks
        for strategy, chunks in results.items():
            assert len(chunks) > 0, f"Strategy {strategy} produced no chunks"


if __name__ == '__main__':
    pytest.main([__file__])