"""
Enhanced text chunking utilities for the AgenticRAG system with multiple strategies
"""

import re
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class ChunkingStrategy(Enum):
    """Available text chunking strategies"""
    CHARACTER_BASED = "character"
    SENTENCE_BASED = "sentence"
    PARAGRAPH_BASED = "paragraph"
    SEMANTIC_BASED = "semantic"


@dataclass
class ChunkMetrics:
    """Metrics for chunk quality assessment"""
    total_chunks: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    overlap_efficiency: float
    sentence_boundary_alignment: float


class TextChunker:
    """Advanced text chunking with multiple strategies and quality metrics"""
    
    def __init__(self, chunk_size: int = 1000, overlap_size: int = 200, 
                 strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE_BASED,
                 preserve_sentences: bool = True, min_chunk_size: int = 50):
        """
        Initialize TextChunker with configurable parameters
        
        Args:
            chunk_size: Target size for each chunk (in tokens)
            overlap_size: Overlap between chunks (in tokens)
            strategy: Chunking strategy to use
            preserve_sentences: Whether to avoid breaking sentences
            min_chunk_size: Minimum acceptable chunk size
        """
        self.chunk_size = max(chunk_size, 100)  # Minimum reasonable chunk size
        self.overlap_size = min(overlap_size, chunk_size // 2)  # Max 50% overlap
        self.strategy = strategy
        self.preserve_sentences = preserve_sentences
        self.min_chunk_size = max(min_chunk_size, 10)
        self.logger = logging.getLogger("text_chunker")
        
        # Compile regex patterns for efficiency
        self.sentence_pattern = re.compile(r'[.!?]+\s+')
        self.paragraph_pattern = re.compile(r'\n\s*\n')
        self.word_pattern = re.compile(r'\b\w+\b')
        
        self.logger.info(f"TextChunker initialized: chunk_size={chunk_size}, overlap={overlap_size}, strategy={strategy.value}")
    
    def estimate_token_count(self, text: str) -> int:
        """
        Improved token count estimation using word-based approach
        
        Args:
            text: Input text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        if not text or not isinstance(text, str):
            return 0
            
        # More accurate estimation using word count with adjustments
        words = len(self.word_pattern.findall(text))
        
        # Adjust for punctuation and special characters
        punctuation_count = len(re.findall(r'[^\w\s]', text))
        
        # Estimate tokens as words + (punctuation / 2) with 1.3 multiplier
        # This accounts for subword tokenization in modern models
        estimated_tokens = int((words + punctuation_count / 2) * 1.3)
        
        return max(estimated_tokens, 1)  # Ensure at least 1 token
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[str, int, int]]:
        """
        Chunk text using the configured strategy
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata for context-aware chunking
            
        Returns:
            List of (chunk_text, token_count, overlap_tokens)
        """
        if not text or not isinstance(text, str):
            return []
            
        text = text.strip()
        if not text:
            return []
        
        # Route to appropriate chunking strategy
        if self.strategy == ChunkingStrategy.SENTENCE_BASED:
            return self.chunk_text_by_sentences(text)
        elif self.strategy == ChunkingStrategy.PARAGRAPH_BASED:
            return self.chunk_text_by_paragraphs(text)
        elif self.strategy == ChunkingStrategy.SEMANTIC_BASED:
            return self.chunk_text_semantically(text)
        else:  # CHARACTER_BASED
            return self.chunk_text_by_characters(text)
    
    def chunk_text_by_characters(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Character-based chunking with smart boundary detection
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of (chunk_text, token_count, overlap_tokens)
        """
        chunks = []
        text_length = len(text)
        
        if text_length == 0:
            return chunks
        
        # Convert token-based sizes to character estimates
        avg_chars_per_token = 4.0  # Baseline estimate
        char_chunk_size = int(self.chunk_size * avg_chars_per_token)
        char_overlap_size = int(self.overlap_size * avg_chars_per_token)
        
        start_pos = 0
        chunk_number = 0
        
        while start_pos < text_length:
            # Calculate target end position
            target_end = min(start_pos + char_chunk_size, text_length)
            
            # Find optimal boundary if preserving sentences
            if self.preserve_sentences and target_end < text_length:
                end_pos = self._find_sentence_boundary(text, start_pos, target_end)
            else:
                end_pos = target_end
            
            # Extract chunk
            chunk_text = text[start_pos:end_pos].strip()
            
            if not chunk_text:
                break
            
            token_count = self.estimate_token_count(chunk_text)
            
            # Skip chunks that are too small unless it's the last chunk
            if token_count < self.min_chunk_size and end_pos < text_length:
                start_pos = end_pos
                continue
            
            # Calculate overlap for this chunk
            overlap_tokens = 0
            if chunk_number > 0:
                overlap_tokens = min(self.overlap_size, token_count)
            
            chunks.append((chunk_text, token_count, overlap_tokens))
            
            # Move to next chunk position
            if end_pos >= text_length:
                break
                
            # Calculate next start position with overlap
            next_start = end_pos - char_overlap_size
            if next_start <= start_pos:  # Prevent infinite loop
                next_start = start_pos + max(1, char_chunk_size - char_overlap_size)
            
            start_pos = max(next_start, start_pos + 1)
            chunk_number += 1
        
        self.logger.debug(f"Character-based chunking: {len(chunks)} chunks from {len(text)} characters")
        return chunks
    
    def chunk_text_by_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Advanced sentence-based chunking with proper overlap handling
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of (chunk_text, token_count, overlap_tokens)
        """
        # More sophisticated sentence splitting
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        chunks = []
        current_sentences = []
        current_token_count = 0
        overlap_sentences = []
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = self.estimate_token_count(sentence)
            
            # Check if adding this sentence exceeds chunk size
            if (current_token_count + sentence_tokens > self.chunk_size and 
                current_sentences and 
                current_token_count >= self.min_chunk_size):
                
                # Finalize current chunk
                chunk_text = ' '.join(current_sentences)
                
                # Calculate actual overlap tokens from previous chunk
                overlap_tokens = self._calculate_sentence_overlap(overlap_sentences, current_sentences)
                
                chunks.append((chunk_text, current_token_count, overlap_tokens))
                
                # Prepare overlap for next chunk
                overlap_sentences = self._select_overlap_sentences(
                    current_sentences, self.overlap_size
                )
                
                # Start new chunk with overlap
                current_sentences = overlap_sentences + [sentence]
                current_token_count = (
                    sum(self.estimate_token_count(s) for s in overlap_sentences) + 
                    sentence_tokens
                )
            else:
                current_sentences.append(sentence)
                current_token_count += sentence_tokens
        
        # Add final chunk if it exists
        if current_sentences:
            chunk_text = ' '.join(current_sentences)
            overlap_tokens = self._calculate_sentence_overlap(overlap_sentences, current_sentences)
            
            # Only add if it meets minimum size or is the only chunk
            if current_token_count >= self.min_chunk_size or not chunks:
                chunks.append((chunk_text, current_token_count, overlap_tokens))
            elif chunks:  # Merge small final chunk with previous chunk
                last_chunk_text, last_token_count, last_overlap = chunks[-1]
                merged_text = last_chunk_text + ' ' + chunk_text
                merged_tokens = last_token_count + current_token_count
                chunks[-1] = (merged_text, merged_tokens, last_overlap)
        
        self.logger.debug(f"Sentence-based chunking: {len(chunks)} chunks from {len(sentences)} sentences")
        return chunks
    
    def chunk_text_by_paragraphs(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Paragraph-based chunking for document structure preservation
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of (chunk_text, token_count, overlap_tokens)
        """
        paragraphs = self.paragraph_pattern.split(text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not paragraphs:
            return []
        
        chunks = []
        current_paragraphs = []
        current_token_count = 0
        overlap_paragraphs = []
        
        for paragraph in paragraphs:
            para_tokens = self.estimate_token_count(paragraph)
            
            # If single paragraph is too large, split it by sentences
            if para_tokens > self.chunk_size:
                if current_paragraphs:
                    # Finalize current chunk first
                    chunk_text = '\n\n'.join(current_paragraphs)
                    overlap_tokens = self._calculate_paragraph_overlap(overlap_paragraphs, current_paragraphs)
                    chunks.append((chunk_text, current_token_count, overlap_tokens))
                    current_paragraphs = []
                    current_token_count = 0
                
                # Split large paragraph by sentences
                para_chunks = self.chunk_text_by_sentences(paragraph)
                chunks.extend(para_chunks)
                overlap_paragraphs = []
                continue
            
            # Check if adding this paragraph exceeds chunk size
            if (current_token_count + para_tokens > self.chunk_size and 
                current_paragraphs and 
                current_token_count >= self.min_chunk_size):
                
                # Finalize current chunk
                chunk_text = '\n\n'.join(current_paragraphs)
                overlap_tokens = self._calculate_paragraph_overlap(overlap_paragraphs, current_paragraphs)
                chunks.append((chunk_text, current_token_count, overlap_tokens))
                
                # Prepare overlap for next chunk
                overlap_paragraphs = self._select_overlap_paragraphs(
                    current_paragraphs, self.overlap_size
                )
                
                # Start new chunk
                current_paragraphs = overlap_paragraphs + [paragraph]
                current_token_count = (
                    sum(self.estimate_token_count(p) for p in overlap_paragraphs) + 
                    para_tokens
                )
            else:
                current_paragraphs.append(paragraph)
                current_token_count += para_tokens
        
        # Add final chunk
        if current_paragraphs:
            chunk_text = '\n\n'.join(current_paragraphs)
            overlap_tokens = self._calculate_paragraph_overlap(overlap_paragraphs, current_paragraphs)
            
            if current_token_count >= self.min_chunk_size or not chunks:
                chunks.append((chunk_text, current_token_count, overlap_tokens))
            elif chunks:
                # Merge with previous chunk
                last_chunk_text, last_token_count, last_overlap = chunks[-1]
                merged_text = last_chunk_text + '\n\n' + chunk_text
                merged_tokens = last_token_count + current_token_count
                chunks[-1] = (merged_text, merged_tokens, last_overlap)
        
        self.logger.debug(f"Paragraph-based chunking: {len(chunks)} chunks from {len(paragraphs)} paragraphs")
        return chunks
    
    def chunk_text_semantically(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Semantic chunking based on topic coherence (simplified implementation)
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of (chunk_text, token_count, overlap_tokens)
        """
        # For now, fall back to sentence-based chunking with enhanced boundary detection
        # In a full implementation, this would use semantic similarity metrics
        self.logger.debug("Using sentence-based chunking as semantic fallback")
        return self.chunk_text_by_sentences(text)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Advanced sentence splitting with abbreviation handling
        
        Args:
            text: Input text to split
            
        Returns:
            List of sentences
        """
        # Handle common abbreviations that shouldn't trigger sentence breaks
        abbreviations = {'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Inc.', 'Ltd.', 'Co.', 
                        'Corp.', 'etc.', 'vs.', 'i.e.', 'e.g.'}
        
        # Replace abbreviations temporarily
        temp_text = text
        replacements = {}
        for i, abbrev in enumerate(abbreviations):
            placeholder = f"__ABBREV_{i}__"
            temp_text = temp_text.replace(abbrev, placeholder)
            replacements[placeholder] = abbrev
        
        # Split on sentence endings
        sentences = self.sentence_pattern.split(temp_text)
        
        # Restore abbreviations and clean up
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Restore abbreviations
            for placeholder, abbrev in replacements.items():
                sentence = sentence.replace(placeholder, abbrev)
            
            cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _find_sentence_boundary(self, text: str, start: int, target_end: int) -> int:
        """
        Find optimal sentence boundary near target position
        
        Args:
            text: Full text
            start: Start position
            target_end: Target end position
            
        Returns:
            Optimal boundary position
        """
        # Look for sentence endings within reasonable distance
        search_start = max(start, target_end - 200)
        search_end = min(len(text), target_end + 200)
        
        search_text = text[search_start:search_end]
        
        # Find sentence endings
        endings = []
        for match in re.finditer(r'[.!?]\s+', search_text):
            absolute_pos = search_start + match.end()
            if start < absolute_pos <= search_end:
                distance = abs(absolute_pos - target_end)
                endings.append((absolute_pos, distance))
        
        if endings:
            # Return closest sentence ending to target
            best_pos, _ = min(endings, key=lambda x: x[1])
            return best_pos
        
        # Fall back to word boundary
        return self._find_word_boundary(text, target_end)
    
    def _find_word_boundary(self, text: str, pos: int) -> int:
        """
        Find word boundary near position
        
        Args:
            text: Full text
            pos: Target position
            
        Returns:
            Word boundary position
        """
        if pos >= len(text):
            return len(text)
        
        # Look backwards for whitespace
        for i in range(pos, max(0, pos - 50), -1):
            if text[i].isspace():
                return i + 1
        
        return pos
    
    def _select_overlap_sentences(self, sentences: List[str], target_overlap_tokens: int) -> List[str]:
        """
        Select sentences for overlap based on token target
        
        Args:
            sentences: List of sentences to select from
            target_overlap_tokens: Target number of overlap tokens
            
        Returns:
            Selected sentences for overlap
        """
        if not sentences or target_overlap_tokens <= 0:
            return []
        
        overlap_sentences = []
        overlap_tokens = 0
        
        # Take sentences from the end
        for sentence in reversed(sentences):
            sentence_tokens = self.estimate_token_count(sentence)
            if overlap_tokens + sentence_tokens <= target_overlap_tokens * 1.2:  # 20% buffer
                overlap_sentences.insert(0, sentence)
                overlap_tokens += sentence_tokens
            else:
                break
        
        return overlap_sentences
    
    def _select_overlap_paragraphs(self, paragraphs: List[str], target_overlap_tokens: int) -> List[str]:
        """
        Select paragraphs for overlap based on token target
        
        Args:
            paragraphs: List of paragraphs to select from
            target_overlap_tokens: Target number of overlap tokens
            
        Returns:
            Selected paragraphs for overlap
        """
        if not paragraphs or target_overlap_tokens <= 0:
            return []
        
        overlap_paragraphs = []
        overlap_tokens = 0
        
        # Take paragraphs from the end
        for paragraph in reversed(paragraphs):
            para_tokens = self.estimate_token_count(paragraph)
            if overlap_tokens + para_tokens <= target_overlap_tokens * 1.2:  # 20% buffer
                overlap_paragraphs.insert(0, paragraph)
                overlap_tokens += para_tokens
            else:
                break
        
        return overlap_paragraphs
    
    def _calculate_sentence_overlap(self, overlap_sentences: List[str], current_sentences: List[str]) -> int:
        """
        Calculate actual overlap tokens between sentence lists
        
        Args:
            overlap_sentences: Previous chunk's overlap sentences
            current_sentences: Current chunk's sentences
            
        Returns:
            Number of overlapping tokens
        """
        if not overlap_sentences:
            return 0
        
        # Find common sentences at the beginning of current chunk
        overlap_count = 0
        for i, overlap_sentence in enumerate(overlap_sentences):
            if i < len(current_sentences) and current_sentences[i] == overlap_sentence:
                overlap_count += self.estimate_token_count(overlap_sentence)
            else:
                break
        
        return overlap_count
    
    def _calculate_paragraph_overlap(self, overlap_paragraphs: List[str], current_paragraphs: List[str]) -> int:
        """
        Calculate actual overlap tokens between paragraph lists
        
        Args:
            overlap_paragraphs: Previous chunk's overlap paragraphs
            current_paragraphs: Current chunk's paragraphs
            
        Returns:
            Number of overlapping tokens
        """
        if not overlap_paragraphs:
            return 0
        
        overlap_count = 0
        for i, overlap_paragraph in enumerate(overlap_paragraphs):
            if i < len(current_paragraphs) and current_paragraphs[i] == overlap_paragraph:
                overlap_count += self.estimate_token_count(overlap_paragraph)
            else:
                break
        
        return overlap_count
    
    def calculate_metrics(self, chunks: List[Tuple[str, int, int]]) -> ChunkMetrics:
        """
        Calculate quality metrics for chunks
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Chunk quality metrics
        """
        if not chunks:
            return ChunkMetrics(
                total_chunks=0,
                avg_chunk_size=0.0,
                min_chunk_size=0,
                max_chunk_size=0,
                overlap_efficiency=0.0,
                sentence_boundary_alignment=0.0
            )
        
        chunk_sizes = [chunk[1] for chunk in chunks]  # token counts
        overlap_tokens = [chunk[2] for chunk in chunks]  # overlap counts
        
        # Calculate sentence boundary alignment
        sentence_aligned = 0
        for chunk_text, _, _ in chunks:
            # Check if chunk ends with sentence boundary
            if re.search(r'[.!?]\s*$', chunk_text.strip()):
                sentence_aligned += 1
        
        total_overlap = sum(overlap_tokens)
        total_tokens = sum(chunk_sizes)
        
        return ChunkMetrics(
            total_chunks=len(chunks),
            avg_chunk_size=sum(chunk_sizes) / len(chunks),
            min_chunk_size=min(chunk_sizes),
            max_chunk_size=max(chunk_sizes),
            overlap_efficiency=total_overlap / max(total_tokens, 1),
            sentence_boundary_alignment=sentence_aligned / len(chunks)
        )