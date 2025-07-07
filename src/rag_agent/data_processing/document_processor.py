"""
Document processing module for the RAG agent.
Handles parsing and chunking of various document formats.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
import re


class DocumentProcessor:
    """Process and chunk documents for RAG pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize document processor.
        
        Args:
            config: Configuration dictionary for document processing
        """
        self.config = config
        self.chunk_size = config.get('chunk_size', 1000)
        self.chunk_overlap = config.get('chunk_overlap', 200)
        self.supported_formats = config.get('supported_formats', ['.txt', '.md'])
        self.logger = logging.getLogger(__name__)
    
    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a document and return chunks.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of document chunks with metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        if file_path.suffix not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Extract text based on file type
        text = self._extract_text(file_path)
        
        # Split into chunks
        chunks = self._split_text(text)
        
        # Add metadata
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunks.append({
                'text': chunk,
                'source': str(file_path),
                'chunk_id': i,
                'metadata': {
                    'file_name': file_path.name,
                    'file_type': file_path.suffix,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
            })
        
        self.logger.info(f"Processed {file_path.name} into {len(chunks)} chunks")
        return processed_chunks
    
    def _extract_text(self, file_path: Path) -> str:
        """Extract text from file based on its format."""
        if file_path.suffix in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        # TODO: Add support for PDF, DOCX, etc.
        # This is a placeholder for more complex document parsing
        raise NotImplementedError(f"Text extraction for {file_path.suffix} not implemented yet")
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Find a good break point (sentence or paragraph end)
            if end < len(text):
                # Look for sentence endings
                for break_char in ['. ', '.\n', '!\n', '?\n']:
                    break_pos = text.rfind(break_char, start, end)
                    if break_pos != -1:
                        end = break_pos + len(break_char)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + 1, end - self.chunk_overlap)
        
        return chunks
