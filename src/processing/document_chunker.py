"""
Document chunking module that implements semantic chunking.
Adapted from the reference implementation with modifications for
synchronous subprocess execution.
"""
import sys
import os
from pathlib import Path
import uuid
from typing import List, Dict, Any, Union, Optional
import torch
import re
import gc

# Add parent directory to sys.path to enable imports from project root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.core.config import CONFIG
from src.utils.logger import setup_logger
from src.utils.resource_monitor import log_memory_usage

logger = setup_logger(__name__)

class DocumentChunker:
    """
    Document chunker that implements semantic chunking.
    """
    
    def __init__(self, status_queue=None):
        """
        Initialize document chunker.
        
        Args:
            status_queue (Queue, optional): Queue for status updates.
        """
        self.chunk_size = CONFIG["document_processing"]["chunk_size"]
        self.chunk_overlap = CONFIG["document_processing"]["chunk_overlap"]
        self.embedding_model_name = CONFIG["models"]["embedding_model"]
        self.status_queue = status_queue
        self.embedding_model = None
        
        logger.info(f"Initializing DocumentChunker with chunk_size={self.chunk_size}, "
                   f"chunk_overlap={self.chunk_overlap}, embedding_model={self.embedding_model_name}")
        
        # Models will be loaded on demand to optimize memory usage
        
        if self.status_queue:
            self.status_queue.put(('status', 'Document chunker initialized'))
        
        log_memory_usage(logger)
    
    def _update_status(self, status, progress=None):
        """
        Update status via queue if available.
        
        Args:
            status (str): Status message
            progress (float, optional): Progress value between 0 and 1
        """
        if self.status_queue:
            if progress is not None:
                self.status_queue.put(('progress', progress, status))
            else:
                self.status_queue.put(('status', status))
        
        # Always log status
        logger.info(status)
    
    def _load_embedding_model(self):
        """
        Load the embedding model if not already loaded.
        """
        if self.embedding_model is not None:
            return
        
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            
            self._update_status(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
            self._update_status(f"Embedding model loaded successfully")
        except Exception as e:
            error_msg = f"Error loading embedding model: {e}"
            self._update_status(error_msg)
            logger.error(error_msg)
            raise
    
    def _unload_embedding_model(self):
        """
        Unload embedding model to free up memory.
        """
        if self.embedding_model is not None:
            self._update_status("Unloading embedding model")
            
            # Delete the model reference
            del self.embedding_model
            self.embedding_model = None
            
            # Clean up CUDA memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            log_memory_usage(logger)
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a document using semantic chunking.
        
        Args:
            document (dict): Document data from DocumentLoader
            
        Returns:
            list: List of chunk dictionaries
        """
        doc_id = document.get('document_id', 'unknown')
        self._update_status(f"Chunking document: {doc_id}")
        
        chunks = []
        
        # First, load the embedding model for semantic chunking
        self._load_embedding_model()
        
        try:
            # Process each content item (page or section)
            content_items = document.get('content', [])
            for i, content_item in enumerate(content_items):
                progress = ((i + 1) / len(content_items)) * 0.9  # Progress up to 90%
                
                page_num = content_item.get('page_num', None)
                text = content_item.get('text', '')
                
                # Skip empty content
                if not text.strip():
                    continue
                
                # Generate semantic chunks for this content
                self._update_status(f"Chunking content item {i+1}/{len(content_items)}" + 
                                  (f" (page {page_num})" if page_num else ""), progress)
                
                content_chunks = self._semantic_chunking(text)
                
                # Create chunk objects with metadata
                for chunk_idx, chunk_text in enumerate(content_chunks):
                    chunk_id = str(uuid.uuid4())
                    
                    chunk = {
                        'chunk_id': chunk_id,
                        'document_id': document.get('document_id', ''),
                        'file_name': document.get('file_name', ''),
                        'text': chunk_text,
                        'page_num': page_num,
                        'chunk_idx': chunk_idx,
                        'metadata': {
                            'document_metadata': document.get('metadata', {}),
                            'file_type': document.get('file_type', ''),
                            'chunk_method': 'semantic'
                        }
                    }
                    
                    chunks.append(chunk)
            
            # Cleanup
            self._unload_embedding_model()
            
            self._update_status(f"Created {len(chunks)} chunks for document {doc_id}", 1.0)
            log_memory_usage(logger)
            
            return chunks
            
        except Exception as e:
            # Make sure to unload model even if there's an error
            self._unload_embedding_model()
            
            error_msg = f"Error chunking document {doc_id}: {e}"
            self._update_status(error_msg)
            logger.error(error_msg)
            raise
    
    def _semantic_chunking(self, text: str) -> List[str]:
        """
        Perform semantic chunking on text using a hierarchical approach:
        1. First split by recursive text boundaries (paragraphs, sentences)
        2. Then apply semantic chunking for more intelligent breaks
        3. Fall back to token-based splitting for oversized chunks
        
        Args:
            text (str): Text to chunk
            
        Returns:
            list: List of text chunks
        """
        try:
            # Step 1: First split by natural boundaries
            initial_chunks = self._split_by_recursive_boundaries(text)
            logger.info(f"Initial boundary splitting created {len(initial_chunks)} chunks")
            
            # Step 2: Apply semantic chunking to each large chunk
            from langchain_experimental.text_splitter import SemanticChunker
            
            semantic_chunks = []
            for chunk in initial_chunks:
                # Only apply semantic chunking to larger chunks
                if len(chunk) > self.chunk_size:
                    # Configure semantic chunker
                    text_splitter = SemanticChunker(
                        self.embedding_model,
                        breakpoint_threshold_type="percentile",
                        breakpoint_threshold_amount=95.0
                    )
                    
                    # Apply semantic chunking
                    docs = text_splitter.create_documents([chunk])
                    semantic_chunks.extend([doc.page_content for doc in docs])
                else:
                    # Keep small chunks as-is
                    semantic_chunks.append(chunk)
            
            # Step 3: Apply fallback splitting for any chunks still too large
            final_chunks = []
            for chunk in semantic_chunks:
                if len(chunk) > self.chunk_size * 1.5:  # Allow some flexibility
                    # Split oversized chunks using sentence boundaries
                    final_chunks.extend(self._sentence_splitting(chunk))
                else:
                    final_chunks.append(chunk)
            
            logger.info(f"Semantic chunking pipeline created {len(final_chunks)} chunks from text of length {len(text)}")
            return final_chunks
            
        except Exception as e:
            logger.error(f"Error during semantic chunking: {e}")
            logger.warning("Falling back to basic chunking")
            return self._basic_chunking(text)
    
    def _split_by_recursive_boundaries(self, text: str) -> List[str]:
        """
        Split text by natural boundaries like paragraphs and sections.
        
        Args:
            text (str): Text to split
            
        Returns:
            list: List of text chunks split by natural boundaries
        """
        # First try splitting by multiple newlines (paragraphs/sections)
        if '\n\n\n' in text:
            return [chunk.strip() for chunk in text.split('\n\n\n') if chunk.strip()]
        
        # Then try double newlines (paragraphs)
        if '\n\n' in text:
            initial_splits = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
            
            # Check if any splits are still too large
            result = []
            for split in initial_splits:
                if len(split) > self.chunk_size * 2:
                    # Try splitting large paragraphs by headings or bullet points
                    subsplits = self._split_by_headings_or_bullets(split)
                    result.extend(subsplits)
                else:
                    result.append(split)
            return result
        
        # If no paragraph breaks, try splitting by headings or bullet points
        heading_splits = self._split_by_headings_or_bullets(text)
        if len(heading_splits) > 1:
            return heading_splits
        
        # Last resort: return the whole text as one chunk
        return [text]
    
    def _split_by_headings_or_bullets(self, text: str) -> List[str]:
        """
        Split text by headings or bullet points.
        
        Args:
            text (str): Text to split
            
        Returns:
            list: List of text chunks split by headings or bullets
        """
        # Try to identify heading patterns or bullet points
        heading_pattern = re.compile(r'\n[A-Z][^\n]{0,50}:\s*\n|\n\d+\.\s+[A-Z]|\n[•\-\*]\s+')
        
        splits = []
        last_end = 0
        
        for match in heading_pattern.finditer(text):
            # Don't split if match is at the beginning
            if match.start() > last_end:
                splits.append(text[last_end:match.start()])
                last_end = match.start()
        
        # Add the final chunk
        if last_end < len(text):
            splits.append(text[last_end:])
        
        # If we found meaningful splits, return them
        if len(splits) > 1:
            return [chunk.strip() for chunk in splits if chunk.strip()]
        
        # Otherwise return the original text
        return [text]
    
    def _sentence_splitting(self, text: str) -> List[str]:
        """
        Split text into sentences and combine into chunks under the target size.
        
        Args:
            text (str): Text to split
            
        Returns:
            list: List of text chunks
        """
        # Split text into sentences
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_endings, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Combine sentences into chunks under the target size
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Current chunk would exceed size limit, finalize it
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _basic_chunking(self, text: str) -> List[str]:
        """
        Fallback chunking method that splits text based on character count.
        
        Args:
            text (str): Text to chunk
            
        Returns:
            list: List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            # Find a good end point (preferably at paragraph or sentence boundary)
            end = min(start + self.chunk_size, len(text))
            
            # Try to find paragraph break
            if end < len(text):
                paragraph_break = text.rfind('\n\n', start, end)
                if paragraph_break != -1 and paragraph_break > start + (self.chunk_size // 2):
                    end = paragraph_break + 2
            
            # If no paragraph break, try sentence break
            if end < len(text) and end == start + self.chunk_size:
                sentence_break = max(
                    text.rfind('. ', start, end),
                    text.rfind('! ', start, end),
                    text.rfind('? ', start, end)
                )
                if sentence_break != -1 and sentence_break > start + (self.chunk_size // 2):
                    end = sentence_break + 2
            
            # Get the chunk and add to list
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position for next chunk, accounting for overlap
            start = end - self.chunk_overlap if end < len(text) else end
        
        logger.info(f"Basic chunking created {len(chunks)} chunks from text of length {len(text)}")
        return chunks
