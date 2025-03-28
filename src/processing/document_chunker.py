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
import time

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
    
    def __init__(self, status_queue=None, load_model_immediately=False, embedding_model=None):
        """
        Initialize document chunker.
        
        Args:
            status_queue (Queue, optional): Queue for status updates.
            load_model_immediately (bool): Whether to load the embedding model immediately.
            embedding_model: A pre-loaded embedding model to use instead of loading a new one.
        """
        self.chunk_size = CONFIG["document_processing"]["chunk_size"]
        self.chunk_overlap = CONFIG["document_processing"]["chunk_overlap"]
        # Use semantic_chunking_model if specified, otherwise fall back to embedding_model
        self.embedding_model_name = CONFIG["models"].get("semantic_chunking_model", 
                                           CONFIG["models"]["embedding_model"])
        self.status_queue = status_queue
        
        # Use the provided model if available
        self.embedding_model = embedding_model
        
        logger.info(f"Initializing DocumentChunker with chunk_size={self.chunk_size}, "
                   f"chunk_overlap={self.chunk_overlap}, embedding_model={self.embedding_model_name}")
        
        print(f"[CHUNKER] Initializing with {'pre-loaded' if embedding_model else 'no'} embedding model")
        
        if self.status_queue:
            self.status_queue.put(('status', 'Document chunker initialized'))
        
        log_memory_usage(logger)
        
        # Load the model immediately if requested and not already provided
        if load_model_immediately and not embedding_model:
            self.load_model()
    
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
    
    def load_model(self):
        """
        Load the embedding model if not already loaded.
        """
        if self.embedding_model is not None:
            logger.info("Embedding model already loaded, skipping load")
            print(f"[CHUNKER] Embedding model already loaded, skipping load")
            return
        
        print(f"[CHUNKER] ===== STARTING MODEL LOAD =====")
        logger.info("===== STARTING MODEL LOAD =====")
        
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            import time
            import os
            import torch
            from transformers import logging as transformers_logging
            
            # Set transformers logging to more verbose level
            transformers_logging.set_verbosity_info()
            
            # Ensure the TRANSFORMERS_CACHE env var is set and logged
            hf_cache = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
            transformers_cache = os.environ.get('TRANSFORMERS_CACHE', os.path.join(hf_cache, 'transformers'))
            
            # Print caching info directly to console for visibility
            print(f"\n[LOADING] Loading semantic chunking embedding model: {self.embedding_model_name}")
            print(f"[CACHE] Using Hugging Face cache directory: {hf_cache}")
            print(f"[CACHE] Using Transformers cache directory: {transformers_cache}")
            
            # Log cache location in application logs
            logger.info(f"Starting to load embedding model for chunking: {self.embedding_model_name}...")
            logger.info(f"Using Hugging Face cache directory: {hf_cache}")
            logger.info(f"Using Transformers cache directory: {transformers_cache}")
            
            self._update_status(f"Loading embedding model: {self.embedding_model_name}")
            
            # Configure PyTorch to use GPU if available
            if torch.cuda.is_available():
                print(f"[INFO] CUDA is available - using device: {torch.cuda.get_device_name(0)}")
                print(f"[INFO] CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB total")
                # Set PyTorch to use CUDA
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
                device = torch.device("cuda")
            else:
                print(f"[INFO] CUDA is not available - using CPU")
                device = torch.device("cpu")
                
            # Configure HuggingFaceEmbeddings with explicit cache location and parameters
            start_time = time.time()
            
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Try to create the model with different parameter sets
            # (to handle different versions of langchain_huggingface)
            model_creation_start = time.time()
            print(f"[CHUNKER] Creating HuggingFaceEmbeddings instance at {model_creation_start:.2f}...")
            logger.info(f"Creating HuggingFaceEmbeddings instance...")
            
            try:
                # First try with the more detailed configuration
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                    cache_folder=transformers_cache,
                    model_kwargs={
                        "device": device,
                        "use_auth_token": False,  # Set to True if using private models
                    },
                    encode_kwargs={"normalize_embeddings": True},
                )
                model_creation_time = time.time() - model_creation_start
                print(f"[CHUNKER] Created embedding model with detailed configuration in {model_creation_time:.2f}s")
                logger.info(f"Created embedding model with detailed configuration in {model_creation_time:.2f}s")
            except TypeError as te:
                # Fall back to simpler configuration if the above fails
                print(f"[WARNING] Detailed configuration failed ({str(te)}), falling back to basic configuration")
                logger.warning(f"Detailed configuration failed: {te}")
                
                fallback_start = time.time()
                print(f"[CHUNKER] Trying fallback configuration at {fallback_start:.2f}...")
                logger.info(f"Trying fallback configuration...")
                
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                )
                
                fallback_time = time.time() - fallback_start
                print(f"[CHUNKER] Created embedding model with basic configuration in {fallback_time:.2f}s")
                logger.info(f"Created embedding model with basic configuration in {fallback_time:.2f}s")
            
            elapsed_time = time.time() - start_time
            
            # Log successful loading with timing information
            logger.info(f"Embedding model {self.embedding_model_name} loaded successfully in {elapsed_time:.2f} seconds")
            print(f"[SUCCESS] Embedding model loaded successfully in {elapsed_time:.2f} seconds")
            
            # Log model device information (safely check for attributes)
            try:
                # Try different attribute names that might exist
                if hasattr(self.embedding_model, 'client'):
                    device = getattr(self.embedding_model.client, 'device', 'unknown')
                elif hasattr(self.embedding_model, '_client'):
                    device = getattr(self.embedding_model._client, 'device', 'unknown')
                else:
                    device = "unknown (client attribute not found)"
                print(f"[INFO] Model loaded on device: {device}")
            except Exception as device_error:
                print(f"[INFO] Model loaded (device info unavailable: {device_error})")
            
            self._update_status(f"Embedding model loaded successfully")
            
        except Exception as e:
            error_msg = f"Error loading embedding model: {e}"
            self._update_status(error_msg)
            logger.error(error_msg)
            print(f"[ERROR] Failed to load embedding model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def shutdown(self):
        """
        Unload embedding model to free up memory.
        """
        if self.embedding_model is not None:
            print(f"[CHUNKER] ===== STARTING MODEL UNLOAD =====")
            logger.info("===== STARTING MODEL UNLOAD =====")
            self._update_status("Unloading embedding model")
            
            # Delete the model reference
            print(f"[CHUNKER] Deleting model reference...")
            logger.info("Deleting model reference...")
            del self.embedding_model
            self.embedding_model = None
            
            # Clean up CUDA memory
            if torch.cuda.is_available():
                print(f"[CHUNKER] Clearing CUDA cache...")
                logger.info("Clearing CUDA cache...")
                torch.cuda.empty_cache()
            
            # Force garbage collection
            print(f"[CHUNKER] Running garbage collection...")
            logger.info("Running garbage collection...")
            gc.collect()
            
            print(f"[CHUNKER] ===== MODEL UNLOAD COMPLETE =====")
            logger.info("===== MODEL UNLOAD COMPLETE =====")
            
            log_memory_usage(logger)
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a document using semantic chunking.
        
        Args:
            document (dict): Document data from DocumentLoader
            
        Returns:
            list: List of chunk dictionaries
        """
        start_time = time.time()
        
        doc_id = document.get('document_id', 'unknown')
        doc_name = document.get('file_name', 'unknown')
        doc_type = document.get('file_type', 'unknown')
        
        logger.info(f"Starting chunking for document: {doc_name} (ID: {doc_id}, Type: {doc_type})")
        print(f"[CHUNKER] Starting chunking for document: {doc_name} (ID: {doc_id}, Type: {doc_type})")
        self._update_status(f"Chunking document: {doc_id}")
        
        chunks = []
        
        try:
            
            # Process each content item (page or section)
            content_items = document.get('content', [])
            total_text_size = sum(len(item.get('text', '')) for item in content_items)
            
            logger.info(f"Document has {len(content_items)} content items, total size: {total_text_size} characters")
            print(f"[CHUNKER] Document has {len(content_items)} items with {total_text_size} characters total")
            
            total_chunks_created = 0
            
            for i, content_item in enumerate(content_items):
                item_start = time.time()
                progress = ((i + 1) / len(content_items)) * 0.9  # Progress up to 90%
                
                page_num = content_item.get('page_num', None)
                text = content_item.get('text', '')
                text_size = len(text)
                
                # Skip empty content
                if not text.strip():
                    logger.info(f"Skipping empty content item {i+1}/{len(content_items)}")
                    continue
                
                # Log item details
                page_info = f" (page {page_num})" if page_num else ""
                logger.info(f"Processing content item {i+1}/{len(content_items)}{page_info}, size: {text_size} chars")
                print(f"[CHUNKER] Processing item {i+1}/{len(content_items)}{page_info}, size: {text_size} chars")
                
                # Generate semantic chunks for this content
                self._update_status(f"Chunking content item {i+1}/{len(content_items)}{page_info}", progress)
                
                chunking_start = time.time()
                print(f"[CHUNKER] Starting semantic chunking for item {i+1} at {chunking_start:.2f}...")
                logger.info(f"Starting semantic chunking for item {i+1}...")
                content_chunks = self._semantic_chunking(text)
                chunking_time = time.time() - chunking_start
                print(f"[CHUNKER] Semantic chunking for item {i+1} completed in {chunking_time:.2f}s")
                logger.info(f"Semantic chunking for item {i+1} completed in {chunking_time:.2f}s")
                
                # Log chunking details
                logger.info(f"Item {i+1} chunked into {len(content_chunks)} chunks in {chunking_time:.2f}s")
                print(f"[CHUNKER] Item {i+1} chunked into {len(content_chunks)} chunks in {chunking_time:.2f}s")
                
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
                
                total_chunks_created += len(content_chunks)
                
                # Log item complete time
                item_time = time.time() - item_start
                if item_time > 1.0:
                    logger.info(f"Content item {i+1} processed in {item_time:.2f}s")
                    print(f"[CHUNKER] Content item {i+1} processed in {item_time:.2f}s")
            
            # End of chunking
            total_time = time.time() - start_time
            avg_chunk_size = sum(len(chunk.get('text', '')) for chunk in chunks) / len(chunks) if chunks else 0
            
            result_msg = (
                f"Created {len(chunks)} chunks for document {doc_name} in {total_time:.2f}s. "
                f"Average chunk size: {avg_chunk_size:.1f} characters"
            )
            
            logger.info(result_msg)
            print(f"[CHUNKER] {result_msg}")
            self._update_status(f"Created {len(chunks)} chunks for document {doc_id}", 1.0)
            log_memory_usage(logger)
            
            return chunks
            
        except Exception as e:
            error_msg = f"Error chunking document {doc_id}: {e}"
            self._update_status(error_msg)
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            print(f"[CHUNKER ERROR] {error_msg}")
            print(traceback.format_exc())
            raise
        finally:
            # We don't unload the model here anymore - it will be unloaded in shutdown()
            pass
    
    def _semantic_chunking(self, text: str) -> List[str]:
        import time
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
            chunk_count = len(initial_chunks)
            logger.info(f"Initial boundary splitting created {chunk_count} chunks")
            print(f"[CHUNKING] Initial boundary splitting created {chunk_count} chunks")
            
            # Step 2: Apply semantic chunking to each large chunk
            from langchain_experimental.text_splitter import SemanticChunker
            
            # Ensure CUDA is set properly for PyTorch if available (fallback in case model setup didn't do it)
            if torch.cuda.is_available():
                print(f"[INFO] CUDA is available - forcing embedded models to use GPU")
                # Try to ensure any future tensor operations use CUDA
                import os
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            
            semantic_chunks = []
            large_chunk_count = sum(1 for chunk in initial_chunks if len(chunk) > self.chunk_size)
            print(f"[CHUNKING] Found {large_chunk_count} large chunks for semantic splitting")
            
            # Initialize text_splitter once outside the loop
            text_splitter = None
            if large_chunk_count > 0:
                print(f"[CHUNKING] Applying semantic chunking to {large_chunk_count} large chunks")
                
                # Ensure model is loaded before creating the splitter
                if self.embedding_model is None:
                    logger.warning("Embedding model not loaded, attempting to load now")
                    print(f"[CHUNKING] Embedding model not loaded, loading now...")
                    self.load_model()
                    print(f"[CHUNKING] Embedding model loaded on-demand")
                else:
                    logger.debug("Using pre-loaded embedding model for semantic chunking")
                
                # Create SemanticChunker once for all chunks
                splitter_start = time.time()
                print(f"[CHUNKING] Creating SemanticChunker (once) at {splitter_start:.2f}...")
                logger.info("Creating SemanticChunker instance (once)...")
                
                text_splitter = SemanticChunker(
                    self.embedding_model,
                    breakpoint_threshold_type="percentile",
                    breakpoint_threshold_amount=95.0
                )
                
                splitter_time = time.time() - splitter_start
                print(f"[CHUNKING] SemanticChunker created in {splitter_time:.2f}s (will be reused)")
                logger.info(f"SemanticChunker created in {splitter_time:.2f}s (will be reused)")
            
            for i, chunk in enumerate(initial_chunks):
                # Only apply semantic chunking to larger chunks
                if len(chunk) > self.chunk_size and text_splitter is not None:
                    try:
                        # Update on progress
                        if large_chunk_count > 1:
                            print(f"[CHUNKING] Processing large chunk {i+1}/{large_chunk_count} with semantic chunker")
                        
                        # Apply semantic chunking with reused splitter object
                        chunking_start = time.time()
                        print(f"[CHUNKING] Starting create_documents for chunk {i+1} at {chunking_start:.2f}...")
                        logger.info(f"Starting create_documents for chunk {i+1}...")
                        
                        docs = text_splitter.create_documents([chunk])
                        results = [doc.page_content for doc in docs]
                        
                        chunking_time = time.time() - chunking_start
                        print(f"[CHUNKING] create_documents for chunk {i+1} completed in {chunking_time:.2f}s")
                        logger.info(f"create_documents for chunk {i+1} completed in {chunking_time:.2f}s")
                        
                        print(f"[CHUNKING] Semantic chunker split text of {len(chunk)} chars into {len(results)} chunks")
                        semantic_chunks.extend(results)
                    except Exception as chunk_error:
                        # Handle errors in individual chunk processing
                        logger.warning(f"Error in semantic chunking for chunk {i}: {chunk_error}")
                        print(f"[WARNING] Error in semantic chunking for chunk {i}: {chunk_error}")
                        print(f"[WARNING] Falling back to sentence splitting for this chunk")
                        
                        # Fall back to sentence splitting for this chunk
                        semantic_chunks.extend(self._sentence_splitting(chunk))
                else:
                    # Keep small chunks as-is
                    semantic_chunks.append(chunk)
            
            # Step 3: Apply fallback splitting for any chunks still too large
            final_chunks = []
            oversized_chunks = sum(1 for chunk in semantic_chunks if len(chunk) > self.chunk_size * 1.5)
            
            if oversized_chunks > 0:
                print(f"[CHUNKING] {oversized_chunks} chunks are still oversized, applying sentence splitting")
                
            for chunk in semantic_chunks:
                if len(chunk) > self.chunk_size * 1.5:  # Allow some flexibility
                    # Split oversized chunks using sentence boundaries
                    sentence_chunks = self._sentence_splitting(chunk)
                    final_chunks.extend(sentence_chunks)
                    print(f"[CHUNKING] Split chunk of {len(chunk)} chars into {len(sentence_chunks)} sentence chunks")
                else:
                    final_chunks.append(chunk)
            
            # Log results with semantic chunker optimization info
            logger.info(f"Semantic chunking pipeline created {len(final_chunks)} chunks from text of length {len(text)}")
            
            if large_chunk_count > 1 and text_splitter is not None:
                print(f"[CHUNKING] Optimization: Used a single SemanticChunker instance for {large_chunk_count} chunks")
                logger.info(f"Optimization: Used a single SemanticChunker instance for {large_chunk_count} chunks")
                
            print(f"[CHUNKING] Final result: {len(final_chunks)} chunks from text of length {len(text)}")
            
            return final_chunks
            
        except Exception as e:
            logger.error(f"Error during semantic chunking: {e}")
            logger.warning("Falling back to basic chunking")
            print(f"[ERROR] Semantic chunking failed: {e}")
            print(f"[WARNING] Falling back to basic chunking")
            import traceback
            traceback.print_exc()
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
