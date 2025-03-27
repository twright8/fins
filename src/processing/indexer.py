"""
Indexing module for BM25 and vector search.
"""
import sys
import os
from pathlib import Path
import pickle
import torch
import gc
from typing import List, Dict, Any, Tuple
import logging
import uuid

# Add parent directory to sys.path to enable imports from project root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.core.config import CONFIG, DATA_DIR
from src.utils.logger import setup_logger
from src.utils.resource_monitor import log_memory_usage

logger = setup_logger(__name__)

class Indexer:
    """
    Indexer for both BM25 and vector search (Qdrant).
    """
    
    def __init__(self, status_queue=None):
        """
        Initialize indexer.
        
        Args:
            status_queue (Queue, optional): Queue for status updates.
        """
        self.status_queue = status_queue
        self.bm25_index = None
        self.embedding_model = None
        
        # BM25 configuration
        self.bm25_dir = DATA_DIR / "bm25_indices"
        os.makedirs(self.bm25_dir, exist_ok=True)
        
        # Qdrant configuration
        self.qdrant_host = CONFIG["qdrant"]["host"]
        self.qdrant_port = CONFIG["qdrant"]["port"]
        self.qdrant_collection = CONFIG["qdrant"]["collection_name"]
        self.vector_size = CONFIG["qdrant"]["vector_size"]
        
        # Embedding model configuration
        self.embedding_model_name = CONFIG["models"]["embedding_model"]
        
        logger.info(f"Initializing Indexer with BM25 directory={self.bm25_dir}, "
                   f"Qdrant={self.qdrant_host}:{self.qdrant_port}, "
                   f"collection={self.qdrant_collection}, "
                   f"embedding_model={self.embedding_model_name}")
        
        if self.status_queue:
            self.status_queue.put(('status', 'Indexer initialized'))
    
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
    
    def index_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Index chunks with both BM25 and vector search.
        
        Args:
            chunks (list): List of chunk dictionaries
            
        Returns:
            bool: Success status
        """
        self._update_status(f"Indexing {len(chunks)} chunks")
        
        try:
            # Extract texts and prepare metadata for indexing
            texts = []
            chunk_metadata = []
            
            for chunk in chunks:
                texts.append(chunk.get('text', ''))
                
                # Prepare metadata for Qdrant
                metadata = {
                    'chunk_id': chunk.get('chunk_id', str(uuid.uuid4())),
                    'document_id': chunk.get('document_id', ''),
                    'file_name': chunk.get('file_name', ''),
                    'page_num': chunk.get('page_num', None),
                    'chunk_idx': chunk.get('chunk_idx', 0)
                }
                
                # Add any other metadata that would be useful for retrieval
                if 'metadata' in chunk and isinstance(chunk['metadata'], dict):
                    for k, v in chunk['metadata'].items():
                        if k not in metadata and isinstance(v, (str, int, float, bool, type(None))):
                            metadata[k] = v
                
                chunk_metadata.append(metadata)
            
            # Index with BM25
            bm25_success = self._index_with_bm25(texts, chunk_metadata)
            
            # Index with Qdrant
            vector_success = self._index_with_vectors(texts, chunk_metadata)
            
            return bm25_success and vector_success
            
        except Exception as e:
            error_msg = f"Error indexing chunks: {e}"
            self._update_status(error_msg)
            logger.error(error_msg)
            raise
    
    def _index_with_bm25(self, texts: List[str], metadata: List[Dict[str, Any]]) -> bool:
        """
        Index texts with BM25.
        
        Args:
            texts (list): List of text strings
            metadata (list): List of metadata dictionaries
            
        Returns:
            bool: Success status
        """
        self._update_status("Building BM25 index", 0.5)
        
        try:
            from rank_bm25 import BM25Okapi
            import nltk
            from nltk.tokenize import word_tokenize
            
            # Ensure NLTK tokenizer is available
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                self._update_status("Downloading NLTK punkt tokenizer")
                nltk.download('punkt')
            
            # Tokenize the texts
            tokenized_texts = []
            for text in texts:
                tokens = word_tokenize(text.lower())
                tokenized_texts.append(tokens)
            
            # Create BM25 index
            self.bm25_index = BM25Okapi(tokenized_texts)
            
            # Append metadata to the index
            self.bm25_index.metadata = metadata
            
            # Save the index
            bm25_file = self.bm25_dir / "latest_index.pkl"
            with open(bm25_file, 'wb') as f:
                pickle.dump((self.bm25_index, tokenized_texts, metadata), f)
            
            self._update_status(f"BM25 index saved to {bm25_file}", 0.6)
            return True
            
        except Exception as e:
            error_msg = f"Error creating BM25 index: {e}"
            self._update_status(error_msg)
            logger.error(error_msg)
            return False
    
    def _index_with_vectors(self, texts: List[str], metadata: List[Dict[str, Any]]) -> bool:
        """
        Index texts with vector search using Infinity Embeddings and Qdrant.
        
        Args:
            texts (list): List of text strings
            metadata (list): List of metadata dictionaries
            
        Returns:
            bool: Success status
        """
        self._update_status("Loading embedding model via Infinity", 0.6)
        
        try:
            from embed import BatchedInference
            from qdrant_client import QdrantClient, models
            import torch
            import gc
            
            # Setup Infinity BatchedInference
            register = BatchedInference(
                model_id=[self.embedding_model_name],
                engine="torch",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            try:
                # Generate embeddings for chunks
                self._update_status(f"Generating embeddings for {len(texts)} chunks", 0.7)
                future = register.embed(sentences=texts, model_id=self.embedding_model_name)
                embeddings = future.result()  # Blocks until completion
                
                # Connect to Qdrant
                self._update_status(f"Connecting to Qdrant ({self.qdrant_host}:{self.qdrant_port})", 0.8)
                client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
                
                # Create collection if it doesn't exist
                collections = client.get_collections().collections
                collection_names = [c.name for c in collections]
                
                if self.qdrant_collection not in collection_names:
                    self._update_status(f"Creating collection: {self.qdrant_collection}")
                    client.create_collection(
                        collection_name=self.qdrant_collection,
                        vectors_config=models.VectorParams(
                            size=self.vector_size,
                            distance=models.Distance.COSINE
                        )
                    )
                
                # Prepare points for upsert
                self._update_status(f"Preparing points for Qdrant", 0.9)
                points = []
                
                for i, embedding in enumerate(embeddings):
                    # Generate a unique Qdrant ID for this point
                    # Use chunk_id from metadata if available, otherwise generate one
                    chunk_id = metadata[i].get('chunk_id', str(uuid.uuid4()))
                    
                    # Create payload with text and metadata
                    payload = {
                        'text': texts[i],
                        **metadata[i]
                    }
                    
                    # Create point
                    points.append(
                        models.PointStruct(
                            id=str(chunk_id),
                            vector=embedding.tolist(),
                            payload=payload
                        )
                    )
                
                # Upsert points to Qdrant
                self._update_status(f"Upserting {len(points)} points to Qdrant", 0.95)
                client.upsert(
                    collection_name=self.qdrant_collection,
                    points=points,
                    wait=True
                )
                
                self._update_status(f"Vector indexing complete", 1.0)
                return True
                
            finally:
                # Cleanup
                self._update_status("Stopping Infinity embedding model")
                register.stop()
                del register
                
                # Clean up CUDA memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Force garbage collection
                gc.collect()
                
                log_memory_usage(logger)
                
        except Exception as e:
            error_msg = f"Error creating vector index: {e}"
            self._update_status(error_msg)
            logger.error(error_msg)
            return False
