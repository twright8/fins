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
import concurrent.futures

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
            from nltk.corpus import stopwords
            
            # Ensure NLTK resources are available
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                self._update_status("Downloading NLTK punkt tokenizer")
                nltk.download('punkt')
            
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                self._update_status("Downloading NLTK stopwords")
                nltk.download('stopwords')
            
            # Get English stopwords
            stop_words = set(stopwords.words('english'))
            
            # Tokenize and clean the texts
            tokenized_texts = []
            for text in texts:
                # Tokenize and convert to lowercase
                tokens = word_tokenize(text.lower())
                
                # Remove stopwords
                filtered_tokens = [token for token in tokens if token not in stop_words]
                
                tokenized_texts.append(filtered_tokens)
            
            # Create BM25 index
            self.bm25_index = BM25Okapi(tokenized_texts)
            
            # Append metadata to the index
            self.bm25_index.metadata = metadata
            
            # Save the index along with tokenized texts
            bm25_file = self.bm25_dir / "latest_index.pkl"
            with open(bm25_file, 'wb') as f:
                pickle.dump((self.bm25_index, tokenized_texts, metadata), f)
            
            # Also save stopwords for consistent retrieval
            stop_words_file = self.bm25_dir / "stopwords.pkl"
            with open(stop_words_file, 'wb') as f:
                pickle.dump(stop_words, f)
            
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
        
        register = None
        try:
            from embed import BatchedInference
            from qdrant_client import QdrantClient, models
            import torch
            import gc
            import time
            import os
            
            # Log cache location for debugging
            hf_cache = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
            logger.info(f"Using Hugging Face cache directory: {hf_cache}")
            
            # Log model loading
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Starting to load Infinity embedding model: {self.embedding_model_name} on {device}...")
            
            start_time = time.time()
            
            # Setup Infinity BatchedInference
            register = BatchedInference(
                model_id=[self.embedding_model_name],
                engine="torch",
                device=device
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"Infinity embedding model loaded successfully in {elapsed_time:.2f} seconds")
            
            # Generate embeddings for chunks
            self._update_status(f"Generating embeddings for {len(texts)} chunks", 0.7)
            
            # Print directly to console for better visibility
            print(f"\n[EMBEDDING] Generating embeddings for {len(texts)} chunks with model {self.embedding_model_name}")
            
            # Additional debug information
            print(f"[DEBUG] First few characters of first text: '{texts[0][:50]}...'")
            
            # We need to handle a few edge cases in the embed library:
            # 1. Sometimes it returns integers for very short or empty texts
            # 2. Sometimes the embedding dimension doesn't match the expected size
            # 3. Sometimes the embeddings are returned as tensors, sometimes as lists
            
            # First remove any empty texts which might cause issues
            filtered_texts = []
            filtered_metadata = []
            for i, text in enumerate(texts):
                if text and len(text.strip()) > 0:
                    filtered_texts.append(text)
                    filtered_metadata.append(metadata[i])
                else:
                    print(f"[WARNING] Skipping empty text at position {i}")
            
            if len(filtered_texts) < len(texts):
                print(f"[WARNING] Removed {len(texts) - len(filtered_texts)} empty texts")
                
            if not filtered_texts:
                raise ValueError("No valid texts to embed")
                
            texts = filtered_texts
            metadata = filtered_metadata
            
            try:
                # Force a token initialization first to ensure model is properly loaded
                # This can help with some edge cases in infinity-embed
                print(f"[INFO] Initializing model with a test embedding...")
                test_future = register.embed(sentences=["This is a test sentence"], model_id=self.embedding_model_name)
                test_embedding = test_future.result()
                test_shape = None
                if isinstance(test_embedding, list) and len(test_embedding) > 0:
                    if hasattr(test_embedding[0], 'shape'):
                        test_shape = test_embedding[0].shape
                    elif hasattr(test_embedding[0], '__len__'):
                        test_shape = len(test_embedding[0])
                print(f"[INFO] Test embedding completed successfully. Shape: {test_shape}")
                
                # For small number of texts, process in a single batch
                # (infinity-embed typically returns a list of numpy arrays)
                print(f"[INFO] Processing {len(texts)} texts using model {self.embedding_model_name}")
                
                # Let's try with a longer timeout to ensure completion
                future = register.embed(sentences=texts, model_id=self.embedding_model_name)
                embeddings = future.result()  # This returns a list of numpy arrays
                
                # Debug information
                print(f"[DEBUG] Embeddings result type: {type(embeddings)}")
                
                # Handle the special case where embeddings is a tuple (reported in the error)
                if isinstance(embeddings, tuple):
                    print(f"[WARNING] Embeddings returned as a tuple with {len(embeddings)} elements")
                    # Check if the first element is a list of embeddings (common scenario)
                    if len(embeddings) > 0 and isinstance(embeddings[0], list) and len(embeddings[0]) > 0:
                        print(f"[INFO] Extracting embeddings from first element of tuple")
                        embeddings = embeddings[0]
                
                if embeddings and len(embeddings) > 0:
                    print(f"[DEBUG] First embedding type: {type(embeddings[0])}")
                    if hasattr(embeddings[0], 'shape'):
                        print(f"[DEBUG] First embedding shape: {embeddings[0].shape}")
                    elif hasattr(embeddings[0], '__len__'):
                        print(f"[DEBUG] First embedding length: {len(embeddings[0])}")
                    else:
                        print(f"[DEBUG] First embedding value: {str(embeddings[0])[:100]}")
                
                # Check if the embeddings might be in a transposed format
                # (e.g., one embedding of dimension [num_texts] instead of [num_texts] embeddings of dimension [vector_size])
                if len(embeddings) == self.vector_size and len(texts) != self.vector_size:
                    print(f"[WARNING] Embeddings appear to be in transposed format. Attempting to fix.")
                    # Use first embedding as a test
                    if hasattr(embeddings[0], '__len__') and len(embeddings[0]) == len(texts):
                        # This suggests we have vector_size embeddings of dimension num_texts
                        # Instead of num_texts embeddings of dimension vector_size
                        # We need to transpose
                        try:
                            import numpy as np
                            embeddings_array = np.array(embeddings)
                            embeddings = embeddings_array.T.tolist()
                            print(f"[INFO] Successfully transposed embeddings from shape ({len(embeddings)}, {len(embeddings[0])}) to ({len(embeddings[0])}, {len(embeddings)})")
                        except Exception as transpose_error:
                            print(f"[ERROR] Failed to transpose embeddings: {transpose_error}")
                
                # Verify we got the expected number of embeddings
                if len(embeddings) != len(texts):
                    print(f"[WARNING] Expected {len(texts)} embeddings but got {len(embeddings)}")
                
            except Exception as embed_error:
                print(f"\n[ERROR] Embedding generation failed with error: {embed_error}")
                print(f"[ERROR] Full traceback:")
                import traceback
                traceback.print_exc()
                
                # Try a fallback approach with smaller batches if we have more than a few texts
                if len(texts) > 4:
                    print(f"[INFO] Attempting fallback: processing in smaller batches")
                    try:
                        embeddings = []
                        # Calculate batch size from config or default to len/4
                        indexing_batch_size = CONFIG["retrieval"].get("indexing_batch_size", 0)
                        batch_size = indexing_batch_size if indexing_batch_size > 0 else max(1, len(texts) // 4)
                        
                        for i in range(0, len(texts), batch_size):
                            batch_end = min(i + batch_size, len(texts))
                            batch = texts[i:batch_end]
                            print(f"[INFO] Processing batch {i//batch_size + 1}/4: {len(batch)} texts")
                            
                            batch_future = register.embed(sentences=batch, model_id=self.embedding_model_name)
                            batch_embeddings = batch_future.result()
                            
                            # Check batch format and fix if needed
                            if isinstance(batch_embeddings, tuple) and len(batch_embeddings) > 0:
                                batch_embeddings = batch_embeddings[0]
                                
                            # Append to overall embeddings list
                            embeddings.extend(batch_embeddings)
                            
                        print(f"[INFO] Successfully processed {len(embeddings)} embeddings in batches")
                    except Exception as batch_error:
                        print(f"[ERROR] Batch fallback also failed: {batch_error}")
                        traceback.print_exc()
                        raise embed_error  # Raise the original error
                else:
                    raise
            
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
            print(f"[DEBUG] Preparing {len(embeddings)} points for Qdrant")
            points = []
            
            for i, embedding in enumerate(embeddings):
                try:
                    # Generate a unique Qdrant ID for this point
                    # Use chunk_id from metadata if available, otherwise generate one
                    chunk_id = metadata[i].get('chunk_id', str(uuid.uuid4()))
                    
                    # Create payload with text and metadata
                    payload = {
                        'text': texts[i],
                        **metadata[i]
                    }
                    
                    # Check for scalar values that should be skipped
                    if isinstance(embedding, (int, float)):
                        print(f"[ERROR] Embedding {i} is a scalar value ({embedding}), not a vector. Skipping.")
                        continue
                    
                    # Best case: it's a numpy array with proper shape
                    if hasattr(embedding, 'shape') and len(embedding.shape) == 1:
                        # Perfect - numpy array with proper shape
                        if embedding.shape[0] == self.vector_size:
                            vector_data = embedding.tolist()
                        else:
                            print(f"[ERROR] Numpy array dimension mismatch: got {embedding.shape[0]}, expected {self.vector_size}. Skipping.")
                            continue
                    # Handle different embedding formats    
                    elif hasattr(embedding, 'tolist'):
                        # It's a tensor or other array-like object
                        vector_data = embedding.tolist()
                        if i == 0:
                            print(f"[DEBUG] Embedding is an array-like that supports tolist()")
                    elif isinstance(embedding, list):
                        # It's already a list
                        vector_data = embedding
                        if i == 0:
                            print(f"[DEBUG] Embedding is already a list with {len(vector_data)} elements")
                    else:
                        # Unknown format - convert to list safely
                        if i == 0:
                            print(f"[DEBUG] Embedding has unexpected type: {type(embedding)}")
                        try:
                            # Try to convert to a list if possible
                            if hasattr(embedding, '__iter__'):
                                vector_data = list(embedding)
                            else:
                                print(f"[ERROR] Embedding {i} is not iterable: {embedding}. Skipping.")
                                continue
                        except Exception as e:
                            print(f"[ERROR] Could not convert embedding {i} to list: {e}")
                            if i == 0:
                                print(f"[DEBUG] Embedding representation: {str(embedding)[:100]}...")
                            print(f"[WARNING] Skipping this embedding instead of failing")
                            continue
                            
                    # Validate vector format and dimensions
                    if not isinstance(vector_data, list):
                        print(f"[ERROR] Vector data is not a list after conversion: {type(vector_data)}. Skipping.")
                        continue
                        
                    if len(vector_data) != self.vector_size:
                        print(f"[ERROR] Vector dimension mismatch: got {len(vector_data)}, expected {self.vector_size}. Skipping.")
                        continue
                        
                    # Validate that all elements are numeric
                    if not all(isinstance(x, (int, float)) for x in vector_data):
                        print(f"[ERROR] Vector contains non-numeric elements. Skipping.")
                        continue
                    
                    # Create point
                    points.append(
                        models.PointStruct(
                            id=str(chunk_id),
                            vector=vector_data,
                            payload=payload
                        )
                    )
                except Exception as point_error:
                    print(f"[ERROR] Error creating point {i}: {point_error}")
                    print(f"[DEBUG] Vector type: {type(embedding)}")
                    print(f"[DEBUG] Vector sample: {str(embedding)[:100] if hasattr(embedding, '__str__') else 'Cannot display'}")
                    raise
            
            # Helper function to upsert a batch of points
            def _upsert_batch_to_qdrant(client, collection_name, points_batch, batch_num, total_batches):
                try:
                    batch_start = time.time()
                    print(f"[INDEXER] Upserting batch {batch_num}/{total_batches} with {len(points_batch)} points")
                    
                    client.upsert(
                        collection_name=collection_name,
                        points=points_batch,
                        wait=True
                    )
                    
                    batch_time = time.time() - batch_start
                    print(f"[INDEXER] Batch {batch_num}/{total_batches} upserted in {batch_time:.2f}s")
                    return len(points_batch), batch_time
                except Exception as e:
                    print(f"[ERROR] Failed to upsert batch {batch_num}: {e}")
                    logger.error(f"Failed to upsert batch {batch_num}: {e}")
                    return 0, 0
            
            # Check if we have any valid points
            if not points:
                error_msg = "No valid embeddings were generated. Vector indexing failed."
                self._update_status(error_msg, 0.95)
                logger.error(error_msg)
                return False
                
            # Log how many points were skipped
            skipped_count = len(embeddings) - len(points)
            if skipped_count > 0:
                logger.warning(f"Skipped {skipped_count} out of {len(embeddings)} points due to invalid embeddings")
                print(f"[WARNING] Skipped {skipped_count} points due to invalid embeddings")
            
            # Get number of workers from config
            num_workers = CONFIG["document_processing"]["parallelism_workers"]["qdrant_upsert_batches"]
            
            # Decide on batch size - use a reasonable size for Qdrant batches
            total_points = len(points)
            batch_size = 100  # Default batch size
            num_batches = (total_points + batch_size - 1) // batch_size
            
            self._update_status(f"Upserting {total_points} points to Qdrant in {num_batches} batches using {num_workers} workers", 0.95)
            print(f"[INDEXER] Upserting {total_points} points to Qdrant in {num_batches} batches using {num_workers} workers")
            
            # Prepare batches
            point_batches = []
            for i in range(0, total_points, batch_size):
                batch = points[i:i+batch_size]
                point_batches.append((batch, i//batch_size + 1, num_batches))
            
            # Upsert batches in parallel
            total_upserted = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all upsert tasks
                future_to_batch = {
                    executor.submit(_upsert_batch_to_qdrant, client, self.qdrant_collection, batch, batch_num, num_batches): batch_num 
                    for batch, batch_num, num_batches in point_batches
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_batch):
                    batch_num = future_to_batch[future]
                    try:
                        points_upserted, batch_time = future.result()
                        total_upserted += points_upserted
                        
                        # Update progress
                        progress = 0.95 + (0.05 * (batch_num / num_batches))
                        self._update_status(f"Upserted batch {batch_num}/{num_batches} ({total_upserted}/{total_points} points total)", progress)
                        
                    except Exception as e:
                        logger.error(f"Error in upsert batch {batch_num}: {e}")
                        print(f"[ERROR] Error in upsert batch {batch_num}: {e}")
            
            self._update_status(f"Vector indexing complete: added {len(points)} vectors to collection '{self.qdrant_collection}'", 1.0)
            return True
                
        except Exception as e:
            error_msg = f"Error creating vector index: {e}"
            self._update_status(error_msg)
            logger.error(error_msg)
            return False
        finally:
            # Always ensure cleanup happens
            if register is not None:
                self._update_status("Stopping Infinity embedding model")
                try:
                    # Safely stop the register, handling potential attribute errors
                    register.stop()
                except AttributeError as ae:
                    # Handle known issue with SyncEngineArray.__del__ in infinity_emb
                    logger.warning(f"Attribute error when stopping infinity embedding model: {ae}")
                    print(f"[WARNING] Known issue with infinity-embed cleanup: {ae}")
                    # We'll still try to clean up as much as possible
                except Exception as e:
                    # Log other exceptions but continue cleanup
                    logger.error(f"Error stopping infinity embedding model: {e}")
                
                # Delete reference regardless of stop() success
                del register
                
                # Clean up CUDA memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Force garbage collection
                gc.collect()
                
                log_memory_usage(logger)
