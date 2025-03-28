"""
Retrieval module that combines BM25 and vector search.
"""
import sys
import os
from pathlib import Path
import pickle
import torch
import numpy as np
import gc
from typing import List, Dict, Any, Union, Optional
import time
import traceback

# Add parent directory to sys.path to enable imports from project root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.core.config import CONFIG, DATA_DIR
from src.utils.logger import setup_logger
from src.utils.resource_monitor import log_memory_usage

logger = setup_logger(__name__)

class Retriever:
    """
    Retriever that combines BM25 and vector search for hybrid retrieval.
    """
    
    def __init__(self, lazy_init=False, qdrant_only=False):
        """
        Initialize the retriever.
        
        Args:
            lazy_init (bool): If True, don't load embedding models on initialization
            qdrant_only (bool): If True, only initialize Qdrant connection without embedding models
        """
        # BM25 configuration
        self.bm25_dir = DATA_DIR / "bm25_indices"
        self.bm25_index = None
        self.tokenized_texts = None
        self.bm25_metadata = None
        
        # Qdrant configuration
        self.qdrant_host = CONFIG["qdrant"]["host"]
        self.qdrant_port = CONFIG["qdrant"]["port"]
        self.qdrant_collection = CONFIG["qdrant"]["collection_name"]
        
        # Embedding model configuration
        self.embedding_model_name = CONFIG["models"]["embedding_model"]
        self.reranking_model_name = CONFIG["models"]["reranking_model"]
        self.embed_register = None
        
        # Retrieval parameters
        self.top_k_vector = CONFIG["retrieval"]["top_k_vector"]
        self.top_k_bm25 = CONFIG["retrieval"]["top_k_bm25"]
        self.top_k_hybrid = CONFIG["retrieval"]["top_k_hybrid"]
        self.top_k_rerank = CONFIG["retrieval"]["top_k_rerank"]
        self.vector_weight = CONFIG["retrieval"]["vector_weight"]
        self.bm25_weight = CONFIG["retrieval"]["bm25_weight"]
        
        logger.info(f"Initializing Retriever with BM25 directory={self.bm25_dir}, "
                   f"Qdrant={self.qdrant_host}:{self.qdrant_port}, "
                   f"collection={self.qdrant_collection}, "
                   f"embedding_model={self.embedding_model_name}")
        
        # Load BM25 index (this is quick)
        self._load_bm25_index()
        
        # Handle initialization based on parameters
        if qdrant_only:
            logger.info("Initializing with Qdrant only - embedding models will not be loaded")
        elif not lazy_init:
            logger.info("Loading embedding models during initialization")
            self._load_embedding_model()
        else:
            logger.info("Using lazy initialization - embedding models will be loaded on first use")
    
    def _load_bm25_index(self) -> bool:
        """
        Load the BM25 index from disk.
        
        Returns:
            bool: Success status
        """
        try:
            bm25_file = self.bm25_dir / "latest_index.pkl"
            if not bm25_file.exists():
                logger.warning(f"BM25 index file not found: {bm25_file}")
                return False
            
            logger.info(f"Loading BM25 index from {bm25_file}")
            with open(bm25_file, 'rb') as f:
                self.bm25_index, self.tokenized_texts, self.bm25_metadata = pickle.load(f)
            
            logger.info(f"BM25 index loaded with {len(self.tokenized_texts)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error loading BM25 index: {e}")
            return False
    
    def _load_embedding_model(self):
        """
        Load the embedding model if not already loaded.
        """
        if self.embed_register is not None:
            return
        
        try:
            from embed import BatchedInference
            import time
            import os
            
            # Log cache location for debugging
            hf_cache = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
            logger.info(f"Using Hugging Face cache directory: {hf_cache}")
            
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Starting to load embedding and reranking models via Infinity:")
            logger.info(f"- Embedding model: {self.embedding_model_name}")
            logger.info(f"- Reranking model: {self.reranking_model_name}")
            logger.info(f"- Device: {device}")
            
            # Time the model loading
            start_time = time.time()
            
            self.embed_register = BatchedInference(
                model_id=[
                    self.embedding_model_name,
                    self.reranking_model_name
                ],
                engine="torch",
                device=device
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"Embedding and reranking models loaded successfully in {elapsed_time:.2f} seconds")
            
            # Log memory usage after loading
            import gc
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
                mem_reserved = torch.cuda.memory_reserved() / (1024**3)    # Convert to GB
                logger.info(f"GPU memory after model loading: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    def _unload_embedding_model(self):
        """
        Unload embedding model to free up memory.
        """
        if self.embed_register is not None:
            logger.info("Unloading embedding models")
            
            try:
                # Safely stop the register, handling potential attribute errors
                self.embed_register.stop()
            except AttributeError as ae:
                # Handle known issue with SyncEngineArray.__del__ in infinity_emb
                logger.warning(f"Attribute error when stopping infinity embedding model: {ae}")
                print(f"[WARNING] Known issue with infinity-embed cleanup: {ae}")
                # We'll still try to clean up as much as possible
            except Exception as e:
                # Log other exceptions but continue cleanup
                logger.error(f"Error stopping infinity embedding model: {e}")
            
            # Delete the register reference regardless of stop() success
            del self.embed_register
            self.embed_register = None
            
            # Clean up CUDA memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            log_memory_usage(logger)
    
    def get_chunks(self, limit: int = 20, search_text: str = None, document_filter: str = None) -> List[Dict[str, Any]]:
        """
        Get chunks from the vector database for exploration.
        
        Args:
            limit (int): Maximum number of chunks to retrieve
            search_text (str, optional): Text to search for within chunks
            document_filter (str, optional): Filter by document name
            
        Returns:
            list: List of chunks with their metadata
        """
        try:
            import time
            start_time = time.time()
            
            from qdrant_client import QdrantClient, models
            from qdrant_client.http import exceptions as qdrant_exceptions
            
            # Connect to Qdrant with timeout settings
            client = QdrantClient(
                host=self.qdrant_host, 
                port=self.qdrant_port,
                timeout=5.0  # 5 second timeout for operations
            )
            
            # Build filter if needed
            filter_conditions = []
            
            if search_text:
                # Use match instead of text for better performance
                filter_conditions.append({
                    "must": [{
                        "key": "text",
                        "match": {"value": search_text}
                    }]
                })
            
            if document_filter:
                filter_conditions.append({
                    "must": [{
                        "key": "file_name",
                        "match": {"value": document_filter}
                    }]
                })
            
            # Combine filters or use empty filter if none provided
            filter_obj = models.Filter(must=filter_conditions) if filter_conditions else None
            
            # Validate collection exists before querying to avoid hanging
            collections = client.get_collections().collections
            collection_names = [c.name for c in collections]
            if self.qdrant_collection not in collection_names:
                logger.warning(f"Collection {self.qdrant_collection} does not exist")
                return []
            
            # Get chunks with timeout protection
            try:
                scroll_result = client.scroll(
                    collection_name=self.qdrant_collection,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                    filter=filter_obj,
                    timeout=10  # 10 second timeout for scroll operation
                )
                
                points = scroll_result[0]
            except qdrant_exceptions.UnexpectedResponse as e:
                logger.error(f"Qdrant returned unexpected response: {e}")
                return []
            except qdrant_exceptions.TimeoutError:
                logger.error("Qdrant query timed out")
                return []
            
            # Format results
            chunks = []
            for point in points:
                # Make sure payload exists to avoid errors
                if not hasattr(point, 'payload') or point.payload is None:
                    continue
                
                chunks.append({
                    'id': point.id,
                    'text': point.payload.get('text', ''),
                    'metadata': {k: v for k, v in point.payload.items() if k != 'text'}
                })
            
            # Log performance
            elapsed_time = time.time() - start_time
            logger.info(f"Retrieved {len(chunks)} chunks in {elapsed_time:.2f}s")
            
            return chunks
            
        except Exception as e:
            import traceback
            logger.error(f"Error retrieving chunks: {e}")
            logger.error(traceback.format_exc())
            return []
            
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the vector collection.
        
        Returns:
            dict: Collection information including point count
        """
        try:
            import time
            start_time = time.time()
            
            from qdrant_client import QdrantClient
            from qdrant_client.http import exceptions as qdrant_exceptions
            
            # Connect to Qdrant with timeout
            client = QdrantClient(
                host=self.qdrant_host, 
                port=self.qdrant_port,
                timeout=3.0  # 3 second timeout
            )
            
            try:
                # Check if collection exists - with timeout
                collections_response = client.get_collections(timeout=3.0)
                collections = collections_response.collections
                collection_names = [c.name for c in collections]
                
                if self.qdrant_collection in collection_names:
                    # Get collection info - with timeout
                    collection_info = client.get_collection(
                        collection_name=self.qdrant_collection,
                        timeout=3.0
                    )
                    
                    # Get only what we need to avoid potential serialization issues
                    result = {
                        'exists': True,
                        'name': self.qdrant_collection,
                        'points_count': getattr(collection_info, 'points_count', 0),
                        'vector_size': getattr(collection_info.config.params.vectors, 'size', 0),
                        'distance': str(getattr(collection_info.config.params.vectors, 'distance', 'unknown'))
                    }
                    
                    # Log performance
                    elapsed_time = time.time() - start_time
                    logger.info(f"Retrieved collection info in {elapsed_time:.2f}s")
                    
                    return result
                else:
                    logger.warning(f"Collection {self.qdrant_collection} does not exist")
                    return {'exists': False}
                    
            except qdrant_exceptions.UnexpectedResponse as e:
                logger.error(f"Qdrant returned unexpected response: {e}")
                return {'exists': False, 'error': f"Unexpected response: {str(e)}"}
            except qdrant_exceptions.TimeoutError:
                logger.error("Qdrant query timed out")
                return {'exists': False, 'error': "Connection timed out"}
            
        except Exception as e:
            import traceback
            logger.error(f"Error getting collection info: {e}")
            logger.error(traceback.format_exc())
            return {'exists': False, 'error': str(e)}
    
    def shutdown(self):
        """
        Shutdown the retriever and free resources.
        """
        logger.info("Shutting down retriever")
        self._unload_embedding_model()
    
    def retrieve(self, query: str, use_reranking: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using hybrid search (BM25 + vector).
        
        Args:
            query (str): Query string
            use_reranking (bool, optional): Whether to use reranking. Defaults to True.
            
        Returns:
            list: List of retrieved chunks with scores
        """
        try:
            # Start timing
            start_time = time.time()
            
            # Validate inputs
            if not query.strip():
                return []
            
            # Check if BM25 index is loaded
            if self.bm25_index is None:
                success = self._load_bm25_index()
                if not success:
                    logger.warning("BM25 index not available, using vector search only")
            
            # Ensure embedding model is loaded
            if self.embed_register is None:
                self._load_embedding_model()
            
            # Get BM25 results
            bm25_results = self._bm25_search(query) if self.bm25_index is not None else []
            
            # Get vector search results
            vector_results = self._vector_search(query)
            
            # Combine results (fusion)
            fused_results = self._fuse_results(bm25_results, vector_results)
            
            # Rerank if requested
            if use_reranking and fused_results:
                reranked_results = self._rerank(query, fused_results)
                final_results = reranked_results
            else:
                final_results = fused_results
            
            # Log timing
            elapsed_time = time.time() - start_time
            logger.info(f"Retrieval completed in {elapsed_time:.2f}s. Found {len(final_results)} results.")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            # Return empty list on error
            return []
    
    def _bm25_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search using BM25.
        
        Args:
            query (str): Query string
            
        Returns:
            list: List of search results with scores
        """
        import nltk
        from nltk.tokenize import word_tokenize
        import pickle
        import os
        
        try:
            # Ensure NLTK tokenizer is available
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
            # Load stopwords if available, otherwise use empty set
            stop_words = set()
            stop_words_file = self.bm25_dir / "stopwords.pkl"
            if os.path.exists(stop_words_file):
                try:
                    with open(stop_words_file, 'rb') as f:
                        stop_words = pickle.load(f)
                except Exception as e:
                    logger.warning(f"Error loading stopwords: {e}. Proceeding without stopwords.")
            else:
                # Try loading from NLTK if available
                try:
                    from nltk.corpus import stopwords
                    nltk.data.find('corpora/stopwords')
                    stop_words = set(stopwords.words('english'))
                except Exception:
                    logger.warning("Stopwords not available. Proceeding without stopwords.")
            
            # Tokenize and filter the query
            tokenized_query = word_tokenize(query.lower())
            
            # Apply stopword filtering consistently with how the index was built
            filtered_query = [token for token in tokenized_query if token not in stop_words]
            
            # If all tokens were stopwords, use the original tokens to avoid empty query
            if not filtered_query and tokenized_query:
                logger.warning("All query tokens were stopwords. Using original query.")
                filtered_query = tokenized_query
            
            # Get BM25 scores
            bm25_scores = self.bm25_index.get_scores(filtered_query)
            
            # Create results array with document ids and scores
            results = []
            for i, score in enumerate(bm25_scores):
                if score > 0:  # Only include non-zero scores
                    results.append({
                        'index': i,
                        'score': float(score),
                        'text': ' '.join(self.tokenized_texts[i]),
                        'metadata': self.bm25_metadata[i] if i < len(self.bm25_metadata) else {}
                    })
            
            # Sort by score (descending)
            results.sort(key=lambda x: x['score'], reverse=True)
            
            # Limit to top_k
            results = results[:self.top_k_bm25]
            
            logger.info(f"BM25 search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []
    
    def _vector_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search using vector search (Qdrant).
        
        Args:
            query (str): Query string
            
        Returns:
            list: List of search results with scores
        """
        try:
            from qdrant_client import QdrantClient
            
            # Generate query embedding
            future = self.embed_register.embed(
                sentences=[query],
                model_id=self.embedding_model_name
            )
            query_embedding = future.result()[0]
            
            # Connect to Qdrant
            client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
            
            # Search in Qdrant
            search_result = client.search(
                collection_name=self.qdrant_collection,
                query_vector=query_embedding.tolist(),
                limit=self.top_k_vector
            )
            
            # Format results
            results = []
            for point in search_result:
                results.append({
                    'id': point.id,
                    'score': float(point.score),
                    'text': point.payload.get('text', ''),
                    'metadata': {k: v for k, v in point.payload.items() if k != 'text'}
                })
            
            logger.info(f"Vector search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    def _fuse_results(self, bm25_results: List[Dict[str, Any]], vector_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fuse results from BM25 and vector search using Reciprocal Rank Fusion (RRF).
        
        Args:
            bm25_results (list): BM25 search results
            vector_results (list): Vector search results
            
        Returns:
            list: Fused search results using RRF
        """
        # If either result set is empty, return the other
        if not bm25_results:
            return vector_results[:self.top_k_hybrid]
        if not vector_results:
            return bm25_results[:self.top_k_hybrid]
        
        # RRF constant k (prevents division by zero and controls the impact of high ranks)
        k = 60
        
        # Create dictionaries for ranking lookup
        bm25_dict = {}
        vector_dict = {}
        
        # Assign ranks to BM25 results (sorted by score in descending order)
        for rank, result in enumerate(sorted(bm25_results, key=lambda x: x['score'], reverse=True)):
            doc_id = result['metadata'].get('chunk_id', result.get('id', f"bm25_{rank}"))
            bm25_dict[doc_id] = {
                'rank': rank + 1,  # 1-based ranking
                'score': result['score'],
                'text': result['text'],
                'metadata': result['metadata']
            }
        
        # Assign ranks to vector search results
        for rank, result in enumerate(sorted(vector_results, key=lambda x: x['score'], reverse=True)):
            doc_id = result['metadata'].get('chunk_id', result.get('id', f"vector_{rank}"))
            vector_dict[doc_id] = {
                'rank': rank + 1,  # 1-based ranking
                'score': result['score'],
                'text': result['text'],
                'metadata': result['metadata']
            }
        
        # Combine all unique document IDs
        all_doc_ids = set(bm25_dict.keys()) | set(vector_dict.keys())
        
        # Calculate RRF scores
        rrf_results = []
        for doc_id in all_doc_ids:
            # Get ranks (defaulting to a large number if not in a result set)
            bm25_rank = bm25_dict.get(doc_id, {'rank': len(bm25_results) + 100})['rank']
            vector_rank = vector_dict.get(doc_id, {'rank': len(vector_results) + 100})['rank']
            
            # Calculate RRF score
            rrf_score = (1 / (k + bm25_rank)) + (1 / (k + vector_rank))
            
            # Get document info from whichever source has it
            doc_info = bm25_dict.get(doc_id) or vector_dict.get(doc_id)
            
            rrf_results.append({
                'id': doc_id,
                'score': rrf_score,
                'bm25_rank': bm25_rank,
                'vector_rank': vector_rank,
                'text': doc_info['text'],
                'metadata': doc_info['metadata']
            })
        
        # Sort by RRF score (descending)
        rrf_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Limit to top_k
        results = rrf_results[:self.top_k_hybrid]
        
        logger.info(f"RRF fusion returned {len(results)} results")
        return results
    
    def _rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank results using a dedicated reranker model.
        
        Args:
            query (str): Query string
            results (list): Search results to rerank
            
        Returns:
            list: Reranked search results
        """
        try:
            # Extract texts from results
            texts = [result['text'] for result in results]
            
            # Skip reranking if no texts
            if not texts:
                return results
            
            # Rerank using the reranking model
            future = self.embed_register.rerank(
                query=query,
                docs=texts,
                model_id=self.reranking_model_name
            )
            rerank_scores = future.result()
            
            # Create new results with reranker scores
            reranked_results = []
            for i, score in enumerate(rerank_scores):
                result = results[i].copy()
                result['original_score'] = result['score']  # Save original score
                result['score'] = float(score)  # Use reranker score
                reranked_results.append(result)
            
            # Sort by reranker score (descending)
            reranked_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Limit to top_k
            reranked_results = reranked_results[:self.top_k_rerank]
            
            logger.info(f"Reranking returned {len(reranked_results)} results")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            return results  # Return original results on error
