"""
Retrieval module that combines BM25 and vector search.
"""
import sys
import os
from pathlib import Path
import pickle
import torch
import numpy as np
from typing import List, Dict, Any, Union, Optional
import time

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
    
    def __init__(self):
        """Initialize the retriever."""
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
        
        # Load BM25 index
        self._load_bm25_index()
    
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
            
            logger.info(f"Loading embedding models via Infinity: {self.embedding_model_name}, {self.reranking_model_name}")
            self.embed_register = BatchedInference(
                model_id=[
                    self.embedding_model_name,
                    self.reranking_model_name
                ],
                engine="torch",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            logger.info("Embedding models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    def _unload_embedding_model(self):
        """
        Unload embedding model to free up memory.
        """
        if self.embed_register is not None:
            logger.info("Unloading embedding models")
            
            # Stop the register
            self.embed_register.stop()
            
            # Delete the register reference
            del self.embed_register
            self.embed_register = None
            
            # Clean up CUDA memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            log_memory_usage(logger)
    
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
            
            # Load embedding model
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
            
            # Unload embedding model
            self._unload_embedding_model()
            
            # Log timing
            elapsed_time = time.time() - start_time
            logger.info(f"Retrieval completed in {elapsed_time:.2f}s. Found {len(final_results)} results.")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            
            # Make sure to unload model even if there's an error
            self._unload_embedding_model()
            
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
        
        try:
            # Ensure NLTK tokenizer is available
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
            # Tokenize the query
            tokenized_query = word_tokenize(query.lower())
            
            # Get BM25 scores
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            
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
        Fuse results from BM25 and vector search using reciprocal rank fusion.
        
        Args:
            bm25_results (list): BM25 search results
            vector_results (list): Vector search results
            
        Returns:
            list: Fused search results
        """
        # If either result set is empty, return the other
        if not bm25_results:
            return vector_results[:self.top_k_hybrid]
        if not vector_results:
            return bm25_results[:self.top_k_hybrid]
        
        # Create a dictionary to store combined scores by document ID
        # Using chunk_id as the key
        combined_docs = {}
        
        # Normalize BM25 scores (max normalization)
        max_bm25_score = max([r['score'] for r in bm25_results]) if bm25_results else 1.0
        for i, result in enumerate(bm25_results):
            doc_id = result['metadata'].get('chunk_id', f"bm25_{i}")
            normalized_score = result['score'] / max_bm25_score if max_bm25_score > 0 else 0
            
            combined_docs[doc_id] = {
                'bm25_score': normalized_score,
                'vector_score': 0.0,
                'text': result['text'],
                'metadata': result['metadata']
            }
        
        # Normalize vector scores (usually already normalized to cosine similarity)
        for i, result in enumerate(vector_results):
            doc_id = result['metadata'].get('chunk_id', result['id'])
            
            if doc_id in combined_docs:
                # Document exists in BM25 results, update vector score
                combined_docs[doc_id]['vector_score'] = result['score']
            else:
                # New document from vector search
                combined_docs[doc_id] = {
                    'bm25_score': 0.0,
                    'vector_score': result['score'],
                    'text': result['text'],
                    'metadata': result['metadata']
                }
        
        # Combine scores with weights
        results = []
        for doc_id, doc in combined_docs.items():
            combined_score = (
                self.bm25_weight * doc['bm25_score'] +
                self.vector_weight * doc['vector_score']
            )
            
            results.append({
                'id': doc_id,
                'score': combined_score,
                'bm25_score': doc['bm25_score'],
                'vector_score': doc['vector_score'],
                'text': doc['text'],
                'metadata': doc['metadata']
            })
        
        # Sort by combined score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Limit to top_k
        results = results[:self.top_k_hybrid]
        
        logger.info(f"Fusion returned {len(results)} results")
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
