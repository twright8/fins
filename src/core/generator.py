"""
Generation module using Aphrodite for LLM-based text generation.
"""
import sys
import os
from pathlib import Path
import traceback
import time
from typing import List, Dict, Any, Optional

# Add parent directory to sys.path to enable imports from project root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.core.config import CONFIG
from src.utils.logger import setup_logger
from src.utils.resource_monitor import log_memory_usage

logger = setup_logger(__name__)

class Generator:
    """
    Text generator using Aphrodite LLM.
    """
    
    def __init__(self):
        """Initialize the generator."""
        # LLM configuration
        self.model_name = CONFIG["models"]["llm_model"]
        self.temperature = CONFIG["generation"]["temperature"]
        self.max_tokens = CONFIG["generation"]["max_tokens"]
        self.top_p = CONFIG["generation"]["top_p"]
        self.top_k = CONFIG["generation"]["top_k"]
        self.presence_penalty = CONFIG["generation"]["presence_penalty"]
        self.max_model_len = CONFIG["generation"]["max_model_len"]
        self.gpu_memory_utilization = CONFIG["generation"]["gpu_memory_utilization"]
        self.quantization = CONFIG["generation"]["quantization"]
        
        # Model instance
        self.model = None
        self.sampling_params = None
        self.model_loaded = False
        
        logger.info(f"Initializing Generator with model={self.model_name}")
    
    def _load_model(self) -> bool:
        """
        Load the LLM model using Aphrodite.
        
        Returns:
            bool: Success status
        """
        if self.model_loaded:
            return True
        
        try:
            logger.info(f"Loading model with Aphrodite: {self.model_name}")
            
            try:
                from aphrodite import LLM, SamplingParams
            except ImportError:
                logger.error("Aphrodite not installed. Please install with: pip install aphrodite-engine")
                return False
            
            aphrodite_kwargs = {
                "model": self.model_name,
                "trust_remote_code": True,
                "dtype": "half",  # Use half precision
                "max_model_len": self.max_model_len,
                "tensor_parallel_size": 1,  # Assuming single GPU for this model size
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "quantization": self.quantization,
                "cpu_offload_gb": 0  # Keep model on GPU
            }
            
            logger.info(f"Starting Aphrodite with options: {aphrodite_kwargs}")
            self.model = LLM(**aphrodite_kwargs)
            
            self.sampling_params = lambda: SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                top_k=self.top_k,
                presence_penalty=self.presence_penalty
            )
            
            self.model_loaded = True
            logger.info(f"Successfully loaded model: {self.model_name}")
            log_memory_usage(logger)
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading Aphrodite model: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def generate(self, prompt: str) -> str:
        """
        Generate text based on the given prompt.
        
        Args:
            prompt (str): Input prompt
            
        Returns:
            str: Generated text
        """
        try:
            # Start timing
            start_time = time.time()
            
            # Validate input
            if not prompt.strip():
                return ""
            
            # Load model if not already loaded
            if not self.model_loaded:
                success = self._load_model()
                if not success:
                    return "Error: Failed to load model"
            
            # Generate text
            logger.info(f"Generating text for prompt of length {len(prompt)}")
            response = self.model.generate(prompt, self.sampling_params())
            
            # Extract generated text
            generated_text = response.outputs[0].text
            
            # Log timing
            elapsed_time = time.time() - start_time
            logger.info(f"Generation completed in {elapsed_time:.2f}s. Generated {len(generated_text)} characters.")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error in text generation: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error in text generation: {str(e)}"
    
    def generate_with_context(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate text based on the query and retrieved context chunks.
        
        Args:
            query (str): User query
            retrieved_chunks (list): List of retrieved context chunks
            
        Returns:
            str: Generated text with citations
        """
        try:
            # Validate inputs
            if not query.strip():
                return "Error: Query is empty"
            
            if not retrieved_chunks:
                return "I don't have enough context to answer that question. Please upload relevant documents or try a different query."
            
            # Format context
            context = self._format_context(retrieved_chunks)
            
            # Construct prompt
            prompt = self._construct_rag_prompt(query, context)
            
            # Generate response
            response = self.generate(prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in context generation: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error in generation: {str(e)}"
    
    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into a context string for the prompt.
        
        Args:
            chunks (list): List of retrieved chunks
            
        Returns:
            str: Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            # Format context with metadata and citation
            doc_id = chunk['metadata'].get('document_id', 'unknown')
            file_name = chunk['metadata'].get('file_name', 'unknown')
            page_num = chunk['metadata'].get('page_num', None)
            
            # Create citation reference
            citation = f"[{i+1}] {file_name}"
            if page_num:
                citation += f", page {page_num}"
            
            # Format the chunk text with citation
            context_parts.append(f"Context {i+1}: {chunk['text']}\nSource: {citation}\n")
        
        return "\n".join(context_parts)
    
    def _construct_rag_prompt(self, query: str, context: str) -> str:
        """
        Construct a RAG prompt with query and context.
        
        Args:
            query (str): User query
            context (str): Formatted context string
            
        Returns:
            str: Complete prompt
        """
        prompt = f"""You are an Anti-Corruption Intelligence Analyst assistant. Your task is to answer questions based on the provided context from documents related to potential corruption cases.

Please follow these guidelines:
1. Only use information present in the provided context to answer the question.
2. If the context doesn't contain enough information to provide a complete answer, acknowledge the limitations of the available information.
3. If you're unsure or the context is contradictory, explain the contradictions or uncertainties.
4. Cite your sources using the reference numbers provided in the context, like [1], [2], etc.
5. Organize your response in a clear, concise, and structured manner.
6. Focus on extracting factual information and relationships between entities.

Here is the context information:

{context}

Question: {query}

Answer: """
        
        return prompt
