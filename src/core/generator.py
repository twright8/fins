"""
Generation module supporting multiple LLM providers for text generation.
"""
import sys
import os
from pathlib import Path
import traceback
import time
import json
import requests
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Add parent directory to sys.path to enable imports from project root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.core.config import CONFIG
from src.utils.logger import setup_logger
from src.utils.resource_monitor import log_memory_usage

logger = setup_logger(__name__)

class Generator:
    """
    Text generator supporting multiple providers (Aphrodite local and DeepSeek API).
    """
    
    def __init__(self):
        """Initialize the generator with the configured provider."""
        # Get provider from config
        self.provider = CONFIG["generation"]["provider"]
        
        # Common generation parameters
        self.temperature = CONFIG["generation"]["temperature"]
        self.max_tokens = CONFIG["generation"]["max_tokens"]
        self.top_p = CONFIG["generation"]["top_p"]
        self.top_k = CONFIG["generation"]["top_k"]
        self.presence_penalty = CONFIG["generation"]["presence_penalty"]
        
        # Aphrodite specific attributes
        self.aphrodite_model = None
        self.aphrodite_sampling_params = None
        self.aphrodite_loaded = False
        
        # DeepSeek specific attributes
        self.deepseek_config = CONFIG["generation"]["deepseek"]
        self.deepseek_api_key = self.deepseek_config["api_key"]
        self.deepseek_available = bool(self.deepseek_api_key.strip())
        
        # Check if DeepSeek should be used but API key is missing
        if self.provider == "deepseek" and not self.deepseek_available:
            logger.warning("DeepSeek selected as provider but API key is missing. Falling back to Aphrodite.")
            self.provider = "aphrodite"
        
        logger.info(f"Initializing Generator with provider={self.provider}")
        
        # Load Aphrodite model if it's the selected provider
        if self.provider == "aphrodite":
            self._load_aphrodite()
    
    def _load_aphrodite(self) -> bool:
        """
        Load the LLM model using Aphrodite.
        
        Returns:
            bool: Success status
        """
        if self.aphrodite_loaded:
            return True
        
        try:
            aphrodite_config = CONFIG["generation"]["aphrodite"]
            model_name = aphrodite_config["model"]
            
            logger.info(f"Loading model with Aphrodite: {model_name}")
            
            try:
                from aphrodite import LLM, SamplingParams
            except ImportError:
                logger.error("Aphrodite not installed. Please install with: pip install aphrodite-engine")
                return False
            
            aphrodite_kwargs = {
                "model": model_name,
                "trust_remote_code": True,
                "dtype": "half",  # Use half precision
                "max_model_len": aphrodite_config["max_model_len"],
                "tensor_parallel_size": 1,  # Assuming single GPU for this model size
                "gpu_memory_utilization": aphrodite_config["gpu_memory_utilization"],
                "quantization": aphrodite_config["quantization"],
                "cpu_offload_gb": 0  # Keep model on GPU
            }
            
            logger.info(f"Starting Aphrodite with options: {aphrodite_kwargs}")
            self.aphrodite_model = LLM(**aphrodite_kwargs)
            
            self.aphrodite_sampling_params = lambda: SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                top_k=self.top_k,
                presence_penalty=self.presence_penalty
            )
            
            self.aphrodite_loaded = True
            logger.info(f"Successfully loaded model: {model_name}")
            log_memory_usage(logger)
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading Aphrodite model: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.Timeout))
    )
    def _call_deepseek_api(self, prompt: str) -> str:
        """
        Call the DeepSeek API to generate text.
        
        Args:
            prompt (str): Input prompt
            
        Returns:
            str: Generated text
        """
        if not self.deepseek_available:
            logger.error("DeepSeek API key not configured")
            return "Error: DeepSeek API key not configured"
        
        try:
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.deepseek_config["model"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "stream": False
            }
            
            url = f"{self.deepseek_config['api_base']}/chat/completions"
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.deepseek_config["timeout"]
            )
            
            # Raise an exception if the request was unsuccessful
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            # Check for errors in the response
            if "error" in result:
                logger.error(f"DeepSeek API error: {result['error']}")
                return f"Error from DeepSeek API: {result['error']}"
            
            # Extract the generated text
            if ("choices" in result and 
                len(result["choices"]) > 0 and 
                "message" in result["choices"][0] and
                "content" in result["choices"][0]["message"]):
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"Unexpected response structure from DeepSeek API: {result}")
                return "Error: Unexpected response structure from DeepSeek API"
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error calling DeepSeek API: {str(e)}")
            raise  # Let the retry decorator handle this
        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {str(e)}")
            return f"Error in text generation: {str(e)}"
    
    def generate(self, prompt: str) -> str:
        """
        Generate text based on the given prompt using the configured provider.
        
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
            
            # Generate using the selected provider
            if self.provider == "aphrodite":
                generated_text = self._generate_with_aphrodite(prompt)
            elif self.provider == "deepseek":
                generated_text = self._call_deepseek_api(prompt)
            else:
                logger.error(f"Unknown provider: {self.provider}")
                return f"Error: Unknown provider '{self.provider}'"
            
            # Log timing
            elapsed_time = time.time() - start_time
            logger.info(f"Generation completed in {elapsed_time:.2f}s using {self.provider}. Generated {len(generated_text)} characters.")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error in text generation: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error in text generation: {str(e)}"
    
    def _generate_with_aphrodite(self, prompt: str) -> str:
        """
        Generate text using local Aphrodite instance.
        
        Args:
            prompt (str): Input prompt
            
        Returns:
            str: Generated text
        """
        # Load model if not already loaded
        if not self.aphrodite_loaded:
            success = self._load_aphrodite()
            if not success:
                return "Error: Failed to load Aphrodite model"
        
        # Generate text
        logger.info(f"Generating text with Aphrodite for prompt of length {len(prompt)}")
        response = self.aphrodite_model.generate(prompt, self.aphrodite_sampling_params())
        
        # Extract generated text
        return response.outputs[0].text
    
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
            
            # Generate response using the selected provider
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
