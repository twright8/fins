"""
Configuration module for the Anti-Corruption RAG system.
Contains all configurable parameters for the system.
"""
import os
import yaml
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
TEMP_DIR = PROJECT_ROOT / "temp"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(DATA_DIR / "bm25_indices", exist_ok=True)

# Default configuration
DEFAULT_CONFIG = {
    # Document Processing
    "document_processing": {
        "chunk_size": 512,  # Target chunk size in tokens
        "chunk_overlap": 50,  # Overlap between chunks in tokens
        "ocr_parallel_jobs": max(1, os.cpu_count() - 2),  # Leave 2 cores free
    },
    
    # Models
    "models": {
        "embedding_model": "intfloat/multilingual-e5-base",  # For document/query embedding
        "reranking_model": "mixedbread-ai/mxbai-rerank-xsmall-v1",  # For reranking
        "ner_model": "flair/ner-english-ontonotes-fast",  # For named entity recognition
        "coref_model": "maverick-coref",  # For coreference resolution
        "llm_model": "Qwen/Qwen2.5-0.5B-Instruct",  # For generation
    },
    
    # Retrieval
    "retrieval": {
        "top_k_vector": 20,  # Number of vector search results
        "top_k_bm25": 10,  # Number of BM25 search results
        "top_k_hybrid": 15,  # Number of results after fusion
        "top_k_rerank": 5,  # Number of results after reranking
        "vector_weight": 0.7,  # Weight for vector search in hybrid retrieval
        "bm25_weight": 0.3,  # Weight for BM25 search in hybrid retrieval
    },
    
    # LLM Generation
    "generation": {
        # Common parameters
        "temperature": 0.3,  # Temperature for generation
        "max_tokens": 1024,  # Maximum tokens to generate
        "top_p": 0.9,  # Top-p sampling parameter
        "top_k": 50,  # Top-k sampling parameter
        "presence_penalty": 0.0,  # Presence penalty
        
        # Provider selection - can be "aphrodite" or "deepseek"
        "provider": "aphrodite",  # Default provider
        
        # Aphrodite (local) configuration
        "aphrodite": {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",  # Model name
            "max_model_len": 8192,  # Maximum model context length
            "gpu_memory_utilization": 0.85,  # GPU memory utilization
            "quantization": "fp8",  # Quantization type
        },
        
        # DeepSeek API configuration
        "deepseek": {
            "api_key": "",  # DeepSeek API key (empty means not configured)
            "api_base": "https://api.deepseek.com/v1",  # API base URL
            "model": "deepseek-chat",  # DeepSeek model ID
            "timeout": 60,  # Request timeout in seconds
        },
    },
    
    # Qdrant Vector DB
    "qdrant": {
        "host": "localhost",  # Qdrant host
        "port": 6333,  # Qdrant port
        "collection_name": "anti_corruption_docs",  # Collection name
        "vector_size": 768,  # Vector size (depends on embedding model)
    },
    
    # Entity Extraction
    "entity_extraction": {
        "confidence_threshold": 0.3,  # Minimum confidence for entity extraction
        "relationship_threshold": 0.3,  # Minimum confidence for relationship extraction
        "fuzzy_match_threshold": 90,  # Fuzzy matching threshold (0-100) for entity deduplication
    },
    
    # UI
    "ui": {
        "page_title": "Anti-Corruption RAG System",
        "page_icon": "🔍",
        "theme_color": "#1E3A8A",  # Primary color (dark blue)
        "accent_color": "#3B82F6",  # Accent color (bright blue)
        "max_upload_size_mb": 200,  # Maximum upload size in MB
    }
}


def load_config():
    """
    Load configuration from YAML file, or create default if not exists.
    
    Returns:
        dict: Configuration dictionary
    """
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                config = yaml.safe_load(f)
                # Merge with defaults to ensure all required keys exist
                merged_config = DEFAULT_CONFIG.copy()
                for section, values in config.items():
                    if section in merged_config and isinstance(values, dict):
                        merged_config[section].update(values)
                    else:
                        merged_config[section] = values
                return merged_config
        except Exception as e:
            print(f"Error loading config: {e}. Using defaults.")
            return DEFAULT_CONFIG
    else:
        # Create default config file
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False)
        return DEFAULT_CONFIG


# Export the loaded configuration
CONFIG = load_config()
