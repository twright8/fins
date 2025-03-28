"""
Centralized model management module to load and maintain ML models
for the processing pipeline. This prevents repeated loading/unloading
of the same models across different pipeline stages.
"""
import sys
import os
from pathlib import Path
import time
import torch
import gc
from typing import Dict, Any, Optional

# Add parent directory to sys.path to enable imports from project root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.core.config import CONFIG
from src.utils.logger import setup_logger
from src.utils.resource_monitor import log_memory_usage

logger = setup_logger(__name__)

class ModelManager:
    """
    Singleton class to manage model loading and provide access to loaded models.
    Ensures models are loaded only once and kept in memory for reuse across
    different pipeline components.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Ensure singleton pattern - only one ModelManager instance exists"""
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, status_queue=None):
        """
        Initialize model manager if not already initialized.
        
        Args:
            status_queue (Queue, optional): Queue for status updates.
        """
        # Skip if already initialized (singleton pattern)
        if self._initialized:
            return
            
        self.status_queue = status_queue
        
        # Initialize model holders
        self.embedding_model = None
        self.flair_ner_model = None
        self.flair_relation_model = None
        self.coref_model = None
        
        # Set model names from config
        self.embedding_model_name = CONFIG["models"]["embedding_model"]
        self.ner_model_name = CONFIG["models"]["ner_model"]
        
        # Configure cache directories
        self._configure_cache_directories()
        
        # Set initialized flag
        self._initialized = True
        
        logger.info(f"Model manager initialized")
        if self.status_queue:
            self.status_queue.put(('status', 'Model manager initialized'))
        
        log_memory_usage(logger)
    
    def _configure_cache_directories(self):
        """Configure cache directories for all model libraries"""
        # Set up consistent cache directories
        cache_root = os.path.expanduser('~/.cache')
        
        # HuggingFace cache
        os.environ['HF_HOME'] = os.path.join(cache_root, 'huggingface')
        
        # Transformers cache
        os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.environ['HF_HOME'], 'transformers')
        
        # Flair cache
        os.environ["FLAIR_CACHE_ROOT"] = os.path.join(cache_root, 'flair')
        
        # Log cache locations
        logger.info(f"Cache directories configured:")
        logger.info(f"  HF_HOME: {os.environ['HF_HOME']}")
        logger.info(f"  TRANSFORMERS_CACHE: {os.environ['TRANSFORMERS_CACHE']}")
        logger.info(f"  FLAIR_CACHE_ROOT: {os.environ['FLAIR_CACHE_ROOT']}")
        
        print(f"[MODELS] Cache directories configured:")
        print(f"[MODELS]   HF_HOME: {os.environ['HF_HOME']}")
        print(f"[MODELS]   TRANSFORMERS_CACHE: {os.environ['TRANSFORMERS_CACHE']}")
        print(f"[MODELS]   FLAIR_CACHE_ROOT: {os.environ['FLAIR_CACHE_ROOT']}")
    
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
    
    def load_all_models(self):
        """
        Load all pipeline models at once.
        """
        print(f"[MODELS] ===== STARTING TO LOAD ALL MODELS =====")
        logger.info("===== STARTING TO LOAD ALL MODELS =====")
        
        self._update_status("Loading all models for processing pipeline")
        
        start_time = time.time()
        
        # 1. Load embedding model (for semantic chunking)
        self.load_embedding_model()
        
        # 2. Load coreference model
        self.load_coref_model()
        
        # 3. Load Flair NER model
        self.load_flair_ner_model()
        
        # 4. Load Flair relation model
        self.load_flair_relation_model()
        
        total_time = time.time() - start_time
        
        self._update_status(f"All models loaded in {total_time:.2f}s")
        log_memory_usage(logger)
        
        print(f"[MODELS] ===== ALL MODELS LOADED IN {total_time:.2f}s =====")
        logger.info(f"===== ALL MODELS LOADED IN {total_time:.2f}s =====")
    
    def load_embedding_model(self):
        """
        Load embedding model for semantic chunking.
        """
        if self.embedding_model is not None:
            logger.info("Embedding model already loaded")
            print(f"[MODELS] Embedding model already loaded")
            return
        
        print(f"[MODELS] ===== LOADING EMBEDDING MODEL =====")
        logger.info(f"===== LOADING EMBEDDING MODEL =====")
        
        start_time = time.time()
        
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            import torch
            from transformers import logging as transformers_logging
            
            # Set transformers logging level
            transformers_logging.set_verbosity_info()
            
            # Print caching info
            hf_cache = os.environ.get('HF_HOME')
            transformers_cache = os.environ.get('TRANSFORMERS_CACHE')
            
            print(f"[MODELS] Loading embedding model: {self.embedding_model_name}")
            print(f"[MODELS] Using HuggingFace cache: {hf_cache}")
            print(f"[MODELS] Using Transformers cache: {transformers_cache}")
            
            # Configure PyTorch to use GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if torch.cuda.is_available():
                print(f"[MODELS] CUDA is available - using device: {torch.cuda.get_device_name(0)}")
            else:
                print(f"[MODELS] CUDA is not available - using CPU")
            
            # Key optimization: preload import time
            import_start = time.time()
            print(f"[MODELS] Preparing HuggingFace libraries at {import_start:.2f}")
            
            # Actual model creation
            model_start = time.time()
            print(f"[MODELS] Creating HuggingFaceEmbeddings instance at {model_start:.2f}")
            
            try:
                # First try with detailed configuration
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                    cache_folder=transformers_cache,
                    model_kwargs={
                        "device": device,
                        "use_auth_token": False,
                    },
                    encode_kwargs={"normalize_embeddings": True},
                )
                print(f"[MODELS] Created embedding model with detailed configuration")
            except TypeError as te:
                # Fall back to simpler configuration
                print(f"[MODELS] Detailed configuration failed ({str(te)}), falling back to basic configuration")
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                )
                print(f"[MODELS] Created embedding model with basic configuration")
            
            model_time = time.time() - model_start
            total_time = time.time() - start_time
            
            # Warm up the model with a test encoding to make sure everything is loaded
            warmup_start = time.time()
            print(f"[MODELS] Warming up embedding model with test encoding")
            _ = self.embedding_model.embed_query("This is a test sentence.")
            warmup_time = time.time() - warmup_start
            
            print(f"[MODELS] Embedding model creation: {model_time:.2f}s")
            print(f"[MODELS] Embedding model warmup: {warmup_time:.2f}s")
            print(f"[MODELS] Total embedding model loading: {total_time:.2f}s")
            
            logger.info(f"Embedding model loaded in {total_time:.2f}s")
            
        except Exception as e:
            error_msg = f"Error loading embedding model: {e}"
            self._update_status(error_msg)
            logger.error(error_msg)
            print(f"[MODELS ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            raise
    
    def load_coref_model(self):
        """
        Load FastCoref model for coreference resolution.
        """
        if self.coref_model is not None:
            logger.info("FastCoref model already loaded")
            print(f"[MODELS] FastCoref model already loaded")
            return
        
        print(f"[MODELS] ===== LOADING COREFERENCE MODEL =====")
        logger.info(f"===== LOADING COREFERENCE MODEL =====")
        
        start_time = time.time()
        
        try:
            # Import FastCoref
            import_start = time.time()
            print(f"[MODELS] Importing FastCoref at {import_start:.2f}")
            from fastcoref import FCoref
            import_time = time.time() - import_start
            print(f"[MODELS] FastCoref imported in {import_time:.2f}s")
            
            # Load model
            model_start = time.time()
            print(f"[MODELS] Creating FastCoref model instance at {model_start:.2f}")
            
            try:
                # Try GPU first
                self.coref_model = FCoref(device='cuda:0')
                print(f"[MODELS] FastCoref model loaded on GPU")
            except Exception as gpu_error:
                print(f"[MODELS] GPU initialization failed: {gpu_error}, falling back to CPU")
                self.coref_model = FCoref(device='cpu')
                print(f"[MODELS] FastCoref model loaded on CPU")
            
            model_time = time.time() - model_start
            total_time = time.time() - start_time
            
            # Warm up
            warmup_start = time.time()
            print(f"[MODELS] Warming up FastCoref model with test prediction")
            _ = self.coref_model.predict(texts=["This is a test sentence."])
            warmup_time = time.time() - warmup_start
            
            print(f"[MODELS] FastCoref model creation: {model_time:.2f}s")
            print(f"[MODELS] FastCoref model warmup: {warmup_time:.2f}s")
            print(f"[MODELS] Total FastCoref model loading: {total_time:.2f}s")
            
            logger.info(f"FastCoref model loaded in {total_time:.2f}s")
            
        except Exception as e:
            error_msg = f"Error loading FastCoref model: {e}"
            self._update_status(error_msg)
            logger.error(error_msg)
            print(f"[MODELS ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            raise
    
    def load_flair_ner_model(self):
        """
        Load Flair NER model for entity extraction.
        """
        if self.flair_ner_model is not None:
            logger.info("Flair NER model already loaded")
            print(f"[MODELS] Flair NER model already loaded")
            return
        
        print(f"[MODELS] ===== LOADING FLAIR NER MODEL =====")
        logger.info(f"===== LOADING FLAIR NER MODEL =====")
        
        start_time = time.time()
        
        try:
            # Import Flair
            import_start = time.time()
            print(f"[MODELS] Importing Flair libraries at {import_start:.2f}")
            from flair.nn import Classifier
            from flair.data import Sentence
            import_time = time.time() - import_start
            print(f"[MODELS] Flair imported in {import_time:.2f}s")
            
            # Check cache
            cache_dir = os.environ.get("FLAIR_CACHE_ROOT")
            model_name = "flair/ner-english-ontonotes-fast"
            
            # Load model
            model_start = time.time()
            print(f"[MODELS] Loading Flair NER model ({model_name}) at {model_start:.2f}")
            
            self.flair_ner_model = Classifier.load(model_name)
            
            model_time = time.time() - model_start
            total_time = time.time() - start_time
            
            # Warm up
            warmup_start = time.time()
            print(f"[MODELS] Warming up Flair NER model with test prediction")
            test_sentence = Sentence("John Smith works for Microsoft in Seattle.")
            self.flair_ner_model.predict(test_sentence)
            warmup_time = time.time() - warmup_start
            
            print(f"[MODELS] Flair NER model loading: {model_time:.2f}s")
            print(f"[MODELS] Flair NER model warmup: {warmup_time:.2f}s")
            print(f"[MODELS] Total Flair NER model loading: {total_time:.2f}s")
            
            logger.info(f"Flair NER model loaded in {total_time:.2f}s")
            
        except Exception as e:
            error_msg = f"Error loading Flair NER model: {e}"
            self._update_status(error_msg)
            logger.error(error_msg)
            print(f"[MODELS ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            raise
    
    def load_flair_relation_model(self):
        """
        Load Flair relation extraction model.
        """
        # Skip if already loaded
        if self.flair_relation_model is not None:
            logger.info("Flair relation model already loaded")
            print(f"[MODELS] Flair relation model already loaded")
            return
        
        print(f"[MODELS] ===== LOADING FLAIR RELATION MODEL =====")
        logger.info(f"===== LOADING FLAIR RELATION MODEL =====")
        
        start_time = time.time()
        
        try:
            # Import Flair if needed
            if not 'Classifier' in globals():
                import_start = time.time()
                print(f"[MODELS] Importing Flair libraries at {import_start:.2f}")
                from flair.nn import Classifier
                from flair.data import Sentence
                import_time = time.time() - import_start
                print(f"[MODELS] Flair imported in {import_time:.2f}s")
            
            # Load model
            model_start = time.time()
            print(f"[MODELS] Loading Flair relation model at {model_start:.2f}")
            
            try:
                self.flair_relation_model = Classifier.load('relations')
                model_time = time.time() - model_start
                total_time = time.time() - start_time
                
                # Warm up
                warmup_start = time.time()
                print(f"[MODELS] Warming up Flair relation model with test prediction")
                test_sentence = Sentence("John Smith works for Microsoft in Seattle.")
                if 'test_sentence' not in locals():
                    test_sentence = Sentence("John Smith works for Microsoft in Seattle.")
                if hasattr(self.flair_ner_model, 'predict'):
                    self.flair_ner_model.predict(test_sentence)
                self.flair_relation_model.predict(test_sentence)
                warmup_time = time.time() - warmup_start
                
                print(f"[MODELS] Flair relation model loading: {model_time:.2f}s")
                print(f"[MODELS] Flair relation model warmup: {warmup_time:.2f}s")
                print(f"[MODELS] Total Flair relation model loading: {total_time:.2f}s")
                
                logger.info(f"Flair relation model loaded in {total_time:.2f}s")
                
            except Exception as rel_error:
                print(f"[MODELS] Error loading relation model: {rel_error}")
                logger.warning(f"Error loading relation model: {rel_error}")
                self.flair_relation_model = None
            
        except Exception as e:
            error_msg = f"Error in relation model loading process: {e}"
            self._update_status(error_msg)
            logger.error(error_msg)
            print(f"[MODELS ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            self.flair_relation_model = None
    
    def unload_all_models(self):
        """
        Unload all models and free resources.
        """
        print(f"[MODELS] ===== UNLOADING ALL MODELS =====")
        logger.info("===== UNLOADING ALL MODELS =====")
        
        self._update_status("Unloading all models to free resources")
        
        start_time = time.time()
        
        # Delete all model references
        self.embedding_model = None
        self.flair_ner_model = None
        self.flair_relation_model = None
        self.coref_model = None
        
        # Clean up CUDA memory
        if torch.cuda.is_available():
            print(f"[MODELS] Clearing CUDA cache")
            torch.cuda.empty_cache()
        
        # Force garbage collection
        print(f"[MODELS] Running garbage collection")
        gc.collect()
        
        total_time = time.time() - start_time
        
        print(f"[MODELS] All models unloaded in {total_time:.2f}s")
        logger.info(f"All models unloaded in {total_time:.2f}s")
        
        log_memory_usage(logger)
    
    def get_models(self):
        """
        Get all loaded models.
        
        Returns:
            dict: Dictionary of all loaded models
        """
        return {
            "embedding_model": self.embedding_model,
            "flair_ner_model": self.flair_ner_model,
            "flair_relation_model": self.flair_relation_model,
            "coref_model": self.coref_model
        }
