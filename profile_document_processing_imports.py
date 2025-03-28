#!/usr/bin/env python3
"""
Script to profile import times for the TI_RAG document processing pipeline.

This script measures and reports the time taken to import each major dependency
used in the document processing pipeline, helping identify bottlenecks in startup time.

Usage:
    python profile_document_processing_imports.py
"""
import os
import sys
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Import the profiler
from src.utils.import_profiler import ImportProfiler

def profile_document_processing_imports():
    """Profile all imports used in the document processing pipeline."""
    # Create a timestamped log file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = project_root / f"logs/import_profile_{timestamp}.txt"
    os.makedirs(project_root / "logs", exist_ok=True)
    
    print(f"Profiling document processing imports. Results will be saved to {log_file}")
    
    # Initialize the profiler
    profiler = ImportProfiler(log_to_file=str(log_file))
    
    # Define module groups to profile
    module_groups = {
        "Basic Libraries": [
            "os",
            "sys",
            "time",
            "pathlib",
            "logging",
            "multiprocessing",
            "concurrent.futures",
            "re",
            "json",
            "pickle",
            "uuid",
            "gc",
        ],
        
        "ML and NLP Essentials": [
            "numpy",
            "torch",
            "transformers",
            "nltk",
        ],
        
        "Document Processing": [
            "PyPDF2",  
            "fitz",    # PyMuPDF
            "docx",
            "pandas",
            "PIL",
            "pytesseract",
        ],
        
        "Deep Learning Libraries": [
            "sentence_transformers",
            "flair.nn",
            "flair.data",
            "fastcoref",
        ],
        
        "LangChain and Embeddings": [
            "langchain",
            "langchain_huggingface",
            "langchain_experimental",
            "embed",  # infinity-embed
        ],
        
        "Vector Databases": [
            "qdrant_client",
            "rank_bm25",
        ],
    }
    
    # Profile each module group
    start_time = time.time()
    profiler.profile_nested_imports(module_groups)
    total_time = time.time() - start_time
    
    # Print overall summary
    profiler.print_summary()
    print(f"\nOverall profiling completed in {total_time:.2f} seconds")
    print(f"Detailed results saved to {log_file}")
    
    # Profile specific model loading functions for fine-grained profiling
    print("\n=== Profiling specific model initializations (this may take several minutes) ===")
    
    try:
        # Basic sentence transformers model
        profile_sentence_transformer()
        
        # HuggingFace embeddings
        profile_huggingface_embeddings()
        
        # FastCoref model
        profile_fastcoref()
        
        # Flair models
        profile_flair_models()
        
    except Exception as e:
        print(f"Error during detailed model profiling: {e}")


def profile_sentence_transformer():
    """Profile creating a SentenceTransformer instance."""
    try:
        start = time.time()
        print("\nProfiling SentenceTransformer initialization...")
        
        from sentence_transformers import SentenceTransformer
        create_start = time.time()
        print(f"  Import time: {create_start - start:.2f}s")
        
        model = SentenceTransformer('intfloat/multilingual-e5-small')
        create_time = time.time() - create_start
        print(f"  Model creation time: {create_time:.2f}s")
        
        # Test embedding
        test_start = time.time()
        _ = model.encode("This is a test sentence")
        test_time = time.time() - test_start
        print(f"  First embedding time: {test_time:.2f}s")
        
        total_time = time.time() - start
        print(f"  Total SentenceTransformer profile time: {total_time:.2f}s")
    except Exception as e:
        print(f"  Error profiling SentenceTransformer: {e}")


def profile_huggingface_embeddings():
    """Profile creating a HuggingFaceEmbeddings instance."""
    try:
        start = time.time()
        print("\nProfiling HuggingFaceEmbeddings initialization...")
        
        from langchain_huggingface import HuggingFaceEmbeddings
        import_time = time.time() - start
        print(f"  Import time: {import_time:.2f}s")
        
        create_start = time.time()
        model = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        create_time = time.time() - create_start
        print(f"  Model creation time: {create_time:.2f}s")
        
        # Test embedding
        test_start = time.time()
        _ = model.embed_query("This is a test sentence")
        test_time = time.time() - test_start
        print(f"  First embedding time: {test_time:.2f}s")
        
        total_time = time.time() - start
        print(f"  Total HuggingFaceEmbeddings profile time: {total_time:.2f}s")
    except Exception as e:
        print(f"  Error profiling HuggingFaceEmbeddings: {e}")


def profile_fastcoref():
    """Profile creating a FastCoref instance."""
    try:
        start = time.time()
        print("\nProfiling FastCoref initialization...")
        
        from fastcoref import FCoref
        import_time = time.time() - start
        print(f"  Import time: {import_time:.2f}s")
        
        create_start = time.time()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = FCoref(device=device)
        create_time = time.time() - create_start
        print(f"  Model creation time: {create_time:.2f}s")
        
        # Test prediction
        test_start = time.time()
        _ = model.predict(texts=["John met Bob. He said hello to him."])
        test_time = time.time() - test_start
        print(f"  First prediction time: {test_time:.2f}s")
        
        total_time = time.time() - start
        print(f"  Total FastCoref profile time: {total_time:.2f}s")
    except Exception as e:
        print(f"  Error profiling FastCoref: {e}")


def profile_flair_models():
    """Profile creating Flair NER and relation models."""
    try:
        start = time.time()
        print("\nProfiling Flair models initialization...")
        
        from flair.nn import Classifier
        from flair.data import Sentence
        import_time = time.time() - start
        print(f"  Import time: {import_time:.2f}s")
        
        # NER model
        ner_start = time.time()
        print("  Loading NER model...")
        ner_model = Classifier.load("flair/ner-english-ontonotes-fast")
        ner_time = time.time() - ner_start
        print(f"  NER model creation time: {ner_time:.2f}s")
        
        # Test NER prediction
        test_sentence = Sentence("George Washington went to Washington.")
        ner_pred_start = time.time()
        ner_model.predict(test_sentence)
        ner_pred_time = time.time() - ner_pred_start
        print(f"  First NER prediction time: {ner_pred_time:.2f}s")
        
        # Relation model
        try:
            rel_start = time.time()
            print("  Loading relation model...")
            rel_model = Classifier.load("relations")
            rel_time = time.time() - rel_start
            print(f"  Relation model creation time: {rel_time:.2f}s")
            
            # Test relation prediction
            rel_pred_start = time.time()
            rel_model.predict(test_sentence)
            rel_pred_time = time.time() - rel_pred_start
            print(f"  First relation prediction time: {rel_pred_time:.2f}s")
        except Exception as rel_error:
            print(f"  Error loading relation model: {rel_error}")
        
        total_time = time.time() - start
        print(f"  Total Flair models profile time: {total_time:.2f}s")
    except Exception as e:
        print(f"  Error profiling Flair models: {e}")


if __name__ == "__main__":
    # Import torch here to ensure it's available for the model initialization functions
    try:
        import torch
    except ImportError:
        print("Error: PyTorch (torch) is required but not installed.")
        sys.exit(1)
        
    profile_document_processing_imports()
