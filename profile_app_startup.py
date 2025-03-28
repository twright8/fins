#!/usr/bin/env python3
"""
Script to profile the startup time of the TI_RAG application.

This script demonstrates how to integrate profiling into the application
startup process, measuring both import times and the time taken to initialize
various components of the system.

Usage:
    python profile_app_startup.py
"""
import os
import sys
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Import the profiler
from src.utils.app_profiler import AppProfiler

# Create a profiler instance
profiler = AppProfiler(
    enabled=True, 
    log_to_console=True,
    profile_folder='logs/startup_profiles'
)

# Profile key imports first
profiler.profile_imports([
    # Core dependencies
    "os", "sys", "time", "pathlib", "multiprocessing", "concurrent.futures",
    
    # Document processing
    "fitz", "docx", "pandas", "PIL", "pytesseract",
    
    # ML/NLP 
    "torch", "transformers", "nltk",
    
    # Key libraries for our system
    "langchain_huggingface", "flair.nn", "fastcoref", "qdrant_client"
])

# Use decorators to profile function execution times
@profiler.profile_function
def initialize_system():
    """Initialize the main system components."""
    print("\nInitializing system components...")
    
    # Initialize modules one by one to see where time is spent
    initialize_config()
    initialize_logging()
    initialize_model_manager()
    initialize_document_loader()
    initialize_pipeline()
    
    # Finish
    print("System initialized successfully")

@profiler.profile_function
def initialize_config():
    """Initialize configuration."""
    print("Loading configuration...")
    from src.core.config import CONFIG
    time.sleep(0.1)  # Simulate some work
    print(f"Configuration loaded with {len(CONFIG)} sections")

@profiler.profile_function    
def initialize_logging():
    """Initialize logging system."""
    print("Setting up logging...")
    from src.utils.logger import setup_logger
    logger = setup_logger("startup_profiling")
    time.sleep(0.1)  # Simulate some work
    logger.info("Logging system initialized")

@profiler.profile_function
def initialize_model_manager():
    """Initialize the model manager (slow due to ML model loading)."""
    print("Initializing model manager...")
    try:
        from src.processing.model_manager import ModelManager
        manager = ModelManager()
        # Note: We're not actually loading models here to keep this script fast
        print("Model manager initialized (without loading models)")
    except Exception as e:
        print(f"Error initializing model manager: {e}")

@profiler.profile_function
def initialize_document_loader():
    """Initialize document loader."""
    print("Initializing document loader...")
    from src.processing.document_loader import DocumentLoader
    loader = DocumentLoader()
    print("Document loader initialized")

@profiler.profile_function
def initialize_pipeline():
    """Initialize document processing pipeline."""
    print("Initializing document processing pipeline...")
    # We don't actually initialize the pipeline to avoid heavy loading
    # But in a real app, you'd do:
    # from src.processing.pipeline import process_documents
    time.sleep(0.2)  # Simulate pipeline initialization
    print("Document processing pipeline initialized")

def main():
    """Main function to demonstrate profiled startup."""
    print("Starting application with profiling...")
    start_time = time.time()
    
    # Initialize the system
    initialize_system()
    
    # Calculate and report total startup time
    total_time = time.time() - start_time
    print(f"\nApplication startup completed in {total_time:.2f} seconds")
    
    # Print function timing statistics
    profiler.print_function_stats()

if __name__ == "__main__":
    main()
