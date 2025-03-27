#!/usr/bin/env python3
"""
Main entry point for the Anti-Corruption RAG system.
This script can be used to process documents directly from the command line.
"""
import argparse
import json
import os
import sys
import time
import multiprocessing
from pathlib import Path
from typing import List, Dict, Any

# Ensure the project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.core.config import CONFIG, DATA_DIR
from src.processing.pipeline import run_processing_pipeline
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def process_documents_cli(file_paths: List[str], output_path: str = None):
    """
    Process documents from the command line.
    
    Args:
        file_paths (list): List of paths to documents to process
        output_path (str, optional): Path to save processing results
    """
    # Check if files exist
    for file_path in file_paths:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return
    
    logger.info(f"Processing {len(file_paths)} documents")
    
    # Create status queue listener
    status_queue = multiprocessing.Queue()
    
    # Start monitor thread for the status queue
    monitor_thread = multiprocessing.Process(
        target=_monitor_status_queue,
        args=(status_queue,)
    )
    monitor_thread.start()
    
    # Run processing pipeline
    process, _ = run_processing_pipeline(file_paths)
    
    # Wait for process to complete
    process.join()
    
    # Terminate monitor thread
    status_queue.put(('end', None))
    monitor_thread.join(timeout=1)
    
    # Check process exit code
    if process.exitcode == 0:
        logger.info("Document processing completed successfully")
        
        # Save results to output file if specified
        if output_path:
            try:
                _save_processing_results(output_path)
                logger.info(f"Results saved to {output_path}")
            except Exception as e:
                logger.error(f"Error saving results: {e}")
    else:
        logger.error(f"Document processing failed with exit code {process.exitcode}")

def _monitor_status_queue(status_queue):
    """
    Monitor the status queue and print updates.
    
    Args:
        status_queue: Queue for status updates
    """
    while True:
        try:
            msg = status_queue.get()
            
            if msg[0] == 'end':
                break
            
            if msg[0] == 'progress':
                progress, status_text = msg[1], msg[2]
                print(f"Progress: {progress:.1%} - {status_text}")
            
            elif msg[0] == 'status':
                print(f"Status: {msg[1]}")
            
            elif msg[0] == 'error':
                print(f"Error: {msg[1]}")
            
            elif msg[0] == 'success':
                print(f"Success: {msg[1]}")
        
        except (EOFError, KeyboardInterrupt):
            break
        
        except Exception as e:
            print(f"Error monitoring queue: {e}")
            break

def _save_processing_results(output_path: str):
    """
    Save processing results to a JSON file.
    
    Args:
        output_path (str): Path to save results
    """
    # Collect results from data directory
    results = {}
    
    # Entities
    entities_file = DATA_DIR / "extracted" / "entities.json"
    if entities_file.exists():
        with open(entities_file, 'r') as f:
            results['entities'] = json.load(f)
    
    # Relationships
    relationships_file = DATA_DIR / "extracted" / "relationships.json"
    if relationships_file.exists():
        with open(relationships_file, 'r') as f:
            results['relationships'] = json.load(f)
    
    # BM25 status
    bm25_file = DATA_DIR / "bm25_indices" / "latest_index.pkl"
    results['bm25_indexed'] = bm25_file.exists()
    
    # Qdrant status
    results['vector_indexed'] = False
    try:
        from qdrant_client import QdrantClient
        
        client = QdrantClient(
            host=CONFIG["qdrant"]["host"],
            port=CONFIG["qdrant"]["port"]
        )
        
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if CONFIG["qdrant"]["collection_name"] in collection_names:
            collection_info = client.get_collection(CONFIG["qdrant"]["collection_name"])
            results['vector_indexed'] = True
            results['vector_count'] = collection_info.points_count
    except Exception as e:
        results['vector_error'] = str(e)
    
    # Save to output file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def initialize_directories():
    """
    Initialize all necessary directories for the system.
    """
    directories = [
        DATA_DIR,
        DATA_DIR / "bm25_indices",
        DATA_DIR / "extracted",
        DATA_DIR / "ocr_cache",
        DATA_DIR / "qdrant_data",
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "temp"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Initialized directory: {directory}")

def main():
    """Main entry point."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Anti-Corruption RAG Document Processing")
    parser.add_argument("files", nargs="+", help="Paths to documents to process")
    parser.add_argument("-o", "--output", help="Path to save processing results")
    parser.add_argument("--init", action="store_true", help="Initialize directories only")
    
    args = parser.parse_args()
    
    # Initialize directories
    initialize_directories()
    
    if args.init:
        logger.info("Directories initialized. Exiting.")
        return
    
    # Process documents
    process_documents_cli(args.files, args.output)

if __name__ == "__main__":
    # Ensure clean CUDA initialization for multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, continue
        pass
    
    main()
