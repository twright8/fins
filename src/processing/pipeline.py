"""
Document processing pipeline that orchestrates all processing steps.
Designed to run synchronously within a dedicated Python subprocess.
"""
import sys
import os
from pathlib import Path
import time
import multiprocessing
from typing import List, Dict, Any, Tuple, Optional
import traceback

# Add parent directory to sys.path to enable imports from project root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.core.config import CONFIG
from src.utils.logger import get_subprocess_logger
from src.utils.resource_monitor import log_memory_usage
from src.processing.document_loader import DocumentLoader
from src.processing.document_chunker import DocumentChunker
from src.processing.coreference_resolver import CoreferenceResolver
from src.processing.entity_extractor import EntityExtractor
from src.processing.indexer import Indexer

def process_documents(file_paths: List[str], status_queue: multiprocessing.Queue) -> bool:
    """
    Main document processing function to be executed in a subprocess.
    Orchestrates the entire pipeline of document processing.
    
    Args:
        file_paths (list): List of paths to documents to process
        status_queue (Queue): Queue for status updates
        
    Returns:
        bool: Success status
    """
    # Configure subprocess logger to use status queue
    logger = get_subprocess_logger()
    
    try:
        logger.info(f"Starting document processing for {len(file_paths)} files")
        status_queue.put(('status', f"Starting document processing for {len(file_paths)} files"))
        
        # Log memory usage at start
        log_memory_usage(logger)
        
        # Step 1: Document Loading
        status_queue.put(('status', "Step 1/5: Document Loading"))
        status_queue.put(('progress', 0.0, "Starting document loading"))
        
        # Initialize document loader
        loader = DocumentLoader(status_queue=status_queue)
        
        # Process each document
        documents = []
        for i, file_path in enumerate(file_paths):
            try:
                # Load document
                document = loader.load_document(file_path)
                documents.append(document)
                
                # Update progress
                progress = (i + 1) / len(file_paths) * 0.2  # 0-20% progress
                status_queue.put(('progress', progress, f"Loaded {i+1}/{len(file_paths)} documents"))
                
            except Exception as e:
                error_msg = f"Error loading document {file_path}: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                status_queue.put(('error', error_msg))
                return False
        
        logger.info(f"Loaded {len(documents)} documents")
        status_queue.put(('status', f"Successfully loaded {len(documents)} documents"))
        log_memory_usage(logger)
        
        # Step 2: Document Chunking
        status_queue.put(('status', "Step 2/5: Document Chunking"))
        status_queue.put(('progress', 0.2, "Starting document chunking"))
        
        # Initialize document chunker
        chunking_start_time = time.time()
        logger.info(f"Initializing document chunker at {chunking_start_time:.2f}")
        print(f"[PIPELINE] Initializing document chunker")
        chunker = DocumentChunker(status_queue=status_queue)
        
        # Process each document
        all_chunks = []
        for i, document in enumerate(documents):
            try:
                # Chunk document
                doc_start_time = time.time()
                file_name = document.get('file_name', 'unknown')
                logger.info(f"Chunking document {i+1}/{len(documents)}: {file_name}")
                print(f"[PIPELINE] Chunking document {i+1}/{len(documents)}: {file_name}")
                
                chunks = chunker.chunk_document(document)
                all_chunks.extend(chunks)
                
                # Log chunking time
                doc_elapsed = time.time() - doc_start_time
                logger.info(f"Document {file_name} chunked in {doc_elapsed:.2f}s, produced {len(chunks)} chunks")
                print(f"[PIPELINE] Document {file_name} chunked in {doc_elapsed:.2f}s, produced {len(chunks)} chunks")
                
                # Update progress
                progress = 0.2 + (i + 1) / len(documents) * 0.2  # 20-40% progress
                status_queue.put(('progress', progress, f"Chunked {i+1}/{len(documents)} documents"))
                
            except Exception as e:
                error_msg = f"Error chunking document {document.get('file_name', '')}: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                status_queue.put(('error', error_msg))
                return False
        
        # Log overall chunking stats
        chunking_elapsed = time.time() - chunking_start_time
        logger.info(f"Chunking complete in {chunking_elapsed:.2f}s. Created {len(all_chunks)} chunks from {len(documents)} documents")
        print(f"[PIPELINE] Chunking complete in {chunking_elapsed:.2f}s. Created {len(all_chunks)} chunks from {len(documents)} documents")
        status_queue.put(('status', f"Created {len(all_chunks)} chunks"))
        log_memory_usage(logger)
        
        # Transition status
        transition_start = time.time()
        logger.info(f"Preparing for coreference resolution at {transition_start:.2f}")
        print(f"[PIPELINE] Preparing for coreference resolution - initializing resolver")
        status_queue.put(('status', "Transitioning to coreference resolution..."))
        
        # Step 3: Coreference Resolution
        status_queue.put(('status', "Step 3/5: Coreference Resolution"))
        status_queue.put(('progress', 0.4, "Starting coreference resolution"))
        logger.info(f"Transition time before coreference resolution: {time.time() - transition_start:.2f}s")
        print(f"[PIPELINE] Starting coreference resolution with {len(all_chunks)} chunks")
        
        # Initialize coreference resolver
        resolver = CoreferenceResolver(status_queue=status_queue)
        
        try:
            # Process all chunks
            resolved_chunks = resolver.process_chunks(all_chunks)
            
            logger.info(f"Applied coreference resolution to {len(resolved_chunks)} chunks")
            status_queue.put(('status', f"Applied coreference resolution"))
            log_memory_usage(logger)
            
        except Exception as e:
            error_msg = f"Error in coreference resolution: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            status_queue.put(('error', error_msg))
            return False
        
        # Step 4: Entity Extraction
        status_queue.put(('status', "Step 4/5: Entity Extraction"))
        status_queue.put(('progress', 0.6, "Starting entity extraction"))
        
        # Initialize entity extractor
        extractor = EntityExtractor(status_queue=status_queue)
        
        try:
            # Process all chunks
            processed_chunks, entities, relationships = extractor.process_chunks(resolved_chunks)
            
            logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships")
            status_queue.put(('status', f"Extracted {len(entities)} entities and {len(relationships)} relationships"))
            log_memory_usage(logger)
            
            # Store entities and relationships
            save_results(entities, relationships)
            
        except Exception as e:
            error_msg = f"Error in entity extraction: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            status_queue.put(('error', error_msg))
            return False
        
        # Step 5: Indexing
        status_queue.put(('status', "Step 5/5: Indexing"))
        status_queue.put(('progress', 0.8, "Starting indexing"))
        
        # Initialize indexer
        indexer = Indexer(status_queue=status_queue)
        
        try:
            # Index all chunks
            success = indexer.index_chunks(processed_chunks)
            
            if success:
                logger.info("Indexing complete")
                status_queue.put(('status', "Indexing complete"))
                log_memory_usage(logger)
            else:
                error_msg = "Indexing failed"
                logger.error(error_msg)
                status_queue.put(('error', error_msg))
                return False
            
        except Exception as e:
            error_msg = f"Error in indexing: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            status_queue.put(('error', error_msg))
            return False
        
        # Final success message
        status_queue.put(('progress', 1.0, "Processing complete"))
        status_queue.put(('success', "Document processing completed successfully"))
        
        return True
        
    except Exception as e:
        error_msg = f"Unexpected error in document processing: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        status_queue.put(('error', error_msg))
        return False

def save_results(entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> None:
    """
    Save extracted entities and relationships to disk.
    
    Args:
        entities (list): List of entity dictionaries
        relationships (list): List of relationship dictionaries
    """
    import json
    
    base_dir = Path(__file__).resolve().parent.parent.parent
    results_dir = base_dir / "data" / "extracted"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save entities
    with open(results_dir / "entities.json", 'w') as f:
        json.dump(entities, f, indent=2)
    
    # Save relationships
    with open(results_dir / "relationships.json", 'w') as f:
        json.dump(relationships, f, indent=2)

def run_processing_pipeline(file_paths: List[str]) -> Tuple[multiprocessing.Process, multiprocessing.Queue]:
    """
    Run the document processing pipeline in a dedicated subprocess.
    
    Args:
        file_paths (list): List of paths to documents to process
        
    Returns:
        tuple: (process, status_queue)
            - process: The subprocess Process object
            - status_queue: Queue for receiving status updates
    """
    # Ensure CUDA uses the right start method
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, continue
        pass
    
    # Create status queue for IPC
    status_queue = multiprocessing.Queue()
    
    # Create and start subprocess
    process = multiprocessing.Process(
        target=process_documents,
        args=(file_paths, status_queue)
    )
    
    # Start the process
    process.start()
    
    return process, status_queue
