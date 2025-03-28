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
from src.processing.model_manager import ModelManager

def process_documents(file_paths: List[str], status_queue: multiprocessing.Queue, preload_models: bool = True) -> bool:
    """
    Main document processing function to be executed in a subprocess.
    Orchestrates the entire pipeline of document processing.
    
    Args:
        file_paths (list): List of paths to documents to process
        status_queue (Queue): Queue for status updates
        preload_models (bool, optional): Whether to preload all models at the beginning. Defaults to True.
        
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
        
        # Initialize variables for model management
        model_manager = None
        models = None
        
        # Step 0: Initialize model manager and load all models upfront
        if preload_models:
            status_queue.put(('status', "Step 0/6: Model Initialization"))
            status_queue.put(('progress', 0.0, "Loading all models"))
            
            print(f"[PIPELINE] ===== INITIALIZING MODEL MANAGER =====")
            logger.info("Initializing model manager and preloading all models")
            model_manager = ModelManager(status_queue=status_queue)
            
            # Progress monitoring variables
            model_types = ["embedding_model", "coref_model", "flair_ner_model", "flair_relation_model"]
            total_models = len(model_types)
            models_loaded = 0
            
            # Start loading models with progress updates
            model_load_start = time.time()
            
            # Load embedding model
            status_queue.put(('status', "Loading embedding model..."))
            model_manager.load_embedding_model()
            models_loaded += 1
            status_queue.put(('progress', models_loaded/total_models * 0.15, f"Loaded embedding model ({models_loaded}/{total_models})"))
            
            # Load coreference model
            status_queue.put(('status', "Loading coreference model..."))
            model_manager.load_coref_model()
            models_loaded += 1
            status_queue.put(('progress', models_loaded/total_models * 0.15, f"Loaded coreference model ({models_loaded}/{total_models})"))
            
            # Load NER model
            status_queue.put(('status', "Loading NER model..."))
            model_manager.load_flair_ner_model()
            models_loaded += 1
            status_queue.put(('progress', models_loaded/total_models * 0.15, f"Loaded NER model ({models_loaded}/{total_models})"))
            
            # Load relation model
            status_queue.put(('status', "Loading relation model..."))
            model_manager.load_flair_relation_model()
            models_loaded += 1
            status_queue.put(('progress', models_loaded/total_models * 0.15, f"Loaded relation model ({models_loaded}/{total_models})"))
            
            model_load_time = time.time() - model_load_start
            print(f"[PIPELINE] All models loaded in {model_load_time:.2f}s")
            logger.info(f"All models loaded in {model_load_time:.2f}s")
            status_queue.put(('status', f"All models loaded in {model_load_time:.2f}s"))
            
            # Pass models to components when initializing them
            models = model_manager.get_models()
            print(f"[PIPELINE] Model dictionary keys: {list(models.keys())}")
            logger.info(f"Model dictionary keys: {list(models.keys())}")
        
        # Step 1: Document Loading
        status_queue.put(('status', "Step 1/6: Document Loading"))
        # If we preloaded models, start at 15% progress, otherwise at 0%
        base_progress = 0.15 if preload_models else 0.0
        status_queue.put(('progress', base_progress, "Starting document loading"))
        
        # Initialize document loader
        loader = DocumentLoader(status_queue=status_queue)
        
        # Process each document
        documents = []
        for i, file_path in enumerate(file_paths):
            try:
                # Load document
                document = loader.load_document(file_path)
                documents.append(document)
                
                # Update progress - document loading takes 15% of total progress
                progress_per_doc = 0.15 / len(file_paths)
                progress = base_progress + (i + 1) * progress_per_doc
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
        status_queue.put(('status', "Step 2/6: Document Chunking"))
        # Document loading takes 15% of progress, so chunking starts at 15% + 15% = 30%
        chunking_base = base_progress + 0.15  # 30% if preloaded models, 15% otherwise
        status_queue.put(('progress', chunking_base, "Starting document chunking"))
        
        # Initialize document chunker - reuse loaded model if available
        chunking_start_time = time.time()
        logger.info(f"Initializing document chunker at {chunking_start_time:.2f}")
        print(f"[PIPELINE] ===== INITIALIZING DOCUMENT CHUNKER =====")
        print(f"[PIPELINE] Creating DocumentChunker instance at {time.time():.2f}")
        
        chunker_init_start = time.time()
        # Use semantic_chunking_model for the document chunker
        semantic_chunking_model = models.get("semantic_chunking_model") if models else None
        chunker = DocumentChunker(
            status_queue=status_queue,
            embedding_model=semantic_chunking_model
        )
        chunker_init_time = time.time() - chunker_init_start
        
        print(f"[PIPELINE] DocumentChunker instance created in {chunker_init_time:.2f}s")
        logger.info(f"DocumentChunker instance created in {chunker_init_time:.2f}s")
        
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
                
                # Update progress - chunking takes 15% of total progress
                progress_per_doc = 0.15 / len(documents)
                progress = chunking_base + (i + 1) * progress_per_doc
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
        
        # We don't shutdown the chunker models if we're using the model manager
        if not preload_models:
            shutdown_start = time.time()
            logger.info("Shutting down document chunker and unloading model...")
            print(f"[PIPELINE] ===== SHUTTING DOWN DOCUMENT CHUNKER =====")
            print(f"[PIPELINE] Starting chunker shutdown at {shutdown_start:.2f}")
            chunker.shutdown()
            shutdown_time = time.time() - shutdown_start
            logger.info(f"Document chunker shutdown in {shutdown_time:.2f}s")
            print(f"[PIPELINE] Document chunker shutdown in {shutdown_time:.2f}s")
            print(f"[PIPELINE] ===== DOCUMENT CHUNKER SHUTDOWN COMPLETE =====")
        
        # Transition status
        transition_start = time.time()
        logger.info(f"Preparing for coreference resolution at {transition_start:.2f}")
        print(f"[PIPELINE] ===== TRANSITION TO COREFERENCE RESOLUTION =====")
        print(f"[PIPELINE] Preparing for coreference resolution - initializing resolver")
        status_queue.put(('status', "Transitioning to coreference resolution..."))
        
        # Step 3: Coreference Resolution
        status_queue.put(('status', "Step 3/6: Coreference Resolution"))
        # Document loading (15%) + chunking (15%) = 30% or 45% with preloaded models
        coref_base = chunking_base + 0.15  # 45% if preloaded models, 30% otherwise
        status_queue.put(('progress', coref_base, "Starting coreference resolution"))
        logger.info(f"Transition time before coreference resolution: {time.time() - transition_start:.2f}s")
        print(f"[PIPELINE] Starting coreference resolution with {len(all_chunks)} chunks")
        
        # Initialize coreference resolver - reuse loaded model if available
        print(f"[PIPELINE] Creating CoreferenceResolver instance at {time.time():.2f}")
        logger.info(f"Creating CoreferenceResolver instance")
        resolver_init_start = time.time()
        
        coref_model = models.get("coref_model") if models else None
        resolver = CoreferenceResolver(
            status_queue=status_queue,
            coref_model=coref_model
        )
        
        resolver_init_time = time.time() - resolver_init_start
        print(f"[PIPELINE] CoreferenceResolver instance created in {resolver_init_time:.2f}s")
        logger.info(f"CoreferenceResolver instance created in {resolver_init_time:.2f}s")
        
        try:
            # Process all chunks with detailed timing and progress updates
            process_chunks_start = time.time()
            print(f"[PIPELINE] Starting coreference resolution process_chunks at {process_chunks_start:.2f}")
            logger.info(f"Starting coreference process_chunks")
            
            # Update progress at the start of processing
            status_queue.put(('progress', coref_base + 0.05, "Processing coreference resolution..."))
            
            # Call process_chunks with progress callback
            resolved_chunks = resolver.process_chunks(all_chunks)
            
            # Update progress at the end of processing
            process_chunks_time = time.time() - process_chunks_start
            status_queue.put(('progress', coref_base + 0.14, "Coreference resolution completed"))
            
            print(f"[PIPELINE] Coreference process_chunks completed in {process_chunks_time:.2f}s")
            logger.info(f"Coreference process_chunks completed in {process_chunks_time:.2f}s")
            
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
        status_queue.put(('status', "Step 4/6: Entity Extraction"))
        # Previous steps (30% or 45%) + coreference (15%) = 45% or 60%
        entity_base = coref_base + 0.15  # 60% if preloaded models, 45% otherwise
        status_queue.put(('progress', entity_base, "Starting entity extraction"))
        
        # Initialize entity extractor - reuse loaded models if available
        print(f"[PIPELINE] ===== INITIALIZING ENTITY EXTRACTOR =====")
        print(f"[PIPELINE] Creating EntityExtractor instance at {time.time():.2f}")
        extractor_init_start = time.time()
        
        # Get models if available
        ner_model = models.get("flair_ner_model") if models else None
        relation_model = models.get("flair_relation_model") if models else None
        
        extractor = EntityExtractor(
            status_queue=status_queue,
            ner_model=ner_model,
            relation_model=relation_model
        )
        
        extractor_init_time = time.time() - extractor_init_start
        print(f"[PIPELINE] EntityExtractor instance created in {extractor_init_time:.2f}s")
        logger.info(f"EntityExtractor instance created in {extractor_init_time:.2f}s")
        
        try:
            # Process all chunks with detailed timing and progress updates
            entity_start_time = time.time()
            print(f"[PIPELINE] ===== STARTING ENTITY EXTRACTION =====")
            print(f"[PIPELINE] Starting entity extraction process_chunks at {entity_start_time:.2f}")
            logger.info(f"Starting entity extraction process_chunks with {len(resolved_chunks)} chunks")
            
            # Update progress at the start of processing
            status_queue.put(('progress', entity_base + 0.05, "Processing entity extraction..."))
            
            # Call process_chunks
            processed_chunks, entities, relationships = extractor.process_chunks(resolved_chunks)
            
            # Update progress at the end of processing
            status_queue.put(('progress', entity_base + 0.14, "Entity extraction completed"))
            
            entity_time = time.time() - entity_start_time
            print(f"[PIPELINE] Entity extraction completed in {entity_time:.2f}s")
            logger.info(f"Entity extraction completed in {entity_time:.2f}s")
            print(f"[PIPELINE] ===== ENTITY EXTRACTION COMPLETE =====")
            
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
        status_queue.put(('status', "Step 5/6: Indexing"))
        # Previous steps (45% or 60%) + entity extraction (15%) = 60% or 75% 
        indexing_base = entity_base + 0.15  # 75% if preloaded models, 60% otherwise
        status_queue.put(('progress', indexing_base, "Starting indexing"))
        
        # Initialize indexer
        indexer = Indexer(status_queue=status_queue)
        
        try:
            # Index all chunks with progress updates
            indexing_start = time.time()
            status_queue.put(('progress', indexing_base + 0.05, "Indexing chunks..."))
            
            success = indexer.index_chunks(processed_chunks)
            
            indexing_time = time.time() - indexing_start
            status_queue.put(('progress', indexing_base + 0.14, "Indexing completed"))
            print(f"[PIPELINE] Indexing completed in {indexing_time:.2f}s")
            
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
        
        # Step 6: Cleanup
        if preload_models and model_manager is not None:
            status_queue.put(('status', "Step 6/6: Resource Cleanup"))
            status_queue.put(('progress', 1.0, "Unloading models and freeing resources"))
            
            print(f"[PIPELINE] ===== FINAL CLEANUP =====")
            model_manager.unload_all_models()
            print(f"[PIPELINE] ===== CLEANUP COMPLETE =====")
        
        return True
        
    except Exception as e:
        error_msg = f"Unexpected error in document processing: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        status_queue.put(('error', error_msg))
        
        # Make sure to unload models even on error
        if preload_models and 'model_manager' in locals() and model_manager is not None:
            try:
                model_manager.unload_all_models()
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {cleanup_error}")
                
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

def run_processing_pipeline(file_paths: List[str], preload_models: bool = True) -> Tuple[multiprocessing.Process, multiprocessing.Queue]:
    """
    Run the document processing pipeline in a dedicated subprocess.
    
    Args:
        file_paths (list): List of paths to documents to process
        preload_models (bool): Whether to preload all models at the start of processing
        
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
        args=(file_paths, status_queue, preload_models)
    )
    
    # Start the process
    process.start()
    
    return process, status_queue