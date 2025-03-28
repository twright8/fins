"""
Coreference resolution module that replaces pronouns with their referred entities.
Uses the FastCoref model for efficient batch processing.
"""
import sys
import re
from pathlib import Path
import torch
import gc
from typing import List, Dict, Any

# Add parent directory to sys.path to enable imports from project root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.logger import setup_logger
from src.utils.resource_monitor import log_memory_usage

logger = setup_logger(__name__)

class CoreferenceResolver:
    """
    Coreference resolution using the FastCoref model.
    Replaces pronouns with their referred entities.
    """
    
    def __init__(self, status_queue=None, coref_model=None):
        """
        Initialize coreference resolver.
        
        Args:
            status_queue (Queue, optional): Queue for status updates.
            coref_model: A pre-loaded coref model to use instead of loading a new one.
        """
        self.status_queue = status_queue
        self.model = coref_model
        
        logger.info("Initializing CoreferenceResolver with FastCoref")
        print(f"[COREF] Initializing with {'pre-loaded' if coref_model else 'no'} FastCoref model")
        
        if self.status_queue:
            self.status_queue.put(('status', 'Coreference resolver initialized'))
        
        log_memory_usage(logger)
    
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
    
    def load_model(self):
        """
        Load the FastCoref model if not already loaded.
        """
        if self.model is not None:
            logger.info("FastCoref model already loaded, skipping load")
            print(f"[COREF] FastCoref model already loaded, skipping load")
            return
        
        print(f"[COREF] ===== STARTING MODEL LOAD =====")
        logger.info("===== STARTING MODEL LOAD =====")
        
        try:
            import time
            start_time = time.time()
            
            # This statement helps see when we start loading vs when the import happens
            logger.info(f"Starting to load FastCoref model at time {start_time:.2f}")
            print(f"[COREF] Starting to load FastCoref model...")
            self._update_status("Preparing to load FastCoref model")
            
            # Log before import to help debug potential import issues
            logger.info("Importing FastCoref library...")
            print(f"[COREF] Importing FastCoref library")
            
            from fastcoref import FCoref
            
            # Log after import success
            import_time = time.time() - start_time
            logger.info(f"FastCoref imported successfully in {import_time:.2f}s")
            print(f"[COREF] FastCoref library imported in {import_time:.2f}s")
            
            # Update status with clear indication that we're initializing the model
            self._update_status("Initializing FastCoref model (this may take some time)")
            print(f"[COREF] Initializing FastCoref model and loading weights...")
            
            # Log before actual model instantiation
            model_start = time.time()
            logger.info("Creating FastCoref model instance...")
            
            # Try to initialize on GPU first, fallback to CPU if it fails
            try:
                # Try initializing on GPU first
                gpu_init_start = time.time()
                print(f"[COREF] Initializing FastCoref on GPU at {gpu_init_start:.2f}...")
                logger.info("Initializing FastCoref on GPU...")
                
                self.model = FCoref(device='cuda:0')
                
                gpu_init_time = time.time() - gpu_init_start
                logger.info(f"FastCoref model loaded on GPU (CUDA) in {gpu_init_time:.2f}s")
                print(f"[COREF] FastCoref model loaded on GPU (CUDA) in {gpu_init_time:.2f}s")
            except Exception as gpu_error:
                # Fallback to CPU if GPU initialization fails
                logger.warning(f"GPU initialization failed: {gpu_error}, falling back to CPU")
                print(f"[COREF] GPU initialization failed, falling back to CPU")
                
                cpu_init_start = time.time()
                print(f"[COREF] Initializing FastCoref on CPU at {cpu_init_start:.2f}...")
                logger.info("Initializing FastCoref on CPU...")
                
                self.model = FCoref(device='cpu')
                
                cpu_init_time = time.time() - cpu_init_start
                logger.info(f"FastCoref model loaded on CPU in {cpu_init_time:.2f}s")
                print(f"[COREF] FastCoref model loaded on CPU in {cpu_init_time:.2f}s")
            
            # Log success and timing
            model_time = time.time() - model_start
            total_time = time.time() - start_time
            logger.info(f"FastCoref model created in {model_time:.2f}s (total time: {total_time:.2f}s)")
            print(f"[COREF] FastCoref model loaded successfully in {total_time:.2f}s")
            
            self._update_status(f"FastCoref model loaded successfully in {total_time:.2f}s")
            log_memory_usage(logger)
        except Exception as e:
            error_msg = f"Error loading FastCoref model: {e}"
            self._update_status(error_msg)
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            print(f"[COREF ERROR] Failed to load FastCoref model: {e}")
            print(traceback.format_exc())
            raise
    
    def shutdown(self):
        """
        Unload model to free up memory.
        """
        if self.model is not None:
            print(f"[COREF] ===== STARTING MODEL UNLOAD =====")
            logger.info("===== STARTING MODEL UNLOAD =====")
            self._update_status("Unloading FastCoref model")
            
            # Delete the model reference
            print(f"[COREF] Deleting model reference...")
            logger.info("Deleting model reference...")
            del self.model
            self.model = None
            
            # Clean up CUDA memory
            if torch.cuda.is_available():
                print(f"[COREF] Clearing CUDA cache...")
                logger.info("Clearing CUDA cache...")
                torch.cuda.empty_cache()
            
            # Force garbage collection
            print(f"[COREF] Running garbage collection...")
            logger.info("Running garbage collection...")
            gc.collect()
            
            print(f"[COREF] ===== MODEL UNLOAD COMPLETE =====")
            logger.info("===== MODEL UNLOAD COMPLETE =====")
            
            log_memory_usage(logger)
    
    def replace_coreferences_fastcoref(self, text, clusters_char_offsets):
        """
        Replaces coreferent mentions in a text string based on character offsets.

        Args:
            text (str): The original text.
            clusters_char_offsets (list): A list of clusters, where each cluster is a
                                        list of (start_char, end_char) tuples
                                        representing mentions.

        Returns:
            str: The text with coreferences replaced.
        """
        replacements = []

        for cluster in clusters_char_offsets:
            if not cluster or len(cluster) < 2:
                continue

            antecedent_start, antecedent_end = cluster[0]
            antecedent_text = text[antecedent_start:antecedent_end]

            for mention_start, mention_end in cluster[1:]:
                original_mention_text = text[mention_start:mention_end]
                replacement_text = ""
                mention_lower = original_mention_text.lower()

                # --- Special Case Handling ---
                if "'" in mention_lower: # Contractions
                    parts = mention_lower.split("'")
                    if len(parts) == 2:
                        pronoun, ending = parts
                        if pronoun in ["he", "she", "it", "they", "we", "you"] and ending in ["s", "re", "ve", "ll", "d"]:
                            verb = ""
                            if ending == "s": verb = "is"
                            elif ending == "re": verb = "are"
                            elif ending == "ve": verb = "have"
                            elif ending == "ll": verb = "will"
                            elif ending == "d": verb = "would"
                            if verb:
                                replacement_text = f"{antecedent_text} {verb} (#{original_mention_text}#)"

                elif mention_lower in ["his", "her", "hers", "its", "their", "theirs", "our", "ours", "my", "mine", "your", "yours"]: # Possessives
                     replacement_text = f"{antecedent_text}'s (#{original_mention_text}#)"

                # --- Default case ---
                if not replacement_text:
                     replacement_text = f"{antecedent_text} (#{original_mention_text}#)"

                replacements.append({
                    "start": mention_start,
                    "end": mention_end,
                    "text": replacement_text
                })

        # Sort replacements by start index in descending order
        replacements.sort(key=lambda x: x['start'], reverse=True)

        # Apply replacements to the text
        modified_text = text
        for rep in replacements:
            modified_text = modified_text[:rep['start']] + rep['text'] + modified_text[rep['end']:]

        # Clean up spacing around punctuation
        modified_text = re.sub(r'\s+([.,?!])', r'\1', modified_text)
        modified_text = re.sub(r'\(\s*#', '(#', modified_text)
        modified_text = re.sub(r'#\s*\)', '#)', modified_text)

        return modified_text
    
    def process_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 16) -> List[Dict[str, Any]]:
        """
        Process a list of document chunks, applying coreference resolution to each in batches.
        
        Args:
            chunks (list): List of chunk dictionaries
            batch_size (int): Size of batches for processing
            
        Returns:
            list: List of processed chunk dictionaries with coreference resolution
        """
        import time
        start_time = time.time()
        
        self._update_status(f"Processing {len(chunks)} chunks for coreference resolution in batches of {batch_size}")
        logger.info(f"Starting batched coreference resolution for {len(chunks)} chunks at {start_time:.2f}")
        print(f"[COREF] Starting batched coreference resolution for {len(chunks)} chunks")
        
        try:
            # Load the model if not already loaded
            if self.model is None:
                model_start = time.time()
                logger.info("Loading FastCoref model...")
                print(f"[COREF] Loading FastCoref model...")
                self.load_model()
                model_time = time.time() - model_start
                logger.info(f"FastCoref model loaded in {model_time:.2f}s")
                print(f"[COREF] FastCoref model loaded in {model_time:.2f}s")
            else:
                logger.info("Using pre-loaded FastCoref model")
                print(f"[COREF] Using pre-loaded FastCoref model")
            
            # Create a map to store resolved texts
            all_resolved_texts = {}
            num_batches = (len(chunks) + batch_size - 1) // batch_size
            
            # Performance tracking
            total_text_len = sum(len(chunk.get('text', '')) for chunk in chunks)
            total_tokens = 0
            batch_times = []
            
            # Process chunks in batches
            for i in range(0, len(chunks), batch_size):
                batch_start = time.time()
                current_batch_num = (i // batch_size) + 1
                
                # Get current batch of chunks
                batch_chunks = chunks[i:i + batch_size]
                
                # Extract texts and chunk IDs
                batch_texts_raw = [chunk.get('text', '') for chunk in batch_chunks]
                batch_chunk_ids_raw = [chunk.get('chunk_id') for chunk in batch_chunks]
                
                # Filter out empty/whitespace-only texts
                valid_indices = [idx for idx, txt in enumerate(batch_texts_raw) if txt and txt.strip()]
                batch_texts_valid = [batch_texts_raw[idx] for idx in valid_indices]
                batch_chunk_ids_valid = [batch_chunk_ids_raw[idx] for idx in valid_indices]
                
                # Handle empty batches
                if not batch_texts_valid:
                    logger.warning(f"Skipping empty batch {current_batch_num}/{num_batches}")
                    for chunk in batch_chunks:  # Ensure all chunks are included in the results
                        all_resolved_texts[chunk.get('chunk_id')] = chunk.get('text', '')
                    continue
                
                # Update progress
                progress = (i + len(batch_chunks)) / len(chunks)
                self._update_status(f"Applying coreference resolution: batch {current_batch_num}/{num_batches}", progress)
                
                # Log batch information
                tokens_in_batch = sum(len(text.split()) for text in batch_texts_valid)
                total_tokens += tokens_in_batch
                logger.info(f"Processing batch {current_batch_num}/{num_batches} with {len(batch_texts_valid)} valid chunks (~{tokens_in_batch} tokens)")
                
                try:
                    # Process the batch with FastCoref
                    batch_resolution_start = time.time()
                    print(f"[COREF] Starting prediction for batch {current_batch_num} at {batch_resolution_start:.2f}...")
                    logger.info(f"Starting prediction for batch {current_batch_num} with {len(batch_texts_valid)} texts...")
                    
                    # Get predictions for the current batch
                    prediction_start = time.time()
                    batch_preds = self.model.predict(texts=batch_texts_valid)
                    prediction_time = time.time() - prediction_start
                    print(f"[COREF] FastCoref prediction completed in {prediction_time:.2f}s")
                    logger.info(f"FastCoref prediction completed in {prediction_time:.2f}s")
                    
                    # Process results within the batch
                    processing_start = time.time()
                    print(f"[COREF] Processing prediction results at {processing_start:.2f}...")
                    logger.info("Processing prediction results...")
                    for j, pred_result in enumerate(batch_preds):
                        chunk_id = batch_chunk_ids_valid[j]
                        original_text = batch_texts_valid[j]
                        
                        # Get clusters as character offsets
                        clusters = pred_result.get_clusters(as_strings=False)
                        
                        # Apply coreference resolution
                        resolved_text = self.replace_coreferences_fastcoref(original_text, clusters)
                        
                        # Store the result
                        all_resolved_texts[chunk_id] = resolved_text
                    
                    processing_time = time.time() - processing_start
                    batch_resolution_time = time.time() - batch_resolution_start
                    logger.info(f"Results processing completed in {processing_time:.2f}s")
                    logger.info(f"Batch processed in {batch_resolution_time:.2f}s ({batch_resolution_time/len(batch_texts_valid):.4f}s per chunk)")
                    print(f"[COREF] Results processing completed in {processing_time:.2f}s")
                    print(f"[COREF] Batch processed in {batch_resolution_time:.2f}s ({batch_resolution_time/len(batch_texts_valid):.4f}s per chunk)")
                    
                    # Make sure any skipped chunks (empty or invalid) are included with original text
                    for idx, chunk in enumerate(batch_chunks):
                        chunk_id = chunk.get('chunk_id')
                        if chunk_id not in all_resolved_texts:
                            all_resolved_texts[chunk_id] = chunk.get('text', '')
                    
                except Exception as e:
                    logger.error(f"Error processing batch {current_batch_num}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    
                    # For any failed batch, use original texts
                    for chunk in batch_chunks:
                        all_resolved_texts[chunk.get('chunk_id')] = chunk.get('text', '')
                
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                logger.info(f"Batch {current_batch_num}/{num_batches} completed in {batch_time:.2f}s")
                print(f"[COREF] Batch {current_batch_num}/{num_batches} completed in {batch_time:.2f}s")
            
            # Create processed chunks using the resolved texts
            processed_chunks = []
            for chunk in chunks:
                chunk_id = chunk.get('chunk_id')
                resolved_text = all_resolved_texts.get(chunk_id, chunk.get('text', ''))
                
                resolved_chunk = chunk.copy()
                resolved_chunk['text'] = resolved_text
                resolved_chunk['coref_applied'] = True
                
                processed_chunks.append(resolved_chunk)
            
            # Log final statistics
            total_time = time.time() - start_time
            avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
            
            completion_msg = (
                f"Coreference resolution complete for {len(chunks)} chunks in {total_time:.2f}s. "
                f"Processed in {len(batch_times)} batches, avg batch time: {avg_batch_time:.2f}s. "
                f"Processed {total_text_len} chars, ~{total_tokens} tokens."
            )
            
            logger.info(completion_msg)
            print(f"[COREF] {completion_msg}")
            self._update_status(f"Coreference resolution complete for {len(chunks)} chunks")
            
            return processed_chunks
            
        except Exception as e:
            error_msg = f"Error in coreference resolution: {e}"
            self._update_status(error_msg)
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            print(f"[COREF ERROR] {error_msg}")
            print(traceback.format_exc())
            raise
        finally:
            # Ensure model is unloaded even if there's an error
            unload_start = time.time()
            logger.info("Unloading FastCoref model...")
            print(f"[COREF] Unloading FastCoref model...")
            self.shutdown()
            logger.info(f"FastCoref model unloaded in {time.time() - unload_start:.2f}s")
            print(f"[COREF] FastCoref model unloaded in {time.time() - unload_start:.2f}s")
