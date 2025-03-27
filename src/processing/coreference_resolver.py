"""
Coreference resolution module that replaces pronouns with their referred entities.
Uses the Maverick model.
"""
import sys
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
    Coreference resolution using the Maverick model.
    Replaces pronouns with their referred entities.
    """
    
    def __init__(self, status_queue=None):
        """
        Initialize coreference resolver.
        
        Args:
            status_queue (Queue, optional): Queue for status updates.
        """
        self.status_queue = status_queue
        self.model = None
        
        logger.info("Initializing CoreferenceResolver")
        
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
    
    def _load_model(self):
        """
        Load the Maverick model if not already loaded.
        """
        if self.model is not None:
            return
        
        try:
            import time
            start_time = time.time()
            
            # This statement helps see when we start loading vs when the import happens
            logger.info(f"Starting to load Maverick coreference model at time {start_time:.2f}")
            print(f"[COREF] Starting to load Maverick coreference model...")
            self._update_status("Preparing to load Maverick coreference model")
            
            # Log before import to help debug potential import issues
            logger.info("Importing Maverick library...")
            print(f"[COREF] Importing Maverick library")
            
            from maverick import Maverick
            
            # Log after import success
            import_time = time.time() - start_time
            logger.info(f"Maverick imported successfully in {import_time:.2f}s")
            print(f"[COREF] Maverick library imported in {import_time:.2f}s")
            
            # Update status with clear indication that we're initializing the model
            self._update_status("Initializing Maverick coreference model (this may take some time)")
            print(f"[COREF] Initializing Maverick model and loading weights...")
            
            # Log before actual model instantiation
            model_start = time.time()
            logger.info("Creating Maverick model instance...")
            
            # Actually create the model
            self.model = Maverick()
            
            # Log success and timing
            model_time = time.time() - model_start
            total_time = time.time() - start_time
            logger.info(f"Maverick model created in {model_time:.2f}s (total time: {total_time:.2f}s)")
            print(f"[COREF] Maverick model loaded successfully in {total_time:.2f}s")
            
            self._update_status(f"Maverick model loaded successfully in {total_time:.2f}s")
            log_memory_usage(logger)
        except Exception as e:
            error_msg = f"Error loading Maverick model: {e}"
            self._update_status(error_msg)
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            print(f"[COREF ERROR] Failed to load Maverick model: {e}")
            print(traceback.format_exc())
            raise
    
    def _unload_model(self):
        """
        Unload model to free up memory.
        """
        if self.model is not None:
            self._update_status("Unloading Maverick model")
            
            # Delete the model reference
            del self.model
            self.model = None
            
            # Clean up CUDA memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            log_memory_usage(logger)
    
    def process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a list of document chunks, applying coreference resolution to each.
        
        Args:
            chunks (list): List of chunk dictionaries
            
        Returns:
            list: List of processed chunk dictionaries with coreference resolution
        """
        import time
        start_time = time.time()
        
        self._update_status(f"Processing {len(chunks)} chunks for coreference resolution")
        logger.info(f"Starting coreference resolution for {len(chunks)} chunks at {start_time:.2f}")
        print(f"[COREF] Starting coreference resolution for {len(chunks)} chunks")
        
        try:
            # Load the model once at the beginning of processing
            model_start = time.time()
            logger.info("Loading Maverick model...")
            print(f"[COREF] Loading Maverick model (may take some time)...")
            self._load_model()
            model_time = time.time() - model_start
            logger.info(f"Maverick model loaded in {model_time:.2f}s")
            print(f"[COREF] Maverick model loaded in {model_time:.2f}s")
            
            # Create a copy of the chunks to avoid modifying the originals
            processed_chunks = []
            
            # Performance tracking
            total_text_len = sum(len(chunk.get('text', '')) for chunk in chunks)
            total_tokens = 0
            total_resolved_len = 0
            resolution_times = []
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                chunk_start = time.time()
                progress = ((i + 1) / len(chunks))
                
                if i % 10 == 0 or i == len(chunks) - 1:  # Log every 10 chunks or the last one
                    logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                    print(f"[COREF] Processing chunk {i+1}/{len(chunks)}")
                
                self._update_status(f"Applying coreference resolution: chunk {i+1}/{len(chunks)}", progress)
                
                # Get the text content
                text = chunk.get('text', '')
                if not text.strip():
                    # Skip empty chunks
                    processed_chunks.append(chunk)
                    continue
                
                # Track text length for analysis
                text_len = len(text)
                
                # Apply coreference resolution
                resolution_start = time.time()
                resolved_text = self._apply_coreference_resolution(text)
                resolution_time = time.time() - resolution_start
                resolution_times.append(resolution_time)
                
                # Track resolved text length
                resolved_len = len(resolved_text)
                total_resolved_len += resolved_len
                
                # Estimate token count (rough approximation)
                tokens = len(text.split())
                total_tokens += tokens
                
                # Performance metrics for this chunk
                chunk_time = time.time() - chunk_start
                if i % 10 == 0 or chunk_time > 1.0:  # Log every 10th chunk or slow chunks
                    logger.info(f"Chunk {i+1} processed in {chunk_time:.2f}s ({text_len} chars, {tokens} tokens)")
                    print(f"[COREF] Chunk {i+1} processed in {chunk_time:.2f}s ({text_len} chars, ~{tokens} tokens)")
                
                # Create a new chunk with the resolved text
                resolved_chunk = chunk.copy()
                resolved_chunk['text'] = resolved_text
                resolved_chunk['coref_applied'] = True
                
                processed_chunks.append(resolved_chunk)
            
            # Log final statistics
            total_time = time.time() - start_time
            avg_resolution_time = sum(resolution_times) / len(resolution_times) if resolution_times else 0
            
            completion_msg = (
                f"Coreference resolution complete for {len(chunks)} chunks in {total_time:.2f}s. "
                f"Avg resolution time: {avg_resolution_time:.4f}s per chunk. "
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
            logger.info("Unloading Maverick model...")
            print(f"[COREF] Unloading Maverick model...")
            self._unload_model()
            logger.info(f"Maverick model unloaded in {time.time() - unload_start:.2f}s")
            print(f"[COREF] Maverick model unloaded in {time.time() - unload_start:.2f}s")
    
    def _apply_coreference_resolution(self, text: str) -> str:
        """
        Apply coreference resolution to text.
        
        Args:
            text (str): Text to process
            
        Returns:
            str: Text with coreference resolution applied
        """
        try:
            # Predict coreferences
            data = self.model.predict(text)
            
            # Replace coreferences with their antecedents
            resolved_text = self._replace_coreferences_with_originals(
                data['tokens'], data['clusters_token_offsets']
            )
            
            return resolved_text
            
        except Exception as e:
            logger.error(f"Error applying coreference resolution: {e}")
            return text  # Return original text on error
    
    def _replace_coreferences_with_originals(self, tokens, clusters_token_offsets):
        """
        Replace coreferences with original entities, noting the original in brackets.
        For example: "Jason (##he) is 30..."
        
        Args:
            tokens (list): List of tokens
            clusters_token_offsets (list): List of cluster token offsets
            
        Returns:
            str: Text with coreferences replaced
        """
        # Create a copy of tokens to modify
        modified_tokens = list(tokens)

        # Process clusters in reverse order to avoid index shifting problems
        for cluster in reversed(list(clusters_token_offsets)):
            if not cluster:
                continue

            # Get the first mention (antecedent) in the cluster
            antecedent_start, antecedent_end = cluster[0]
            antecedent = tokens[antecedent_start:antecedent_end + 1]
            antecedent_text = " ".join(antecedent)

            # Replace all subsequent mentions with the antecedent
            # Process in reverse order to maintain correct indices
            for mention_start, mention_end in reversed(cluster[1:]):
                original_mention = tokens[mention_start:mention_end + 1]
                original_text = " ".join(original_mention)

                # Check for special cases
                if len(original_mention) == 1:
                    mention = original_mention[0].lower()

                    # Handle contractions (he's, she's, they're, etc.)
                    if "'" in mention:
                        parts = mention.split("'")
                        if len(parts) == 2 and parts[1] in ["s", "re", "ve", "ll", "d"]:
                            # Handle different contractions
                            if parts[1] == "s":  # he's, she's (can be "is" or "has")
                                # For simplicity, we'll assume 's means "is"
                                replacement = [f"{antecedent_text} is", f"(#{original_text}#)"]
                            elif parts[1] == "re":  # they're
                                replacement = [f"{antecedent_text} are", f"(#{original_text}#)"]
                            elif parts[1] == "ve":  # they've
                                replacement = [f"{antecedent_text} have", f"(#{original_text}#)"]
                            elif parts[1] == "ll":  # they'll
                                replacement = [f"{antecedent_text} will", f"(#{original_text}#)"]
                            elif parts[1] == "d":  # they'd (can be "would" or "had")
                                # Defaulting to 'would' for simplicity
                                replacement = [f"{antecedent_text} would", f"(#{original_text}#)"]
                            else: # Should not happen based on list above
                                replacement = antecedent + [f"(#{original_text}#)"]

                            modified_tokens[mention_start:mention_end + 1] = replacement
                            continue

                    # Handle possessive pronouns (his, her, their)
                    if mention in ["his", "her", "their", "hers", "theirs"]:
                        replacement = [f"{antecedent_text}'s", f"(#{original_text}#)"] # Possessive form
                        modified_tokens[mention_start:mention_end + 1] = replacement
                        continue

                # Default case: just replace with antecedent + original in brackets
                replacement = antecedent + [f"(#{original_text}#)"]
                modified_tokens[mention_start:mention_end + 1] = replacement

        # Join back into text
        text = " ".join(modified_tokens)

        # Clean up spacing around punctuation (basic examples)
        text = text.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
        text = text.replace(" 's", "'s") # Handle spacing before possessive 's
        
        return text
