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
            from maverick import Maverick
            
            self._update_status("Loading Maverick coreference model")
            self.model = Maverick()
            self._update_status("Maverick model loaded successfully")
            log_memory_usage(logger)
        except Exception as e:
            error_msg = f"Error loading Maverick model: {e}"
            self._update_status(error_msg)
            logger.error(error_msg)
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
        self._update_status(f"Processing {len(chunks)} chunks for coreference resolution")
        
        try:
            # Load the model once at the beginning of processing
            self._load_model()
            
            # Create a copy of the chunks to avoid modifying the originals
            processed_chunks = []
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                progress = ((i + 1) / len(chunks))
                self._update_status(f"Applying coreference resolution: chunk {i+1}/{len(chunks)}", progress)
                
                # Get the text content
                text = chunk.get('text', '')
                if not text.strip():
                    # Skip empty chunks
                    processed_chunks.append(chunk)
                    continue
                
                # Apply coreference resolution
                resolved_text = self._apply_coreference_resolution(text)
                
                # Create a new chunk with the resolved text
                resolved_chunk = chunk.copy()
                resolved_chunk['text'] = resolved_text
                resolved_chunk['coref_applied'] = True
                
                processed_chunks.append(resolved_chunk)
            
            self._update_status(f"Coreference resolution complete for {len(chunks)} chunks")
            return processed_chunks
            
        except Exception as e:
            error_msg = f"Error in coreference resolution: {e}"
            self._update_status(error_msg)
            logger.error(error_msg)
            raise
        finally:
            # Ensure model is unloaded even if there's an error
            self._unload_model()
    
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
