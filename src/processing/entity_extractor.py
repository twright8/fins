"""
Entity extraction and relationship classification module using Flair.
Extracts named entities and relationships from text.
"""
import sys
from pathlib import Path
import torch
import gc
from typing import List, Dict, Any, Tuple, Set
import uuid
import time
import concurrent.futures

# Add parent directory to sys.path to enable imports from project root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.core.config import CONFIG
from src.utils.logger import setup_logger
from src.utils.resource_monitor import log_memory_usage

logger = setup_logger(__name__)

class EntityExtractor:
    """
    Extract named entities and relationships from text using Flair.
    """
    
    # Entity Types Mapping
    ENTITY_MAPPING = {
        'PERSON': 'person', # Ontonotes uses uppercase
        'PER': 'person',
        'ORGANIZATION': 'organization',
        'ORG': 'organization',
        'GPE': 'location',  # Geopolitical entity
        'LOC': 'location',
        'LOCATION': 'location',
        'MONEY': 'money',
        'PRODUCT': 'product',
        'LAW': 'law',
        'NORP': 'norp',   # Nationalities, religious or political groups
        'EVENT': 'event'
    }
    
    # Unwanted entity types to skip
    UNWANTED_TYPES = [
        'CARDINAL', 'DATE', 'FAC', 'LANGUAGE', 'QUANTITY',
        'WORK_OF_ART', 'TIME', 'ORDINAL', 'PERCENT', 'DURATION'
    ]
    
    def __init__(self, status_queue=None, ner_model=None, relation_model=None):
        """
        Initialize entity extractor.
        
        Args:
            status_queue (Queue, optional): Queue for status updates.
            ner_model: A pre-loaded NER model to use instead of loading a new one.
            relation_model: A pre-loaded relation model to use instead of loading a new one.
        """
        self.status_queue = status_queue
        self.tagger = ner_model
        self.relation_extractor = relation_model
        self.confidence_threshold = CONFIG["entity_extraction"]["confidence_threshold"]
        self.relationship_threshold = CONFIG["entity_extraction"]["relationship_threshold"]
        self.ner_model_name = CONFIG["models"]["ner_model"]
        
        logger.info(f"Initializing EntityExtractor with model={self.ner_model_name}")
        print(f"[ENTITY] Initializing with {'pre-loaded' if ner_model else 'no'} NER model and {'pre-loaded' if relation_model else 'no'} relation model")
        
        if self.status_queue:
            self.status_queue.put(('status', 'Entity extractor initialized'))
        
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
    
    def _load_models(self):
        """
        Load NER and relation extraction models if not already loaded.
        """
        if self.tagger is not None and self.relation_extractor is not None:
            logger.info("Flair models already loaded, skipping load")
            print(f"[ENTITY] Flair models already loaded, skipping load")
            return
        
        print(f"[ENTITY] ===== STARTING FLAIR MODEL LOAD =====")
        logger.info("===== STARTING FLAIR MODEL LOAD =====")
        
        try:
            # Use Classifier for both NER and relations as per example code
            from flair.nn import Classifier
            from flair.data import Sentence
            import time
            import os
            
            # Configure Flair cache directory explicitly
            os.environ["FLAIR_CACHE_ROOT"] = os.path.expanduser("~/.cache/flair")
            
            # Print cache information for debugging
            cache_dir = os.environ.get("FLAIR_CACHE_ROOT", "Default cache location")
            hf_cache = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
            
            print(f"[ENTITY] Using Flair cache directory: {cache_dir}")
            print(f"[ENTITY] Using HuggingFace cache directory: {hf_cache}")
            logger.info(f"Using Flair cache directory: {cache_dir}")
            logger.info(f"Using HuggingFace cache directory: {hf_cache}")
            
            # NER model loading with detailed logging
            logger.info(f"Starting to load Flair NER model: {self.ner_model_name}...")
            print(f"[ENTITY] Starting to load Flair NER model: {self.ner_model_name} at {time.time():.2f}")
            self._update_status(f"Loading NER model: {self.ner_model_name}")
            
            # Check if model already exists in cache before loading
            import os
            model_name = "flair/ner-english-ontonotes-fast"
            cache_dir = os.environ.get("FLAIR_CACHE_ROOT", os.path.expanduser("~/.cache/flair"))
            hf_cache = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
            
            # Check various potential cache locations
            hf_potential_path = os.path.join(hf_cache, "hub", "models--" + model_name.replace("/", "--"))
            flair_potential_path = os.path.join(cache_dir, model_name.replace("/", "--"))
            
            if os.path.exists(hf_potential_path):
                print(f"[ENTITY] Model found in HuggingFace cache: {hf_potential_path}")
                logger.info(f"Model found in HuggingFace cache: {hf_potential_path}")
            elif os.path.exists(flair_potential_path):
                print(f"[ENTITY] Model found in Flair cache: {flair_potential_path}")
                logger.info(f"Model found in Flair cache: {flair_potential_path}")
            else:
                print(f"[ENTITY] Model not found in cache, will be downloaded")
                logger.info(f"Model not found in cache, will be downloaded")
            
            start_time = time.time()
            print(f"[ENTITY] Executing Classifier.load for {model_name} at {start_time:.2f}")
            
            # Using Classifier.load instead of SequenceTagger.load
            self.tagger = Classifier.load(model_name)
            elapsed_time = time.time() - start_time
            
            logger.info(f"Flair NER model loaded successfully in {elapsed_time:.2f} seconds")
            print(f"[ENTITY] Flair NER model loaded successfully in {elapsed_time:.2f} seconds")
            
            # Verify if model was loaded from cache
            if elapsed_time < 5.0:
                print(f"[ENTITY] Model likely loaded from cache (fast load time: {elapsed_time:.2f}s)")
                logger.info(f"Model likely loaded from cache (fast load time: {elapsed_time:.2f}s)")
            else:
                print(f"[ENTITY] Model may have been downloaded (slow load time: {elapsed_time:.2f}s)")
                logger.info(f"Model may have been downloaded (slow load time: {elapsed_time:.2f}s)")
            
            # Relation extraction model loading
            logger.info("Starting to load relation extraction model...")
            print(f"[ENTITY] Starting to load relation extraction model at {time.time():.2f}")
            self._update_status("Loading relation extraction model")
            
            # Use a general relation extraction model (adjust as needed)
            model_name = 'relations'
            
            # Check various potential cache locations for relation model
            hf_relation_path = os.path.join(hf_cache, "hub", "models--" + model_name)
            flair_relation_path = os.path.join(cache_dir, model_name)
            
            if os.path.exists(hf_relation_path):
                print(f"[ENTITY] Relation model found in HuggingFace cache: {hf_relation_path}")
                logger.info(f"Relation model found in HuggingFace cache: {hf_relation_path}")
            elif os.path.exists(flair_relation_path):
                print(f"[ENTITY] Relation model found in Flair cache: {flair_relation_path}")
                logger.info(f"Relation model found in Flair cache: {flair_relation_path}")
            else:
                print(f"[ENTITY] Relation model not found in cache, will be downloaded")
                logger.info(f"Relation model not found in cache, will be downloaded")
            
            start_time = time.time()
            try:
                print(f"[ENTITY] Executing relation Classifier.load for {model_name} at {start_time:.2f}")
                self.relation_extractor = Classifier.load(model_name)
                elapsed_time = time.time() - start_time
                logger.info(f"Relation extraction model loaded successfully in {elapsed_time:.2f} seconds")
                print(f"[ENTITY] Relation extraction model loaded successfully in {elapsed_time:.2f} seconds")
                
                # Verify if model was loaded from cache
                if elapsed_time < 5.0:
                    print(f"[ENTITY] Relation model likely loaded from cache (fast load time: {elapsed_time:.2f}s)")
                    logger.info(f"Relation model likely loaded from cache (fast load time: {elapsed_time:.2f}s)")
                else:
                    print(f"[ENTITY] Relation model may have been downloaded (slow load time: {elapsed_time:.2f}s)")
                    logger.info(f"Relation model may have been downloaded (slow load time: {elapsed_time:.2f}s)")
            except Exception as rel_error:
                logger.warning(f"Could not load relation model: {rel_error}")
                logger.warning("Relationships won't be extracted. This is expected if no relation model is available.")
                print(f"[ENTITY] Could not load relation model: {rel_error}")
                print(f"[ENTITY] Relationships won't be extracted. This is expected if no relation model is available.")
                self.relation_extractor = None
            
            self._update_status("Models loaded successfully")
            log_memory_usage(logger)
            
        except Exception as e:
            error_msg = f"Error loading models: {e}"
            self._update_status(error_msg)
            logger.error(error_msg)
            raise
    
    def _unload_models(self):
        """
        Unload models to free up memory.
        """
        if self.tagger is not None or self.relation_extractor is not None:
            print(f"[ENTITY] ===== STARTING FLAIR MODEL UNLOAD =====")
            logger.info("===== STARTING FLAIR MODEL UNLOAD =====")
            self._update_status("Unloading NER and relation models")
            
            # Delete model references
            print(f"[ENTITY] Deleting tagger reference at {time.time():.2f}")
            logger.info("Deleting tagger reference")
            del self.tagger
            self.tagger = None
            
            if self.relation_extractor:
                print(f"[ENTITY] Deleting relation_extractor reference at {time.time():.2f}")
                logger.info("Deleting relation_extractor reference")
                del self.relation_extractor
                self.relation_extractor = None
            
            # Clean up CUDA memory
            if torch.cuda.is_available():
                print(f"[ENTITY] Clearing CUDA cache at {time.time():.2f}")
                logger.info("Clearing CUDA cache")
                torch.cuda.empty_cache()
            
            # Force garbage collection
            print(f"[ENTITY] Running garbage collection at {time.time():.2f}")
            logger.info("Running garbage collection")
            gc.collect()
            
            print(f"[ENTITY] ===== FLAIR MODEL UNLOAD COMPLETE =====")
            logger.info("===== FLAIR MODEL UNLOAD COMPLETE =====")
            
            # Check cache status after unload
            self._check_cache_status()
            
            log_memory_usage(logger)
    
    def process_chunks(self, chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process chunks to extract entities and relationships using batched sentence processing.
        
        Args:
            chunks (list): List of chunk dictionaries
            
        Returns:
            tuple: (processed_chunks, entities, relationships)
                - processed_chunks: List of chunks with entity annotations
                - entities: List of extracted entities with metadata
                - relationships: List of extracted relationships
        """
        from collections import defaultdict
        import time
        
        start_time = time.time()
        self._update_status(f"Processing {len(chunks)} chunks for entity extraction using batch processing")
        
        try:
            # Load models if not already loaded
            if self.tagger is None or self.relation_extractor is None:
                print(f"[ENTITY] Loading models as they were not pre-loaded")
                self._load_models()
            else:
                print(f"[ENTITY] Using pre-loaded models")
                logger.info("Using pre-loaded NER and relation models")
            
            # 1. Sentence Aggregation - collect all sentences from all chunks
            all_sentences = []
            sentence_to_chunk_map = []  # Stores tuples: (sentence_index, chunk_id, document_id)
            
            import nltk
            from flair.data import Sentence
            
            # Download NLTK punkt tokenizer data if not available
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
            logger.info(f"Splitting {len(chunks)} chunks into sentences for batch processing...")
            self._update_status(f"Preparing sentences for batch processing", 0.1)
            
            empty_chunk_ids = set()  # Track empty chunks
            
            # Helper function to split a chunk text into sentences
            def _split_chunk_text_nltk(chunk, empty_chunk_ids_set, idx, total_chunks):
                chunk_id = chunk.get('chunk_id')
                document_id = chunk.get('document_id')
                text = chunk.get('text', '')
                
                if not text.strip():
                    # Skip empty chunks but track them
                    return chunk_id, [], []
                
                # Split chunk into sentences
                try:
                    split_start = time.time()
                    print(f"[ENTITY] Splitting chunk {chunk_id} ({idx+1}/{total_chunks}) at {split_start:.2f}")
                    
                    # Use NLTK's sent_tokenize instead of Flair's splitter
                    sentences_text = nltk.sent_tokenize(text)
                    
                    # Convert sentence strings to Flair Sentence objects
                    result_sentences = []
                    result_mappings = []
                    
                    for sent_text in sentences_text:
                        if sent_text.strip():
                            flair_sentence = Sentence(sent_text)
                            result_sentences.append(flair_sentence)
                            result_mappings.append((chunk_id, document_id))
                    
                    print(f"[ENTITY] Split chunk {chunk_id} into {len(sentences_text)} sentences in {time.time() - split_start:.4f}s")
                    return chunk_id, result_sentences, result_mappings
                        
                except Exception as e:
                    logger.error(f"Error splitting chunk {chunk_id}: {e}")
                    return chunk_id, [], []
            
            # Get number of workers from config for sentence splitting
            num_workers = CONFIG["document_processing"]["parallelism_workers"]["entity_splitting"]
            logger.info(f"Splitting {len(chunks)} chunks into sentences using {num_workers} workers")
            print(f"[ENTITY] Splitting {len(chunks)} chunks into sentences using {num_workers} workers")
            
            # Process chunks in parallel using ThreadPoolExecutor
            split_start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all chunk splitting tasks
                future_to_idx = {
                    executor.submit(_split_chunk_text_nltk, chunk, empty_chunk_ids, i, len(chunks)): i 
                    for i, chunk in enumerate(chunks) if chunk.get('text', '').strip()
                }
                
                # Process results as they complete
                completed = 0
                for future in concurrent.futures.as_completed(future_to_idx):
                    chunk_id, chunk_sentences, chunk_mappings = future.result()
                    
                    if not chunk_sentences:
                        empty_chunk_ids.add(chunk_id)
                    else:
                        # Add sentences to the main list
                        start_idx = len(all_sentences)
                        all_sentences.extend(chunk_sentences)
                        
                        # Create mapping entries with correct indices
                        for i, (c_id, doc_id) in enumerate(chunk_mappings):
                            sentence_to_chunk_map.append((start_idx + i, c_id, doc_id))
                    
                    # Update progress
                    completed += 1
                    if completed % 10 == 0 or completed == len(future_to_idx):
                        progress = 0.1 + (0.2 * (completed / len(future_to_idx)))
                        self._update_status(f"Split {completed}/{len(future_to_idx)} chunks into sentences", progress)
            
            split_time = time.time() - split_start_time
            logger.info(f"Split {len(chunks)} chunks into {len(all_sentences)} sentences in {split_time:.2f}s")
            
            # 2. Batch Prediction with Flair
            if all_sentences:
                # Configure batch sizes from config
                ner_batch_size = CONFIG["entity_extraction"].get("ner_batch_size", 64)  # Default to 64 if not in config
                re_batch_size = CONFIG["entity_extraction"].get("relation_batch_size", 32)  # Default to 32 if not in config
                
                # Predict named entities in batches
                ner_start_time = time.time()
                print(f"[ENTITY] ===== STARTING NER PREDICTION =====")
                print(f"[ENTITY] Predicting named entities for {len(all_sentences)} sentences at {ner_start_time:.2f}...")
                self._update_status(f"Predicting named entities for {len(all_sentences)} sentences...", 0.3)
                num_batches = (len(all_sentences) + ner_batch_size - 1) // ner_batch_size
                print(f"[ENTITY] Processing in {num_batches} batches with batch size {ner_batch_size}")
                
                self.tagger.predict(all_sentences, mini_batch_size=ner_batch_size)
                
                ner_time = time.time() - ner_start_time
                logger.info(f"NER prediction completed for {len(all_sentences)} sentences in {ner_time:.2f}s")
                print(f"[ENTITY] NER prediction completed for {len(all_sentences)} sentences in {ner_time:.2f}s")
                print(f"[ENTITY] Average processing time: {ner_time/len(all_sentences):.4f}s per sentence")
                print(f"[ENTITY] ===== NER PREDICTION COMPLETE =====")
                
                # Predict relationships in batches if relation extractor is available
                if self.relation_extractor:
                    re_start_time = time.time()
                    print(f"[ENTITY] ===== STARTING RELATIONSHIP PREDICTION =====")
                    print(f"[ENTITY] Predicting relationships for {len(all_sentences)} sentences at {re_start_time:.2f}...")
                    self._update_status(f"Predicting relationships for {len(all_sentences)} sentences...", 0.6)
                    rel_num_batches = (len(all_sentences) + re_batch_size - 1) // re_batch_size
                    print(f"[ENTITY] Processing in {rel_num_batches} batches with batch size {re_batch_size}")
                    
                    self.relation_extractor.predict(all_sentences, mini_batch_size=re_batch_size)
                    
                    re_time = time.time() - re_start_time
                    logger.info(f"Relationship prediction completed for {len(all_sentences)} sentences in {re_time:.2f}s")
                    print(f"[ENTITY] Relationship prediction completed for {len(all_sentences)} sentences in {re_time:.2f}s")
                    print(f"[ENTITY] Average processing time: {re_time/len(all_sentences):.4f}s per sentence")
                    print(f"[ENTITY] ===== RELATIONSHIP PREDICTION COMPLETE =====")
            else:
                logger.warning("No valid sentences found in any chunks")
            
            # 3. Result Aggregation
            chunk_entities = defaultdict(list)      # Entities grouped by chunk_id
            chunk_relationships = defaultdict(list) # Relationships grouped by chunk_id
            chunk_annotated_texts = defaultdict(list) # Annotated texts grouped by chunk_id
            
            self._update_status(f"Extracting entities and relationships from processed sentences...", 0.8)
            
            # Process each sentence and collect results by chunk
            for i, sentence in enumerate(all_sentences):
                _, chunk_id, document_id = sentence_to_chunk_map[i]
                sentence_original_text = sentence.to_original_text()
                sentence_entities_for_annotation = []  # Entities just for this sentence's annotation
                
                # Extract entities from this sentence using get_labels('ner')
                ner_labels = sentence.get_labels('ner')
                
                for entity_label in ner_labels:
                    entity = entity_label.data_point
                    entity_type = entity.tag
                    
                    # Apply filters (unwanted types, confidence threshold)
                    if entity_type in self.UNWANTED_TYPES:
                        continue
                    if entity_label.score < self.confidence_threshold:
                        continue
                    
                    # Map entity type to our schema
                    mapped_type = self.ENTITY_MAPPING.get(entity_type, 'other')
                    
                    # Generate entity fingerprint/ID
                    import hashlib
                    entity_text = entity.text
                    fingerprint = hashlib.md5(f"{entity_text.lower()}:{mapped_type}".encode()).hexdigest()
                    entity_id = fingerprint
                    
                    # Create entity dictionary
                    entity_dict = {
                        'entity_id': entity_id,
                        'text': entity_text,
                        'type': mapped_type,
                        'original_type': entity_type,
                        'confidence': entity_label.score,
                        'start_pos': getattr(entity, 'start_position', 0),  # Relative to sentence
                        'end_pos': getattr(entity, 'end_position', len(entity_text)),  # Relative to sentence
                        'document_id': document_id,
                        'chunk_id': chunk_id,
                        'sentence': sentence_original_text
                    }
                    
                    # Add to chunk-specific entities and to annotation list
                    chunk_entities[chunk_id].append(entity_dict)
                    sentence_entities_for_annotation.append(entity_dict)
                
                # Extract relationships if available
                if self.relation_extractor:
                    relations = sentence.get_labels('relation')
                    
                    # Log relationship information for debugging
                    if len(relations) > 0:
                        print(f"[ENTITY] Found {len(relations)} relationships in sentence: {sentence_original_text[:50]}...")
                        logger.info(f"Found {len(relations)} relationships in sentence")
                    
                    for relation in relations:
                        # Skip low confidence relationships
                        if relation.score < self.relationship_threshold:
                            continue
                        
                        # Try different ways to access relationship data based on Flair version
                        relation_data = {
                            'relationship_id': str(uuid.uuid4()),
                            'confidence': relation.score,
                            'type': relation.value,
                            'document_id': document_id,
                            'chunk_id': chunk_id,
                            'sentence': sentence_original_text
                        }
                        
                        # Try different attribute access patterns based on Flair versions
                        try:
                            # Get relationship data - version 1: data_point with head/tail entities
                            if hasattr(relation.data_point, 'head_entity') and hasattr(relation.data_point, 'tail_entity'):
                                relation_data['subject'] = relation.data_point.head_entity.text
                                relation_data['object'] = relation.data_point.tail_entity.text
                                print(f"[ENTITY] Relation v1: {relation_data['subject']} -> {relation_data['object']} ({relation.value})")
                            
                            # Version 2: first/second attributes
                            elif hasattr(relation.data_point, 'first') and hasattr(relation.data_point, 'second'):
                                relation_data['subject'] = relation.data_point.first.text
                                relation_data['object'] = relation.data_point.second.text
                                print(f"[ENTITY] Relation v2: {relation_data['subject']} -> {relation_data['object']} ({relation.value})")
                            
                            # Version 3: Direct text attribute
                            elif hasattr(relation.data_point, 'text'):
                                # Parse from format like "Jason -> 30 years old"
                                rel_text = relation.data_point.text
                                print(f"[ENTITY] Raw relation: {rel_text} ({relation.value})")
                                
                                if ' -> ' in rel_text:
                                    subject, obj = rel_text.split(' -> ', 1)
                                    relation_data['subject'] = subject
                                    relation_data['object'] = obj
                                    print(f"[ENTITY] Relation v3: {relation_data['subject']} -> {relation_data['object']} ({relation.value})")
                            
                            # Add relationship only if we found both subject and object
                            if 'subject' in relation_data and 'object' in relation_data:
                                chunk_relationships[chunk_id].append(relation_data)
                            else:
                                print(f"[ENTITY] Could not extract subject/object from relation: {relation}")
                                
                        except Exception as rel_err:
                            logger.warning(f"Error extracting relationship data: {rel_err}")
                            print(f"[ENTITY] Error extracting relationship: {rel_err}")
                
                # Create annotated text for this sentence
                annotated_sentence = self._annotate_sentence_with_entities(
                    sentence_original_text, sentence_entities_for_annotation
                )
                
                # Add to chunk-specific annotated texts
                chunk_annotated_texts[chunk_id].append(annotated_sentence)
            
            # 4. Final Assembly - build processed chunks and aggregate entities/relationships
            processed_chunks = []
            all_aggregated_entities = []
            all_aggregated_relationships = []
            entity_mentions = {}  # Track all mentions of each entity
            
            for chunk in chunks:
                chunk_id = chunk.get('chunk_id')
                
                # Create a copy of the chunk
                processed_chunk = chunk.copy()
                
                if chunk_id in empty_chunk_ids:
                    # No changes for empty chunks
                    processed_chunks.append(processed_chunk)
                    continue
                
                # Get entities and relationships for this chunk
                entities = chunk_entities.get(chunk_id, [])
                relationships = chunk_relationships.get(chunk_id, [])
                
                # Reconstruct annotated text by joining annotated sentences
                annotated_text = "\n".join(chunk_annotated_texts.get(chunk_id, [chunk.get('text', '')]))
                
                # Update processed chunk
                processed_chunk['text'] = annotated_text
                processed_chunk['entity_count'] = len(entities)
                processed_chunk['relationship_count'] = len(relationships)
                processed_chunks.append(processed_chunk)
                
                # Add to result collections
                all_aggregated_entities.extend(entities)
                all_aggregated_relationships.extend(relationships)
                
                # Update entity mentions tracking
                for entity in entities:
                    entity_id = entity['entity_id']
                    if entity_id not in entity_mentions:
                        entity_mentions[entity_id] = []
                    
                    entity_mentions[entity_id].append({
                        'text': entity['text'],
                        'document_id': entity['document_id'],
                        'chunk_id': entity['chunk_id'],
                        'sentence': entity['sentence']
                    })
            
            # Deduplicate entities and add mention counts
            deduplicated_entities = self._deduplicate_entities(all_aggregated_entities)
            
            # Add mention information to entities
            for entity in deduplicated_entities:
                entity_id = entity['entity_id']
                if entity_id in entity_mentions:
                    entity['mention_count'] = len(entity_mentions[entity_id])
                    entity['mentions'] = entity_mentions[entity_id]
            
            # Deduplicate relationships
            deduplicated_relationships = self._deduplicate_relationships(all_aggregated_relationships)
            
            total_time = time.time() - start_time
            self._update_status(
                f"Entity extraction complete in {total_time:.2f}s. Found {len(deduplicated_entities)} entities "
                f"and {len(deduplicated_relationships)} relationships",
                1.0
            )
            
            logger.info(
                f"Batched entity extraction complete in {total_time:.2f}s. "
                f"Processed {len(chunks)} chunks, {len(all_sentences)} sentences. "
                f"Found {len(deduplicated_entities)} entities and {len(deduplicated_relationships)} relationships."
            )
            
            return processed_chunks, deduplicated_entities, deduplicated_relationships
            
        except Exception as e:
            error_msg = f"Error in entity extraction: {e}"
            self._update_status(error_msg)
            logger.error(error_msg)
            raise
        finally:
            # Ensure models are unloaded even if there's an error
            self._unload_models()
    
    def _process_text(self, text: str, document_id: str, chunk_id: str) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process text to extract entities and relationships.
        
        Args:
            text (str): Text to process
            document_id (str): Document ID
            chunk_id (str): Chunk ID
            
        Returns:
            tuple: (annotated_text, entities, relationships)
        """
        import nltk
        from flair.data import Sentence
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            # Split text into sentences using NLTK
            sentences_text = nltk.sent_tokenize(text)
            sentences = [Sentence(sent_text) for sent_text in sentences_text if sent_text.strip()]
            
            # Initialize results
            entities = []
            relationships = []
            annotated_parts = []
            current_position = 0
            
            # Process each sentence
            for sentence in sentences:
                sentence_text = sentence.to_original_text()
                
                # Find sentence position in original text (to preserve spacing)
                sentence_position = text.find(sentence_text, current_position)
                if sentence_position > current_position:
                    # Add text between sentences
                    annotated_parts.append(text[current_position:sentence_position])
                
                # Update current position
                current_position = sentence_position + len(sentence_text)
                
                # Process sentence for entities
                self.tagger.predict(sentence)
                
                # Extract entities from sentence using get_labels('ner')
                sentence_entities = []
                ner_labels = sentence.get_labels('ner')
                
                for entity_label in ner_labels:
                    entity = entity_label.data_point
                    entity_type = entity.tag
                    
                    # Skip unwanted entity types
                    if entity_type in self.UNWANTED_TYPES:
                        continue
                    
                    # Skip entities with low confidence
                    if entity_label.score < self.confidence_threshold:
                        continue
                    
                    # Map entity type to our schema
                    mapped_type = self.ENTITY_MAPPING.get(entity_type, 'other')
                    
                    # Create unique entity ID based on text and type (for deduplication)
                    entity_text = entity.text
                    
                    # Generate fingerprint for entity
                    import hashlib
                    fingerprint = hashlib.md5(f"{entity_text.lower()}:{mapped_type}".encode()).hexdigest()
                    entity_id = fingerprint
                    
                    # Store entity
                    sentence_entities.append({
                        'entity_id': entity_id,
                        'text': entity_text,
                        'type': mapped_type,
                        'original_type': entity_type,
                        'confidence': entity_label.score,
                        'start_pos': getattr(entity, 'start_position', 0),
                        'end_pos': getattr(entity, 'end_position', len(entity_text)),
                        'document_id': document_id,
                        'chunk_id': chunk_id,
                        'sentence': sentence_text
                    })
                    
                    # Track the entity for annotation
                    entities.append(sentence_entities[-1])
                
                # Extract relationships if relation extractor is available
                if self.relation_extractor:
                    try:
                        self.relation_extractor.predict(sentence)
                        
                        for relation in sentence.get_labels('relation'):
                            # Skip relations with low confidence
                            if relation.score < self.relationship_threshold:
                                continue
                            
                            # Create relation object
                            relationship = {
                                'relationship_id': str(uuid.uuid4()),
                                'subject': relation.head_entity.text,
                                'object': relation.tail_entity.text,
                                'type': relation.value,
                                'confidence': relation.score,
                                'document_id': document_id,
                                'chunk_id': chunk_id,
                                'sentence': sentence_text
                            }
                            
                            relationships.append(relationship)
                    except Exception as e:
                        logger.warning(f"Error extracting relationships: {e}")
                
                # Create annotated sentence text with entity types
                annotated_sentence = self._annotate_sentence_with_entities(sentence_text, sentence_entities)
                annotated_parts.append(annotated_sentence)
            
            # Add any remaining text
            if current_position < len(text):
                annotated_parts.append(text[current_position:])
            
            # Combine annotated parts
            annotated_text = ''.join(annotated_parts)
            
            return annotated_text, entities, relationships
            
        except Exception as e:
            logger.error(f"Error processing text for entities: {e}")
            return text, [], []  # Return original text on error
    
    def _annotate_sentence_with_entities(self, sentence_text: str, entities: List[Dict[str, Any]]) -> str:
        """
        Annotate a sentence with entity type markers.
        For example: "John Smith (##person) works at Acme Corp (##organization)"
        
        Args:
            sentence_text (str): Original sentence text
            entities (list): List of entities in the sentence
            
        Returns:
            str: Annotated sentence text
        """
        # Sort entities by their end position in reverse order (to avoid position shifts)
        sorted_entities = sorted(entities, key=lambda e: e['end_pos'], reverse=True)
        
        # Insert entity type annotations
        for entity in sorted_entities:
            start_pos = entity['start_pos']
            end_pos = entity['end_pos']
            entity_type = entity['type']
            
            # Extract text before, entity, and text after
            text_before = sentence_text[:end_pos]
            entity_text = sentence_text[start_pos:end_pos]
            text_after = sentence_text[end_pos:]
            
            # Add annotation
            sentence_text = f"{text_before} (##{ entity_type }){text_after}"
        
        return sentence_text
    
    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate entities using fuzzy string matching.
        
        Args:
            entities (list): List of extracted entities
            
        Returns:
            list: Deduplicated entities
        """
        try:
            from thefuzz import fuzz
            
            # Get fuzzy match threshold from config
            match_threshold = CONFIG["entity_extraction"]["fuzzy_match_threshold"]
            logger.info(f"Using fuzzy matching with threshold {match_threshold}")
            
            # Sort entities by confidence (highest first)
            sorted_entities = sorted(entities, key=lambda e: e.get('confidence', 0), reverse=True)
            
            # List to hold deduplicated entities
            deduplicated = []
            
            # Track processed entity texts by type
            processed_texts = {}
            
            for entity in sorted_entities:
                entity_text = entity.get('text', '').lower()
                entity_type = entity.get('type', 'unknown')
                
                # Initialize entry for this entity type if it doesn't exist
                if entity_type not in processed_texts:
                    processed_texts[entity_type] = []
                
                # Check if this entity should be considered a duplicate
                is_duplicate = False
                
                # Only compare with entities of the same type
                for existing_text in processed_texts[entity_type]:
                    # Calculate similarity ratio using token_sort_ratio
                    # This handles word reordering, e.g., "John Smith" vs "Smith, John"
                    similarity = fuzz.token_sort_ratio(entity_text, existing_text)
                    
                    if similarity >= match_threshold:
                        is_duplicate = True
                        logger.debug(f"Found duplicate: '{entity_text}' matches '{existing_text}' with score {similarity}")
                        break
                
                if not is_duplicate:
                    # If unique, add to results and track the text
                    deduplicated.append(entity)
                    processed_texts[entity_type].append(entity_text)
                    logger.debug(f"Added unique entity: '{entity_text}' ({entity_type})")
            
            logger.info(f"Deduplicated entities from {len(entities)} to {len(deduplicated)} using fuzzy matching")
            return deduplicated
            
        except ImportError:
            logger.warning("thefuzz library not available. Falling back to exact ID matching.")
            return self._fallback_deduplicate_entities(entities)
    
    def _fallback_deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fallback method to deduplicate entities based on entity_id if fuzzy matching is unavailable.
        
        Args:
            entities (list): List of extracted entities
            
        Returns:
            list: Deduplicated entities
        """
        # Use a dictionary to track unique entities by ID
        unique_entities = {}
        
        for entity in entities:
            entity_id = entity.get('entity_id')
            
            if entity_id and entity_id not in unique_entities:
                # This is a new entity
                unique_entities[entity_id] = entity
            elif entity_id and entity['confidence'] > unique_entities[entity_id]['confidence']:
                # Update existing entity if this one has higher confidence
                unique_entities[entity_id] = entity
        
        return list(unique_entities.values())
    
    def _check_cache_status(self):
        """
        Check and report on the status of cached models.
        """
        import os
        import glob
        
        print(f"[ENTITY] ===== CHECKING FLAIR MODEL CACHE STATUS =====")
        logger.info("===== CHECKING FLAIR MODEL CACHE STATUS =====")
        
        # Define cache directories to check
        cache_dir = os.environ.get("FLAIR_CACHE_ROOT", os.path.expanduser("~/.cache/flair"))
        hf_cache = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
        transformers_cache = os.environ.get('TRANSFORMERS_CACHE', os.path.join(hf_cache, 'transformers'))
        
        # Check if directories exist
        print(f"[ENTITY] Flair cache directory exists: {os.path.exists(cache_dir)}")
        print(f"[ENTITY] HuggingFace cache directory exists: {os.path.exists(hf_cache)}")
        print(f"[ENTITY] Transformers cache directory exists: {os.path.exists(transformers_cache)}")
        
        logger.info(f"Flair cache directory exists: {os.path.exists(cache_dir)}")
        logger.info(f"HuggingFace cache directory exists: {os.path.exists(hf_cache)}")
        logger.info(f"Transformers cache directory exists: {os.path.exists(transformers_cache)}")
        
        # List models in Flair cache directory if it exists
        if os.path.exists(cache_dir):
            flair_models = glob.glob(os.path.join(cache_dir, "*"))
            print(f"[ENTITY] Found {len(flair_models)} items in Flair cache:")
            logger.info(f"Found {len(flair_models)} items in Flair cache:")
            for model in flair_models:
                print(f"[ENTITY] - {os.path.basename(model)}")
                logger.info(f"- {os.path.basename(model)}")
        
        # Look for NER models in HuggingFace cache
        ner_patterns = ["*ner*", "*flair*", "*ontonotes*"]
        hf_models_dir = os.path.join(hf_cache, "hub")
        
        if os.path.exists(hf_models_dir):
            for pattern in ner_patterns:
                matching_models = glob.glob(os.path.join(hf_models_dir, pattern))
                if matching_models:
                    print(f"[ENTITY] Found {len(matching_models)} models matching '{pattern}' in HuggingFace cache:")
                    logger.info(f"Found {len(matching_models)} models matching '{pattern}' in HuggingFace cache:")
                    for model in matching_models:
                        print(f"[ENTITY] - {os.path.basename(model)}")
                        logger.info(f"- {os.path.basename(model)}")
        
        print(f"[ENTITY] ===== CACHE STATUS CHECK COMPLETE =====")
        logger.info("===== CACHE STATUS CHECK COMPLETE =====")
        
    def _deduplicate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate relationships.
        
        Args:
            relationships (list): List of extracted relationships
            
        Returns:
            list: Deduplicated relationships
        """
        # Create fingerprints for relationships
        unique_relationships = {}
        
        for rel in relationships:
            # Create a fingerprint for the relationship based on its core properties
            import hashlib
            fingerprint = hashlib.md5(
                f"{rel['subject'].lower()}:{rel['type']}:{rel['object'].lower()}".encode()
            ).hexdigest()
            
            if fingerprint not in unique_relationships:
                # This is a new relationship
                rel['fingerprint'] = fingerprint
                unique_relationships[fingerprint] = rel
            else:
                # Update existing relationship if this one has higher confidence
                if rel['confidence'] > unique_relationships[fingerprint]['confidence']:
                    rel['fingerprint'] = fingerprint
                    unique_relationships[fingerprint] = rel
        
        return list(unique_relationships.values())
