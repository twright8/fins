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
    
    def __init__(self, status_queue=None):
        """
        Initialize entity extractor.
        
        Args:
            status_queue (Queue, optional): Queue for status updates.
        """
        self.status_queue = status_queue
        self.tagger = None
        self.relation_extractor = None
        self.confidence_threshold = CONFIG["entity_extraction"]["confidence_threshold"]
        self.relationship_threshold = CONFIG["entity_extraction"]["relationship_threshold"]
        self.ner_model_name = CONFIG["models"]["ner_model"]
        
        logger.info(f"Initializing EntityExtractor with model={self.ner_model_name}")
        
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
            return
        
        try:
            from flair.models import SequenceTagger
            from flair.nn import Classifier
            from flair.data import Sentence
            
            self._update_status(f"Loading NER model: {self.ner_model_name}")
            self.tagger = SequenceTagger.load(self.ner_model_name)
            
            self._update_status("Loading relation extraction model")
            # Use a general relation extraction model (adjust as needed)
            # This is a placeholder - actual implementation depends on available models
            try:
                self.relation_extractor = Classifier.load('relations')
            except:
                logger.warning("Could not load relation model, relationships won't be extracted")
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
            self._update_status("Unloading NER and relation models")
            
            # Delete model references
            del self.tagger
            self.tagger = None
            
            if self.relation_extractor:
                del self.relation_extractor
                self.relation_extractor = None
            
            # Clean up CUDA memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            log_memory_usage(logger)
    
    def process_chunks(self, chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process chunks to extract entities and relationships.
        
        Args:
            chunks (list): List of chunk dictionaries
            
        Returns:
            tuple: (processed_chunks, entities, relationships)
                - processed_chunks: List of chunks with entity annotations
                - entities: List of extracted entities with metadata
                - relationships: List of extracted relationships
        """
        self._update_status(f"Processing {len(chunks)} chunks for entity extraction")
        
        # Load models
        self._load_models()
        
        try:
            # Create results containers
            processed_chunks = []
            all_entities = []
            all_relationships = []
            entity_mentions = {}  # Track all mentions of each entity
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                progress = ((i + 1) / len(chunks))
                self._update_status(f"Extracting entities and relationships: chunk {i+1}/{len(chunks)}", progress)
                
                # Get the text content
                text = chunk.get('text', '')
                if not text.strip():
                    # Skip empty chunks
                    processed_chunks.append(chunk)
                    continue
                
                # Process text to extract entities and relationships
                annotated_text, chunk_entities, chunk_relationships = self._process_text(
                    text, chunk.get('document_id', ''), chunk.get('chunk_id', ''))
                
                # Update entity mentions tracking
                for entity in chunk_entities:
                    entity_id = entity['entity_id']
                    if entity_id not in entity_mentions:
                        entity_mentions[entity_id] = []
                    
                    entity_mentions[entity_id].append({
                        'text': entity['text'],
                        'document_id': entity['document_id'],
                        'chunk_id': entity['chunk_id'],
                        'sentence': entity['sentence']
                    })
                
                # Create processed chunk with entity annotations
                processed_chunk = chunk.copy()
                processed_chunk['text'] = annotated_text
                processed_chunk['entity_count'] = len(chunk_entities)
                processed_chunk['relationship_count'] = len(chunk_relationships)
                processed_chunks.append(processed_chunk)
                
                # Add to result collections
                all_entities.extend(chunk_entities)
                all_relationships.extend(chunk_relationships)
            
            # Deduplicate entities and add mention counts
            deduplicated_entities = self._deduplicate_entities(all_entities)
            
            # Add mention information to entities
            for entity in deduplicated_entities:
                entity_id = entity['entity_id']
                if entity_id in entity_mentions:
                    entity['mention_count'] = len(entity_mentions[entity_id])
                    entity['mentions'] = entity_mentions[entity_id]
            
            # Deduplicate relationships
            deduplicated_relationships = self._deduplicate_relationships(all_relationships)
            
            # Unload models to free memory
            self._unload_models()
            
            self._update_status(f"Entity extraction complete. Found {len(deduplicated_entities)} entities and {len(deduplicated_relationships)} relationships")
            return processed_chunks, deduplicated_entities, deduplicated_relationships
            
        except Exception as e:
            # Make sure to unload models even if there's an error
            self._unload_models()
            
            error_msg = f"Error in entity extraction: {e}"
            self._update_status(error_msg)
            logger.error(error_msg)
            raise
    
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
        from flair.data import Sentence
        from flair.splitter import SegtokSentenceSplitter
        
        splitter = SegtokSentenceSplitter()
        
        try:
            # Split text into sentences
            sentences = splitter.split(text)
            
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
                
                # Extract entities from sentence
                sentence_entities = []
                for entity in sentence.get_spans('ner'):
                    entity_type = entity.tag
                    
                    # Skip unwanted entity types
                    if entity_type in self.UNWANTED_TYPES:
                        continue
                    
                    # Skip entities with low confidence
                    if entity.score < self.confidence_threshold:
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
                        'confidence': entity.score,
                        'start_pos': entity.start_pos,
                        'end_pos': entity.end_pos,
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
        Deduplicate entities based on entity_id.
        
        Args:
            entities (list): List of extracted entities
            
        Returns:
            list: Deduplicated entities
        """
        # Use a dictionary to track unique entities by ID
        unique_entities = {}
        
        for entity in entities:
            entity_id = entity['entity_id']
            
            if entity_id not in unique_entities:
                # This is a new entity
                unique_entities[entity_id] = entity
            else:
                # Update existing entity if this one has higher confidence
                if entity['confidence'] > unique_entities[entity_id]['confidence']:
                    unique_entities[entity_id] = entity
        
        return list(unique_entities.values())
    
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
