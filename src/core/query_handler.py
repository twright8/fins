"""
Query handler that combines retrieval and generation.
"""
import sys
from pathlib import Path
import time
from typing import List, Dict, Any, Tuple, Optional

# Add parent directory to sys.path to enable imports from project root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.core.config import CONFIG
from src.core.retriever import Retriever
from src.core.generator import Generator
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class QueryHandler:
    """
    Handles user queries by retrieving relevant context and generating responses.
    """
    
    def __init__(self, lazy_init=False, qdrant_only=False):
        """
        Initialize query handler with retriever and generator.
        
        Args:
            lazy_init (bool): If True, don't load embedding models on initialization
            qdrant_only (bool): If True, only initialize Qdrant connection without embedding models
        """
        # Track config
        self.lazy_init = lazy_init
        self.qdrant_only = qdrant_only
        # Initialize basic properties
        self.entities_data = None
        self.relationships_data = None
        self._retriever = None
        self._generator = None
        
        # Load entity and relationship data (this is quick)
        self._load_processed_data()
        
        # Initialize components based on config
        if not lazy_init:
            self._init_retriever()
            if not qdrant_only:
                self._init_generator()
        
        logger.info(f"QueryHandler initialized with lazy_init={lazy_init}, qdrant_only={qdrant_only}")
    
    def _init_retriever(self):
        """Initialize the retriever (can be expensive)."""
        if self._retriever is None:
            logger.info("Initializing retriever")
            self._retriever = Retriever(lazy_init=self.lazy_init, qdrant_only=self.qdrant_only)
            logger.info("Retriever initialized")
    
    def _init_generator(self):
        """Initialize the generator (can be expensive)."""
        if self._generator is None:
            logger.info("Initializing generator")
            self._generator = Generator()
            logger.info("Generator initialized")
    
    @property
    def retriever(self):
        """Get the retriever, initializing it if necessary."""
        if self._retriever is None:
            self._init_retriever()
        return self._retriever
    
    @property
    def generator(self):
        """Get the generator, initializing it if necessary."""
        if self._generator is None:
            self._init_generator()
        return self._generator
    
    def _load_processed_data(self):
        """
        Load entities and relationships from disk into memory.
        """
        try:
            import json
            from src.core.config import DATA_DIR
            
            # Load entities
            entities_file = DATA_DIR / "extracted" / "entities.json"
            if entities_file.exists():
                with open(entities_file, 'r') as f:
                    self.entities_data = json.load(f)
                    logger.info(f"Loaded {len(self.entities_data)} entities into memory")
            else:
                self.entities_data = []
                logger.warning("No entities file found")
            
            # Load relationships
            relationships_file = DATA_DIR / "extracted" / "relationships.json"
            if relationships_file.exists():
                with open(relationships_file, 'r') as f:
                    self.relationships_data = json.load(f)
                    logger.info(f"Loaded {len(self.relationships_data)} relationships into memory")
            else:
                self.relationships_data = []
                logger.warning("No relationships file found")
                
        except Exception as e:
            logger.error(f"Error loading entity/relationship data: {e}")
            self.entities_data = []
            self.relationships_data = []
    
    def process_query(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process a user query by retrieving context and generating a response.
        
        Args:
            query (str): User query
            
        Returns:
            tuple: (answer, context_chunks)
                - answer: Generated answer
                - context_chunks: Retrieved context chunks
        """
        try:
            # Start timing
            start_time = time.time()
            
            # Validate input
            if not query.strip():
                return "Please enter a valid query.", []
            
            logger.info(f"Processing query: {query}")
            
            # Step 1: Retrieve relevant chunks
            retrieved_chunks = self.retriever.retrieve(query, use_reranking=True)
            
            # Log retrieval results
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
            if not retrieved_chunks:
                return "I couldn't find any relevant information in the uploaded documents. Please try a different query or upload more relevant documents.", []
            
            # Step 2: Generate response using retrieved chunks
            answer = self.generator.generate_with_context(query, retrieved_chunks)
            
            # Log generation
            logger.info(f"Generated answer of length {len(answer)}")
            
            # Log timing
            elapsed_time = time.time() - start_time
            logger.info(f"Query processing completed in {elapsed_time:.2f}s")
            
            return answer, retrieved_chunks
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"An error occurred while processing your query: {str(e)}", []
    
    def get_entity_information(self, entity_id: str) -> Dict[str, Any]:
        """
        Get information about a specific entity.
        
        Args:
            entity_id (str): Entity ID
            
        Returns:
            dict: Entity information
        """
        try:
            # Ensure data is loaded
            if self.entities_data is None:
                self._load_processed_data()
            
            # Return error if still no data
            if not self.entities_data:
                return {"error": "No entity data available"}
            
            # Find the entity by ID
            for entity in self.entities_data:
                if entity.get('entity_id') == entity_id:
                    return entity
            
            return {"error": f"Entity {entity_id} not found"}
            
        except Exception as e:
            logger.error(f"Error getting entity information: {e}")
            return {"error": str(e)}
    
    def get_relationships_for_entity(self, entity_id: str) -> List[Dict[str, Any]]:
        """
        Get relationships involving a specific entity.
        
        Args:
            entity_id (str): Entity ID
            
        Returns:
            list: Relationships involving the entity
        """
        try:
            # Ensure data is loaded
            if self.entities_data is None or self.relationships_data is None:
                self._load_processed_data()
            
            # Return empty list if no data
            if not self.entities_data or not self.relationships_data:
                return []
            
            # Find the entity by ID
            entity_text = None
            for entity in self.entities_data:
                if entity.get('entity_id') == entity_id:
                    entity_text = entity.get('text')
                    break
            
            if not entity_text:
                return []
            
            # Find relationships involving the entity
            entity_relationships = []
            for rel in self.relationships_data:
                if rel.get('subject') == entity_text or rel.get('object') == entity_text:
                    entity_relationships.append(rel)
            
            return entity_relationships
            
        except Exception as e:
            logger.error(f"Error getting relationships: {e}")
            return []
