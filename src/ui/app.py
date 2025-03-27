"""
Main Streamlit application for the Anti-Corruption RAG system.
"""
import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
import json

# Add parent directory to sys.path to enable imports from project root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.core.config import CONFIG, DATA_DIR
from src.processing.pipeline import run_processing_pipeline
from src.core.query_handler import QueryHandler
from src.ui.components import (
    styled_header,
    file_uploader,
    processing_status,
    document_explorer,
    entity_explorer,
    entity_profile,
    chat_interface
)
from src.utils.logger import setup_logger
from src.utils.file_utils import save_uploaded_files, clear_temp_files

logger = setup_logger(__name__)

# Page configuration
st.set_page_config(
    page_title=CONFIG["ui"]["page_title"],
    page_icon=CONFIG["ui"]["page_icon"],
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown(f"""
<style>
    :root {{
        --primary-color: {CONFIG["ui"]["theme_color"]};
        --accent-color: {CONFIG["ui"]["accent_color"]};
    }}
    
    .stButton button {{
        background-color: var(--primary-color);
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }}
    
    .stButton button:hover {{
        background-color: var(--accent-color);
    }}
    
    .stProgress .st-bo {{
        background-color: var(--accent-color);
    }}
    
    .st-bq {{
        border-left-color: var(--primary-color);
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: var(--primary-color);
    }}
    
    a {{
        color: var(--accent-color);
    }}
    
    /* Sidebar styling */
    .css-1d391kg {{
        background-color: #f8fafc;
    }}
</style>
""", unsafe_allow_html=True)

def load_processed_data():
    """
    Load processed data from files.
    
    Returns:
        tuple: (documents, entities, relationships)
    """
    # Initialize empty data
    documents = []
    entities = []
    relationships = []
    
    try:
        # Documents - no central storage yet, placeholder
        documents_file = DATA_DIR / "processed_documents.json"
        if documents_file.exists():
            with open(documents_file, 'r') as f:
                documents = json.load(f)
        
        # Entities
        entities_file = DATA_DIR / "extracted" / "entities.json"
        if entities_file.exists():
            with open(entities_file, 'r') as f:
                entities = json.load(f)
        
        # Relationships
        relationships_file = DATA_DIR / "extracted" / "relationships.json"
        if relationships_file.exists():
            with open(relationships_file, 'r') as f:
                relationships = json.load(f)
        
    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        st.error(f"Error loading processed data: {str(e)}")
    
    return documents, entities, relationships

def initialize_data_directories():
    """
    Initialize all necessary data directories for the system.
    """
    directories = [
        DATA_DIR,
        DATA_DIR / "bm25_indices",
        DATA_DIR / "extracted",
        DATA_DIR / "ocr_cache"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    """Main application function."""
    # Initialize directories
    initialize_data_directories()
    # Initialize session state
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "processed_documents" not in st.session_state:
        st.session_state.processed_documents = []
    if "selected_entity" not in st.session_state:
        st.session_state.selected_entity = None
    if "view" not in st.session_state:
        st.session_state.view = "upload"  # 'upload', 'explore', 'chat', 'entity_profile'
    
    # Display header
    styled_header()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## Navigation")
        
        # Navigation buttons
        if st.button("📄 Upload & Process", use_container_width=True):
            st.session_state.view = "upload"
            st.session_state.selected_entity = None
        
        if st.button("🔍 Explore Documents", use_container_width=True):
            st.session_state.view = "explore"
            st.session_state.selected_entity = None
        
        if st.button("💬 Query Documents", use_container_width=True):
            st.session_state.view = "chat"
            st.session_state.selected_entity = None
        
        # System info
        st.markdown("---")
        st.markdown("## System Info")
        
        # BM25 index info
        bm25_file = DATA_DIR / "bm25_indices" / "latest_index.pkl"
        if bm25_file.exists():
            st.markdown("✅ BM25 Index: Available")
        else:
            st.markdown("❌ BM25 Index: Not available")
        
        # Vector DB info
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
                st.markdown(f"✅ Vector DB: {collection_info.points_count} vectors indexed")
            else:
                st.markdown("❌ Vector DB: Not available")
                
        except Exception as e:
            st.markdown("❌ Vector DB: Connection error")
        
        # Entity count
        entities_file = DATA_DIR / "extracted" / "entities.json"
        if entities_file.exists():
            with open(entities_file, 'r') as f:
                entities = json.load(f)
            st.markdown(f"✅ Entities: {len(entities)} extracted")
        else:
            st.markdown("❌ Entities: None extracted")
        
        # Relationship count
        relationships_file = DATA_DIR / "extracted" / "relationships.json"
        if relationships_file.exists():
            with open(relationships_file, 'r') as f:
                relationships = json.load(f)
            st.markdown(f"✅ Relationships: {len(relationships)} extracted")
        else:
            st.markdown("❌ Relationships: None extracted")
        
        # Settings (advanced options)
        st.markdown("---")
        with st.expander("⚙️ Advanced Settings"):
            # Retrieval settings
            st.markdown("#### Retrieval Settings")
            st.slider(
                "Vector Search Results",
                min_value=5,
                max_value=50,
                value=CONFIG["retrieval"]["top_k_vector"],
                step=5,
                key="top_k_vector"
            )
            
            st.slider(
                "BM25 Search Results",
                min_value=5,
                max_value=50,
                value=CONFIG["retrieval"]["top_k_bm25"],
                step=5,
                key="top_k_bm25"
            )
            
            st.slider(
                "Vector Weight",
                min_value=0.0,
                max_value=1.0,
                value=CONFIG["retrieval"]["vector_weight"],
                step=0.1,
                key="vector_weight"
            )
            
            # Generation settings
            st.markdown("#### Generation Settings")
            st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=CONFIG["generation"]["temperature"],
                step=0.1,
                key="temperature"
            )
            
            st.slider(
                "Max Tokens",
                min_value=100,
                max_value=2000,
                value=CONFIG["generation"]["max_tokens"],
                step=100,
                key="max_tokens"
            )
    
    # Main content
    if st.session_state.processing:
        # Display processing status
        st.markdown("## Document Processing")
        processing_success = processing_status(
            st.session_state.status_queue,
            st.session_state.process
        )
        
        # Check if processing is complete
        if not st.session_state.process.is_alive():
            st.session_state.processing = False
            
            # Clean up temp files
            if hasattr(st.session_state, 'temp_file_paths'):
                clear_temp_files(st.session_state.temp_file_paths)
                del st.session_state.temp_file_paths
            
            if processing_success:
                # Switch to explore view on success
                st.session_state.view = "explore"
                st.rerun()
            
    elif st.session_state.view == "upload":
        # Display file uploader
        uploaded_files, process_clicked = file_uploader()
        
        if process_clicked and uploaded_files:
            # Save uploaded files to temp location
            temp_paths = save_uploaded_files(uploaded_files)
            
            # Start processing in subprocess
            st.session_state.process, st.session_state.status_queue = run_processing_pipeline([str(path) for path in temp_paths])
            st.session_state.processing = True
            
            # Store paths for cleanup
            st.session_state.temp_file_paths = temp_paths
            
            # Rerun to show processing status
            st.rerun()
    
    elif st.session_state.view == "explore":
        # Load processed data
        documents, entities, relationships = load_processed_data()
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Documents", "Entities & Relationships"])
        
        with tab1:
            # Display document explorer
            document_explorer(documents)
        
        with tab2:
            # Handle entity selection
            def select_entity(entity_id):
                st.session_state.selected_entity = entity_id
                st.session_state.view = "entity_profile"
                st.rerun()
            
            # Display entity explorer
            entity_explorer(entities, relationships, select_entity)
    
    elif st.session_state.view == "entity_profile":
        # Display entity profile
        if st.session_state.selected_entity:
            # Load entities and relationships
            _, entities, relationships = load_processed_data()
            
            # Find entity by ID
            entity_data = None
            for entity in entities:
                if entity.get("entity_id") == st.session_state.selected_entity:
                    entity_data = entity
                    break
            
            # Find relationships for entity
            entity_text = entity_data.get("text") if entity_data else None
            entity_relationships = []
            
            if entity_text:
                for rel in relationships:
                    if rel.get("subject") == entity_text or rel.get("object") == entity_text:
                        entity_relationships.append(rel)
            
            # Display entity profile
            entity_profile(entity_data, entity_relationships)
            
            # Back button
            if st.button("← Back to Entity Explorer"):
                st.session_state.view = "explore"
                st.session_state.selected_entity = None
                st.rerun()
        else:
            st.error("No entity selected")
            st.session_state.view = "explore"
            st.rerun()
    
    elif st.session_state.view == "chat":
        # Initialize query handler
        query_handler = QueryHandler()
        
        # Display chat interface
        chat_interface(query_handler)

if __name__ == "__main__":
    main()
