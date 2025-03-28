"""
Main Streamlit application for the Anti-Corruption RAG system.
"""
import os
import sys
import time
import json
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
import json

# Configure basic logging to ensure output goes to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

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

# Enable more verbose logging for libraries
logging.getLogger('transformers').setLevel(logging.INFO)
logging.getLogger('flair').setLevel(logging.INFO)
logging.getLogger('embed').setLevel(logging.INFO)
logging.getLogger('qdrant_client').setLevel(logging.INFO)

# Setup our own logger
logger = setup_logger(__name__)

# Print startup message for better CLI visibility
print("\n" + "=" * 80)
print(" Anti-Corruption RAG System - Starting Up ".center(80, "="))
print("=" * 80)
print("Command-line output enabled for better visibility of background operations")
print(f"Model cache location: {os.path.expanduser('~/.cache/huggingface')}")
print("=" * 80 + "\n")

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
    Load processed entity and relationship data from files.
    
    Returns:
        tuple: (entities, relationships)
    """
    # Initialize empty data
    entities = []
    relationships = []
    
    try:
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
    
    return entities, relationships

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
    if "show_delete_confirmation" not in st.session_state:
        st.session_state.show_delete_confirmation = False
    
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
            
            # LLM Provider selection
            provider_options = ["aphrodite", "deepseek"]
            current_provider = CONFIG["generation"]["provider"]
            
            # Check if DeepSeek API key is configured
            deepseek_api_key = CONFIG["generation"]["deepseek"]["api_key"]
            deepseek_available = bool(deepseek_api_key.strip())
            
            if not deepseek_available and current_provider == "deepseek":
                st.warning("DeepSeek selected but API key not configured. Will use Aphrodite instead.")
            
            # Create a selectbox for provider selection
            new_provider = st.selectbox(
                "LLM Provider",
                options=provider_options,
                index=provider_options.index(current_provider) if current_provider in provider_options else 0,
                help="Select the LLM provider to use for text generation"
            )
            
            # If provider has changed, update config
            if new_provider != current_provider:
                # Only allow switching to DeepSeek if API key is configured
                if new_provider == "deepseek" and not deepseek_available:
                    st.error("Cannot switch to DeepSeek - API key not configured in config.yaml")
                else:
                    # Update CONFIG in memory (this won't persist to disk)
                    CONFIG["generation"]["provider"] = new_provider
                    st.success(f"Switched to {new_provider} provider. Restart the app for changes to take effect.")
            
            # Generation parameters
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
        # Display processing status with spinner for visual feedback
        st.markdown("## Document Processing")
        
        with st.spinner("Processing documents... This may take several minutes depending on document size and complexity."):
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
                # Show success message
                st.success("Document processing completed successfully! You can now explore the documents and extracted entities or query the system.")
                
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
        entities, relationships = load_processed_data()
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Documents", "Chunks", "Entities & Relationships", "Named Entity Recognition"])
        
        with tab1:
            # Display document explorer with placeholder data
            # Note: We don't have a central documents repository yet
            document_explorer([])
            
        with tab2:
            # Display chunk explorer
            st.markdown("""
            <h2 style='color: #1E3A8A; margin-bottom: 0.5rem;'>Chunk Explorer</h2>
            <p style='color: #64748B; margin-bottom: 1rem;'>
                Explore individual chunks created from your documents and stored in the vector database.
            </p>
            """, unsafe_allow_html=True)
            
            try:
                # Put a status indicator to show we're working
                status_container = st.empty()
                status_container.info("Connecting to vector database...")
                
                # Use a direct connection to Qdrant instead of the full retriever
                try:
                    from qdrant_client import QdrantClient, models
                    
                    # Connect directly to Qdrant without initializing embedding models
                    client = QdrantClient(
                        host=CONFIG["qdrant"]["host"],
                        port=CONFIG["qdrant"]["port"]
                    )
                    
                    # Create a retriever wrapper to avoid loading embedding models
                    class QdrantRetriever:
                        def __init__(self, client):
                            self.client = client
                            
                        def get_collection_info(self):
                            try:
                                # Check if collection exists
                                collections = self.client.get_collections().collections
                                collection_names = [c.name for c in collections]
                            
                                if CONFIG["qdrant"]["collection_name"] in collection_names:
                                    # Get collection info
                                    collection_info = self.client.get_collection(CONFIG["qdrant"]["collection_name"])
                                
                                    return {
                                        'exists': True,
                                        'name': CONFIG["qdrant"]["collection_name"],
                                        'points_count': collection_info.points_count,
                                        'vector_size': collection_info.config.params.vectors.size,
                                        'distance': str(collection_info.config.params.vectors.distance)
                                    }
                                else:
                                    return {'exists': False}
                            except Exception as e:
                                return {'exists': False, 'error': str(e)}
                    
                        def get_chunks(self, limit=20, search_text=None, document_filter=None):
                            """
                            Get chunks from the vector database.
                            """
                            try:
                                # Build filter conditions
                                filter_conditions = []
                                
                                # We won't use filter conditions for text search - we'll handle it in Python
                                # for better word-by-word matching
                                
                                if document_filter:
                                    filter_conditions.append(
                                        models.FieldCondition(
                                            key="file_name",
                                            match=models.MatchValue(value=document_filter)
                                        )
                                    )
                                
                                # Combine filters - use 'must' only if we have conditions
                                filter_obj = models.Filter(must=filter_conditions) if filter_conditions else None
                                
                                # Use scroll_filter parameter instead of filter (correct parameter name for scroll method)
                                scroll_result = self.client.scroll(
                                    collection_name=CONFIG["qdrant"]["collection_name"],
                                    limit=limit,
                                    with_payload=True,
                                    with_vectors=False,
                                    scroll_filter=filter_obj  # This is the key fix - using scroll_filter instead of filter
                                )
                                
                                points = scroll_result[0]
                                
                                # Apply text search manually for better matching
                                filtered_points = points
                                
                                if search_text and search_text.strip():
                                    search_terms = search_text.lower().split()
                                    filtered_points = []
                                    
                                    for point in points:
                                        text = point.payload.get('text', '').lower()
                                        # Check if ANY of the search terms are in the text
                                        if any(term in text for term in search_terms):
                                            filtered_points.append(point)
                                else:
                                    filtered_points = points
                                
                                # Format results
                                chunks = []
                                for point in filtered_points:
                                    # Try to get original_filename from metadata, fallback to file_name
                                    file_name = point.payload.get('original_filename', 
                                                point.payload.get('file_name', 'Unknown'))
                                    
                                    chunks.append({
                                        'id': point.id,
                                        'text': point.payload.get('text', ''),
                                        'metadata': {
                                            **{k: v for k, v in point.payload.items() if k != 'text'},
                                            'file_name': file_name  # Ensure we use the best file name
                                        }
                                    })
                                
                                return chunks
                                
                            except Exception as e:
                                print(f"Error getting chunks: {e}")
                                import traceback
                                print(traceback.format_exc())
                                return []
                            
                            # No redundant code needed here
                    
                    # Create a retriever instance without loading embedding models
                    retriever = QdrantRetriever(client)
                
                except Exception as e:
                    st.error(f"Error setting up retriever: {str(e)}")
                    status_container.empty()
                    return
                
                # Update status
                status_container.info("Fetching collection info...")
                
                # Get collection info with a timeout
                collection_info = None
                try:
                    # Use a timeout to prevent UI hanging
                    collection_info_result = [None]
                    error_result = [None]
                    
                    def get_info():
                        try:
                            collection_info_result[0] = retriever.get_collection_info()
                        except Exception as e:
                            error_result[0] = str(e)
                            logger.error(f"Error getting collection info: {str(e)}")
                    
                    # Start a thread to get collection info
                    thread = threading.Thread(target=get_info)
                    thread.start()
                    
                    # Wait for the thread to complete with a timeout
                    thread.join(timeout=5.0)  # 5 second timeout
                    
                    # Check if we got the info
                    if error_result[0] is not None:
                        st.error(f"Error getting collection info: {error_result[0]}")
                        status_container.empty()
                        st.warning("Could not retrieve collection information.")
                        return
                    
                    if collection_info_result[0] is None:
                        st.error("Collection info retrieval timed out")
                        status_container.empty()
                        st.warning("Retrieving collection information timed out. Please try again later.")
                        return
                    
                    collection_info = collection_info_result[0]
                    
                except Exception as e:
                    logger.error(f"Exception in collection info thread management: {str(e)}")
                    st.error(f"Error getting collection info: {str(e)}")
                    status_container.empty()
                    return
                    
                # Clear the status now that we have the info
                status_container.empty()
                
                if collection_info and collection_info.get('exists', False):
                    chunk_count = collection_info.get('points_count', 0)
                    st.success(f"Found {chunk_count:,} chunks in the vector database.")
                
                # Create a card with collection stats and add a more visible Delete button
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(
                        f"""
                        <div style="background-color: #f8f9fa; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
                            <div style="display: flex; justify-content: space-between;">
                                <div><strong>Collection:</strong> {collection_info.get('name')}</div>
                                <div><strong>Vectors:</strong> {chunk_count:,}</div>
                                <div><strong>Dimensions:</strong> {collection_info.get('vector_size')}</div>
                                <div><strong>Distance:</strong> {collection_info.get('distance')}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
                    if st.button("🗑️ DELETE ALL DATA", type="secondary", use_container_width=True, 
                                help="Delete all vectors and indices from the database"):
                        st.session_state.show_delete_confirmation = True
                
                # Handle deletion confirmation
                if st.session_state.get('show_delete_confirmation'):
                    st.warning("⚠️ Are you sure you want to delete all indices? This cannot be undone.")
                    delete_col1, delete_col2 = st.columns(2)
                    
                    with delete_col1:
                        if st.button("✓ Yes, Delete Everything", type="primary", key="confirm_delete_button"):
                            try:
                                # Delete the collection
                                success = client.delete_collection(CONFIG["qdrant"]["collection_name"])
                                if success:
                                    # Also delete entities and relationships files
                                    entities_file = DATA_DIR / "extracted" / "entities.json"
                                    relationships_file = DATA_DIR / "extracted" / "relationships.json"
                                    
                                    if entities_file.exists():
                                        os.remove(entities_file)
                                    
                                    if relationships_file.exists():
                                        os.remove(relationships_file)
                                    
                                    # Reset session state
                                    st.session_state.show_delete_confirmation = False
                                    st.success(f"All data deleted successfully!")
                                    time.sleep(1)  # Brief pause to show success message
                                    st.rerun()
                                else:
                                    st.error("Failed to delete collection")
                            except Exception as e:
                                st.error(f"Error deleting data: {str(e)}")
                    
                    with delete_col2:
                        if st.button("✗ Cancel", type="secondary", key="cancel_delete_button"):
                            st.session_state.show_delete_confirmation = False
                            st.rerun()
                
                # Let user select how many chunks to view
                num_chunks = st.slider("Number of chunks to view", 10, min(100, chunk_count), 20)
                
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    search_text = st.text_input("Search in chunks", placeholder="Enter text to search for...")
                
                with col2:
                    doc_filter = st.text_input("Filter by document", placeholder="Enter document name...")
                
                # Get chunks
                if chunk_count > 0:
                    with st.spinner("Fetching chunks..."):
                        chunks = retriever.get_chunks(
                            limit=num_chunks,
                            search_text=search_text if search_text else None,
                            document_filter=doc_filter if doc_filter else None
                        )
                    
                    if chunks:
                        # Let user choose display mode
                        display_mode = st.radio(
                            "Display mode",
                            ["Simple", "Card", "Detailed"],
                            horizontal=True,
                            index=1
                        )
                        
                        # Display chunks based on selected mode
                        if display_mode == "Simple":
                            # Simple list view
                            for i, chunk in enumerate(chunks):
                                document_id = chunk['metadata'].get('document_id', 'Unknown')
                                file_name = chunk['metadata'].get('file_name', 'Unknown')
                                page_num = chunk['metadata'].get('page_num', None)
                                
                                # Create expander with summary information
                                page_info = f", Page {page_num}" if page_num else ""
                                with st.expander(f"Chunk {i+1}: {file_name}{page_info}"):
                                    st.text_area(
                                        f"Content",
                                        value=chunk['text'],
                                        height=200,
                                        key=f"chunk_{i}"
                                    )
                        
                        elif display_mode == "Card":
                            # Card view with highlighting
                            for i, chunk in enumerate(chunks):
                                document_id = chunk['metadata'].get('document_id', 'Unknown')
                                file_name = chunk['metadata'].get('file_name', 'Unknown')
                                page_num = chunk['metadata'].get('page_num', None)
                                chunk_id = chunk['id']
                                
                                # Create expander with card styling
                                page_info = f", Page {page_num}" if page_num else ""
                                with st.expander(f"Chunk {i+1}: {file_name}{page_info}", expanded=i==0):
                                    # Header with metadata
                                    st.markdown(
                                        f"""
                                        <div style="background-color: #f1f5f9; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
                                            <div><strong>ID:</strong> {chunk_id}</div>
                                            <div><strong>Document:</strong> {document_id}</div>
                                            <div><strong>File:</strong> {file_name}</div>
                                            {f"<div><strong>Page:</strong> {page_num}</div>" if page_num else ""}
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                                    
                                    # Highlight search terms if provided
                                    if search_text and search_text.strip():
                                        # Add HTML highlighting for search term
                                        import re
                                        pattern = re.compile(f"({re.escape(search_text)})", re.IGNORECASE)
                                        highlighted_content = pattern.sub(r'<span style="background-color: #FFFF00; font-weight: bold;">\1</span>', chunk['text'])
                                        
                                        # Display with highlighted search terms
                                        st.markdown(
                                            f"<div style='border: 1px solid #e2e8f0; border-radius: 4px; padding: 12px; margin-top: 10px; font-family: monospace; white-space: pre-wrap; background-color: white;'>{highlighted_content}</div>",
                                            unsafe_allow_html=True
                                        )
                                    else:
                                        # Regular display
                                        st.markdown(
                                            f"<div style='border: 1px solid #e2e8f0; border-radius: 4px; padding: 12px; margin-top: 10px; font-family: monospace; white-space: pre-wrap; background-color: white;'>{chunk['text']}</div>",
                                            unsafe_allow_html=True
                                        )
                        
                        else:  # Detailed view
                            # Full detailed view with all metadata
                            for i, chunk in enumerate(chunks):
                                document_id = chunk['metadata'].get('document_id', 'Unknown')
                                file_name = chunk['metadata'].get('file_name', 'Unknown')
                                page_num = chunk['metadata'].get('page_num', None)
                                chunk_id = chunk['id']
                                
                                # Create expander with detailed information
                                page_info = f", Page {page_num}" if page_num else ""
                                with st.expander(f"Chunk {i+1}: {file_name}{page_info}", expanded=i==0):
                                    # Two columns for metadata and content
                                    col1, col2 = st.columns([1, 2])
                                    
                                    with col1:
                                        st.markdown("#### Metadata")
                                        # Display all metadata
                                        for key, value in chunk['metadata'].items():
                                            if key != 'document_metadata':  # Skip nested metadata
                                                st.markdown(f"**{key}:** {value}")
                                        
                                        # Display document metadata if available
                                        doc_metadata = chunk['metadata'].get('document_metadata', {})
                                        if doc_metadata:
                                            st.markdown("#### Document Metadata")
                                            for key, value in doc_metadata.items():
                                                st.markdown(f"**{key}:** {value}")
                                    
                                    with col2:
                                        st.markdown("#### Content")
                                        # Highlight search terms if provided
                                        if search_text and search_text.strip():
                                            # Add HTML highlighting for search term
                                            import re
                                            pattern = re.compile(f"({re.escape(search_text)})", re.IGNORECASE)
                                            highlighted_content = pattern.sub(r'<span style="background-color: #FFFF00; font-weight: bold;">\1</span>', chunk['text'])
                                            
                                            # Display with highlighted search terms
                                            st.markdown(
                                                f"<div style='border: 1px solid #e2e8f0; border-radius: 4px; padding: 12px; height: 300px; overflow-y: auto; font-family: monospace; white-space: pre-wrap; background-color: white;'>{highlighted_content}</div>",
                                                unsafe_allow_html=True
                                            )
                                        else:
                                            # Regular display with text area
                                            st.text_area(
                                                "Content",
                                                value=chunk['text'],
                                                height=300,
                                                key=f"chunk_detail_{i}"
                                            )
                    else:
                        st.info("No chunks match your search criteria.")
            except Exception as e:
                st.warning("Error accessing vector database. Please process documents first.")
                st.error(f"Error details: {str(e)}")
                
                # Print stack trace to console for debugging
                import traceback
                print(f"Error in chunk explorer: {traceback.format_exc()}")
        
        with tab3:
            # Handle entity selection
            def select_entity(entity_id):
                st.session_state.selected_entity = entity_id
                st.session_state.view = "entity_profile"
                st.rerun()
            
            # Display entity explorer
            entity_explorer(entities, relationships, select_entity)
            
        with tab4:
            # Named Entity Recognition tab
            st.markdown("""
            <h2 style='color: #1E3A8A; margin-bottom: 0.5rem;'>Named Entity Recognition</h2>
            <p style='color: #64748B; margin-bottom: 1rem;'>
                Extract named entities and relationships from text using Flair NER.
            </p>
            """, unsafe_allow_html=True)
            
            # Input methods
            input_method = st.radio(
                "Select input method",
                ["Enter text", "Use chunk from database"],
                horizontal=True
            )
            
            if input_method == "Enter text":
                # Text input area
                input_text = st.text_area(
                    "Enter text for entity extraction",
                    value="""Jason is 30 years old and earns thirty thousand pounds per year.
He lives in London and works as a Software Engineer. His employee ID is 12345.
Susan, aged 9, does not have a job and earns zero pounds.
She is a student in elementary school and lives in Manchester.
Michael, a 45-year-old Doctor, earns ninety-five thousand pounds annually.
He resides in Birmingham and has an employee ID of 67890.
Emily is a 28-year-old Data Scientist who earns seventy-two thousand pounds.
She is based in Edinburgh and her employee ID is 54321.""",
                    height=200
                )
            else:
                # Show dropdown of chunks
                status_container = st.empty()
                status_container.info("Loading chunks from database...")
                
                try:
                    # Connect to Qdrant client
                    from qdrant_client import QdrantClient
                    
                    client = QdrantClient(
                        host=CONFIG["qdrant"]["host"],
                        port=CONFIG["qdrant"]["port"]
                    )
                    
                    # Get chunks
                    scroll_result = client.scroll(
                        collection_name=CONFIG["qdrant"]["collection_name"],
                        limit=30,  # Get a reasonable number of chunks
                        with_payload=True,
                        with_vectors=False
                    )
                    
                    points = scroll_result[0]
                    
                    # Create a list of chunk previews
                    chunk_options = []
                    for point in points:
                        text = point.payload.get('text', '')
                        file_name = point.payload.get('original_filename', point.payload.get('file_name', 'Unknown'))
                        preview = f"{file_name}: {text[:50]}..." if len(text) > 50 else text
                        chunk_options.append((preview, text))
                    
                    status_container.empty()
                    
                    if chunk_options:
                        selected_preview = st.selectbox(
                            "Select a chunk",
                            options=[preview for preview, _ in chunk_options],
                            format_func=lambda x: x
                        )
                        
                        # Get the full text for the selected preview
                        input_text = next((text for preview, text in chunk_options if preview == selected_preview), "")
                    else:
                        st.warning("No chunks found in the database")
                        input_text = ""
                        
                except Exception as e:
                    status_container.empty()
                    st.error(f"Error loading chunks: {str(e)}")
                    input_text = ""
            
            # Process the text with Flair NER
            if input_text and st.button("Extract Entities and Relations"):
                with st.spinner("Processing with Flair NER..."):
                    try:
                        # Import Flair components
                        import flair
                        from flair.nn import Classifier
                        from flair.data import Sentence
                        from flair.splitter import SegtokSentenceSplitter
                        
                        # Add debugging info
                        st.info("Loading Flair models and processing text...")
                        
                        # Create a Text object from input
                        splitter = SegtokSentenceSplitter()
                        
                        # Debug: Print what models are being loaded
                        st.write("Loading NER model (flair/ner-english-ontonotes-fast)...")
                        
                        # Use NER tagger
                        tagger = Classifier.load("flair/ner-english-ontonotes-fast")
                        
                        # Split text into sentences
                        st.write("Splitting text into sentences...")
                        sentences = splitter.split(input_text)
                        st.write(f"Found {len(sentences)} sentences.")
                        
                        # Process sentences for entities
                        st.write("Processing sentences for named entities...")
                        for sentence in sentences:
                            tagger.predict(sentence)
                            
                        # Debug: Check if entities were found
                        entity_count = sum(len(sentence.get_labels('ner')) for sentence in sentences)
                        st.write(f"Found {entity_count} entities.")
                        
                        # Load relation extractor if entities were found
                        if entity_count > 0:
                            st.write("Loading relation extractor...")
                            try:
                                extractor = Classifier.load('flair/relations-english')
                                
                                # Process sentences for relationships
                                st.write("Processing sentences for relationships...")
                                for sentence in sentences:
                                    extractor.predict(sentence)
                            except Exception as rel_error:
                                st.warning(f"Could not load relation extractor: {rel_error}")
                                st.warning("Proceeding with entity extraction only.")
                        
                        # Collect entities and relations
                        entities = []
                        for sentence in sentences:
                            for entity in sentence.get_labels('ner'):
                                entities.append({
                                    'text': entity.data_point.text,
                                    'tag': entity.data_point.tag,
                                    'score': entity.score
                                })
                        
                        relationships = []
                        for sentence in sentences:
                            for relation in sentence.get_labels('relation'):
                                if hasattr(relation.data_point, 'first') and hasattr(relation.data_point, 'second'):
                                    relationships.append({
                                        'first': relation.data_point.first.text,
                                        'second': relation.data_point.second.text,
                                        'tag': relation.data_point.tag,
                                        'score': relation.score
                                    })
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Entities")
                            if entities:
                                # Group entities by tag
                                entities_by_tag = {}
                                for entity in entities:
                                    tag = entity['tag']
                                    if tag not in entities_by_tag:
                                        entities_by_tag[tag] = []
                                    entities_by_tag[tag].append(entity)
                                
                                # Display entities grouped by tag
                                for tag, tag_entities in entities_by_tag.items():
                                    with st.expander(f"{tag} ({len(tag_entities)})", expanded=True):
                                        for entity in tag_entities:
                                            st.markdown(f"**{entity['text']}** (Score: {entity['score']:.2f})")
                            else:
                                st.info("No entities found")
                        
                        with col2:
                            st.subheader("Relationships")
                            if relationships:
                                # Group relationships by tag
                                relationships_by_tag = {}
                                for relation in relationships:
                                    tag = relation['tag']
                                    if tag not in relationships_by_tag:
                                        relationships_by_tag[tag] = []
                                    relationships_by_tag[tag].append(relation)
                                
                                # Display relationships grouped by tag
                                for tag, tag_relations in relationships_by_tag.items():
                                    with st.expander(f"{tag} ({len(tag_relations)})", expanded=True):
                                        for relation in tag_relations:
                                            st.markdown(f"**{relation['first']}** → **{relation['second']}** (Score: {relation['score']:.2f})")
                            else:
                                st.info("No relationships found")
                                
                        # Add visualized text with highlighted entities
                        st.subheader("Highlighted Text")
                        html_parts = []
                        
                        # Process each sentence
                        for sentence in sentences:
                            text = sentence.text
                            html = text
                            
                            # Get entities in this sentence
                            sent_entities = []
                            for entity in sentence.get_labels('ner'):
                                sent_entities.append({
                                    'text': entity.data_point.text,
                                    'tag': entity.data_point.tag,
                                    'start_pos': entity.data_point.start_position,
                                    'end_pos': entity.data_point.end_position
                                })
                            
                            # Sort entities by start position, in reverse (to start replacing from the end)
                            sent_entities.sort(key=lambda x: x['start_pos'], reverse=True)
                            
                            # Replace text with highlighted spans
                            for entity in sent_entities:
                                highlight = f'<span style="background-color: rgba(255, 255, 0, 0.5); border-radius: 3px; padding: 1px 3px;">{entity["text"]} <small style="color: #1E3A8A;">({entity["tag"]})</small></span>'
                                html = html[:entity['start_pos']] + highlight + html[entity['end_pos']:]
                            
                            html_parts.append(html)
                        
                        # Join all sentences with line breaks
                        final_html = "<br>".join(html_parts)
                        st.markdown(f'<div style="border: 1px solid #e2e8f0; border-radius: 4px; padding: 12px; font-family: sans-serif; line-height: 1.6;">{final_html}</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error processing with Flair NER: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
    
    elif st.session_state.view == "entity_profile":
        # Display entity profile
        if st.session_state.selected_entity:
            # Load entities and relationships
            entities, relationships = load_processed_data()
            
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
        # Display chat interface with loading state and error handling
        st.markdown("""
        <h2 style='color: #1E3A8A; margin-bottom: 0.5rem;'>Conversational Interface</h2>
        <p style='color: #64748B; margin-bottom: 1rem;'>
            Ask questions about your documents and get contextual answers.
        </p>
        """, unsafe_allow_html=True)
        
        try:
            # Put a status indicator to show we're initializing
            status_container = st.empty()
            status_container.info("Initializing query system...")
            
            try:
                # Use a timeout to prevent UI hanging
                # Global variable to store the query handler
                query_handler_result = [None]
                error_result = [None]
                
                def initialize_handler():
                    try:
                        query_handler_result[0] = QueryHandler()
                    except Exception as e:
                        error_result[0] = str(e)
                        import traceback
                        print(f"Error initializing query handler: {traceback.format_exc()}")
                
                # Start a thread to initialize the query handler
                thread = threading.Thread(target=initialize_handler)
                thread.start()
                
                # Wait for the thread to complete with a timeout
                thread.join(timeout=10.0)  # 10 second timeout
                
                # Check if initialization succeeded
                if error_result[0] is not None:
                    st.error(f"Error initializing query handler: {error_result[0]}")
                    status_container.empty()
                    st.warning("Could not initialize the query system. Please check if the vector database is running and try again.")
                    return
                
                if query_handler_result[0] is None:
                    st.error("Query handler initialization timed out")
                    status_container.empty()
                    st.warning("Initialization timed out. This might be due to slow model loading or connection issues.")
                    return
                
                query_handler = query_handler_result[0]
                
                # Clear the status
                status_container.empty()
                
                # Display chat interface
                chat_interface(query_handler)
                
            except Exception as e:
                st.error(f"Error setting up query handler: {str(e)}")
                status_container.empty()
                import traceback
                st.error(traceback.format_exc())
                return
                
        except Exception as e:
            st.error(f"Unexpected error in chat view: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
