"""
UI components for the Streamlit interface.
"""
import streamlit as st
import time
from typing import List, Dict, Any, Optional, Callable, Tuple
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network

def styled_header():
    """
    Display a styled header for the application.
    """
    # Use columns for layout
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.image("https://img.icons8.com/fluency/96/anti-fraud.png", width=80)
    
    with col2:
        st.markdown("""
        <h1 style='margin-bottom: 0; padding-bottom: 0; color: #1E3A8A;'>Anti-Corruption RAG System</h1>
        <p style='margin-top: 0; padding-top: 0; color: #3B82F6; font-size: 1.2em;'>
            Document Analysis & Intelligence Extraction
        </p>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

def file_uploader():
    """
    Display a file uploader with progress indication.
    
    Returns:
        tuple: (uploaded_files, process_clicked)
            - uploaded_files: List of uploaded files
            - process_clicked: Boolean indicating whether the process button was clicked
    """
    st.markdown("""
    <h2 style='color: #1E3A8A; margin-bottom: 0.5rem;'>Document Upload</h2>
    <p style='color: #64748B; margin-bottom: 1rem;'>
        Upload documents for analysis. Supported formats: PDF, DOCX, TXT, CSV, XLSX.
    </p>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "csv", "xlsx"],
        accept_multiple_files=True,
        help="Upload PDF, Word, text, or spreadsheet files for processing"
    )
    
    process_clicked = False
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} files selected**")
        
        # Process button
        col1, col2, col3 = st.columns([3, 2, 3])
        with col2:
            process_clicked = st.button(
                "🔍 Process Documents",
                type="primary",
                use_container_width=True,
                help="Start document processing"
            )
    
    return uploaded_files, process_clicked

def processing_status(status_queue, process):
    """
    Display processing status with progress bar.
    
    Args:
        status_queue: Queue for status updates
        process: Process object
    
    Returns:
        bool: True if processing completed successfully, False otherwise
    """
    # Create status containers
    status_container = st.empty()
    progress_container = st.empty()
    details_container = st.empty()
    
    # Initialize progress bar
    progress_bar = progress_container.progress(0)
    status_container.info("Starting document processing...")
    
    success = False
    error_message = None
    
    try:
        # Status monitoring loop
        while process.is_alive():
            # Check for status updates
            while not status_queue.empty():
                msg = status_queue.get()
                
                if msg[0] == 'progress':
                    # Update progress bar
                    progress, status_text = msg[1], msg[2]
                    progress_bar.progress(float(progress))
                    details_container.markdown(f"**{status_text}**")
                    
                elif msg[0] == 'status':
                    # Update status text
                    details_container.markdown(f"**{msg[1]}**")
                    
                elif msg[0] == 'error':
                    # Display error
                    error_message = msg[1]
                    details_container.error(error_message)
                    
                elif msg[0] == 'success':
                    # Processing completed successfully
                    success = True
                    details_container.success(msg[1])
            
            # Small sleep to prevent CPU hogging
            time.sleep(0.1)
        
        # Check final status
        while not status_queue.empty():
            msg = status_queue.get()
            
            if msg[0] == 'progress':
                progress_bar.progress(float(msg[1]))
                details_container.markdown(f"**{msg[2]}**")
                
            elif msg[0] == 'error':
                error_message = msg[1]
                details_container.error(error_message)
                
            elif msg[0] == 'success':
                success = True
                details_container.success(msg[1])
                progress_bar.progress(1.0)
        
        # Final status update
        if success:
            status_container.success("Document processing completed successfully!")
            progress_bar.progress(1.0)
        else:
            if error_message:
                status_container.error(f"Processing failed: {error_message}")
            else:
                status_container.error("Processing failed. Check logs for details.")
        
        return success
        
    except Exception as e:
        status_container.error(f"Error monitoring process: {str(e)}")
        return False

def document_explorer(documents: List[Dict[str, Any]]):
    """
    Display a document explorer interface.
    
    Args:
        documents (list): List of document metadata
    """
    st.markdown("""
    <h2 style='color: #1E3A8A; margin-bottom: 0.5rem;'>Document Explorer</h2>
    <p style='color: #64748B; margin-bottom: 1rem;'>
        View and explore processed documents.
    </p>
    """, unsafe_allow_html=True)
    
    if not documents:
        st.info("No documents have been processed yet. Upload and process documents to see them here.")
        return
    
    # Create document summary
    doc_summaries = []
    for doc in documents:
        doc_summaries.append({
            "Document": doc.get("file_name", "Unknown"),
            "Type": doc.get("file_type", "Unknown").upper(),
            "Pages": len(doc.get("content", [])),
            "Entities": doc.get("entity_count", 0),
            "Relationships": doc.get("relationship_count", 0)
        })
    
    # Display document summary
    st.dataframe(
        pd.DataFrame(doc_summaries),
        hide_index=True,
        use_container_width=True
    )
    
    # Document selector
    selected_doc = st.selectbox(
        "Select a document to view details",
        options=[doc.get("file_name", "Unknown") for doc in documents],
        index=0
    )
    
    # Get selected document
    selected_doc_data = next((doc for doc in documents if doc.get("file_name") == selected_doc), None)
    
    if selected_doc_data:
        # Display document details
        with st.expander("Document Details", expanded=True):
            # Document metadata
            st.markdown(f"**File Name:** {selected_doc_data.get('file_name', 'Unknown')}")
            st.markdown(f"**File Type:** {selected_doc_data.get('file_type', 'Unknown').upper()}")
            
            metadata = selected_doc_data.get("metadata", {})
            if metadata:
                st.markdown("#### Metadata")
                for key, value in metadata.items():
                    st.markdown(f"**{key}:** {value}")
            
            # Document content
            st.markdown("#### Content")
            
            content = selected_doc_data.get("content", [])
            tabs = st.tabs([f"Page {i+1}" for i in range(len(content))])
            
            for i, tab in enumerate(tabs):
                with tab:
                    if i < len(content):
                        page_content = content[i].get("text", "")
                        st.text_area(
                            f"Content - Page {i+1}",
                            value=page_content,
                            height=300,
                            key=f"page_{i}_{selected_doc}"
                        )

def entity_explorer(entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]], callback: Optional[Callable] = None):
    """
    Display an entity explorer interface.
    
    Args:
        entities (list): List of entity metadata
        relationships (list): List of relationship metadata
        callback (callable, optional): Callback function for entity selection
    """
    st.markdown("""
    <h2 style='color: #1E3A8A; margin-bottom: 0.5rem;'>Entity Explorer</h2>
    <p style='color: #64748B; margin-bottom: 1rem;'>
        Explore entities and relationships extracted from documents.
    </p>
    """, unsafe_allow_html=True)
    
    if not entities:
        st.info("No entities have been extracted yet. Process documents to extract entities.")
        return
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Entity List", "Entity Graph", "Relationship Table"])
    
    # Tab 1: Entity List
    with tab1:
        # Entity type filter
        entity_types = sorted(list(set(entity.get("type", "unknown") for entity in entities)))
        selected_types = st.multiselect(
            "Filter by entity type",
            options=entity_types,
            default=entity_types
        )
        
        # Filter entities
        filtered_entities = [
            entity for entity in entities
            if entity.get("type", "unknown") in selected_types
        ]
        
        # Create entity summary
        entity_summaries = []
        for entity in filtered_entities:
            entity_summaries.append({
                "Entity": entity.get("text", "Unknown"),
                "Type": entity.get("type", "Unknown"),
                "Confidence": f"{entity.get('confidence', 0):.2f}",
                "Mentions": entity.get("mention_count", 1)
            })
        
        # Display entity summary
        st.dataframe(
            pd.DataFrame(entity_summaries),
            hide_index=True,
            use_container_width=True
        )
        
        # Entity selector
        if filtered_entities:
            selected_entity = st.selectbox(
                "Select an entity to view details",
                options=[entity.get("text", "Unknown") for entity in filtered_entities],
                index=0
            )
            
            # Get selected entity
            selected_entity_data = next((entity for entity in filtered_entities if entity.get("text") == selected_entity), None)
            
            if selected_entity_data and callback:
                # Call the callback with entity ID
                st.button(
                    f"📊 View Detailed Profile for '{selected_entity}'",
                    type="primary",
                    on_click=callback,
                    args=(selected_entity_data.get("entity_id"),)
                )
    
    # Tab 2: Entity Graph
    with tab2:
        if not entities or not relationships:
            st.info("No relationships available for visualization.")
            return
        
        # Limit number of entities and relationships for visualization
        top_entities_count = st.slider("Number of top entities to include", 10, 100, 30)
        
        # Filter top entities by mention count
        top_entities = sorted(
            entities,
            key=lambda e: e.get("mention_count", 1),
            reverse=True
        )[:top_entities_count]
        
        # Create network graph
        entity_texts = [entity.get("text") for entity in top_entities]
        
        # Filter relationships involving top entities
        filtered_relationships = [
            rel for rel in relationships
            if rel.get("subject") in entity_texts and rel.get("object") in entity_texts
        ]
        
        # Create network
        G = nx.DiGraph()
        
        # Add nodes (entities)
        for entity in top_entities:
            G.add_node(
                entity.get("text"),
                type=entity.get("type", "unknown"),
                weight=entity.get("mention_count", 1)
            )
        
        # Add edges (relationships)
        for rel in filtered_relationships:
            G.add_edge(
                rel.get("subject"),
                rel.get("object"),
                type=rel.get("type", "unknown"),
                confidence=rel.get("confidence", 0)
            )
        
        # Create visualization with PyVis
        net = Network(height="600px", width="100%", directed=True, notebook=False)
        
        # Node color map by type
        colors = {
            "person": "#3B82F6",  # Blue
            "organization": "#10B981",  # Green
            "location": "#F59E0B",  # Yellow
            "money": "#EC4899",  # Pink
            "product": "#8B5CF6",  # Purple
            "law": "#6B7280",  # Gray
            "norp": "#EF4444",  # Red
            "event": "#6366F1",  # Indigo
            "unknown": "#9CA3AF"  # Light gray
        }
        
        # Add nodes with attributes
        for node, attrs in G.nodes(data=True):
            size = 10 + (attrs.get("weight", 1) * 3)  # Size based on mention count
            color = colors.get(attrs.get("type", "unknown"), colors["unknown"])
            
            net.add_node(
                node,
                label=node,
                title=f"Type: {attrs.get('type', 'unknown')}\nMentions: {attrs.get('weight', 1)}",
                size=size,
                color=color
            )
        
        # Add edges with attributes
        for source, target, attrs in G.edges(data=True):
            width = 1 + (attrs.get("confidence", 0) * 5)  # Width based on confidence
            
            net.add_edge(
                source,
                target,
                title=f"Type: {attrs.get('type', 'unknown')}\nConfidence: {attrs.get('confidence', 0):.2f}",
                width=width
            )
        
        # Set physics options
        net.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {
              "enabled": true,
              "iterations": 1000,
              "updateInterval": 25
            }
          }
        }
        """)
        
        # Save and display HTML
        html_file = "entity_graph.html"
        net.save_graph(html_file)
        
        with open(html_file, "r", encoding="utf-8") as f:
            html = f.read()
        
        st.components.v1.html(html, height=600)
    
    # Tab 3: Relationship Table
    with tab3:
        if not relationships:
            st.info("No relationships have been extracted.")
            return
        
        # Create relationship summary
        rel_summaries = []
        for rel in relationships:
            rel_summaries.append({
                "Subject": rel.get("subject", "Unknown"),
                "Relationship": rel.get("type", "Unknown"),
                "Object": rel.get("object", "Unknown"),
                "Confidence": f"{rel.get('confidence', 0):.2f}"
            })
        
        # Display relationship summary
        st.dataframe(
            pd.DataFrame(rel_summaries),
            hide_index=True,
            use_container_width=True
        )

def entity_profile(entity_data: Dict[str, Any], entity_relationships: List[Dict[str, Any]]):
    """
    Display a detailed profile for an entity.
    
    Args:
        entity_data (dict): Entity metadata
        entity_relationships (list): Relationships involving the entity
    """
    if not entity_data or "error" in entity_data:
        st.error(entity_data.get("error", "Entity not found"))
        return
    
    # Entity header
    entity_type = entity_data.get("type", "unknown")
    entity_name = entity_data.get("text", "Unknown Entity")
    
    type_colors = {
        "person": "#3B82F6",  # Blue
        "organization": "#10B981",  # Green
        "location": "#F59E0B",  # Yellow
        "money": "#EC4899",  # Pink
        "product": "#8B5CF6",  # Purple
        "law": "#6B7280",  # Gray
        "norp": "#EF4444",  # Red
        "event": "#6366F1",  # Indigo
        "unknown": "#9CA3AF"  # Light gray
    }
    
    type_color = type_colors.get(entity_type, type_colors["unknown"])
    
    st.markdown(f"""
    <h2 style='color: #1E3A8A; margin-bottom: 0.5rem;'>Entity Profile</h2>
    <div style='display: flex; align-items: center; margin-bottom: 1rem;'>
        <div style='background-color: {type_color}; color: white; padding: 4px 8px; border-radius: 4px; margin-right: 10px;'>
            {entity_type.upper()}
        </div>
        <div style='font-size: 1.5em; font-weight: bold;'>{entity_name}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Entity details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Entity Details")
        st.markdown(f"**Type:** {entity_data.get('type', 'Unknown')}")
        st.markdown(f"**Original Type:** {entity_data.get('original_type', 'Unknown')}")
        st.markdown(f"**Confidence:** {entity_data.get('confidence', 0):.3f}")
        st.markdown(f"**Mentions:** {entity_data.get('mention_count', 0)}")
    
    with col2:
        st.markdown("#### Document References")
        references = entity_data.get("mentions", [])
        
        if references:
            # Group mentions by document
            docs = {}
            for ref in references:
                doc_id = ref.get("document_id", "unknown")
                if doc_id not in docs:
                    docs[doc_id] = []
                docs[doc_id].append(ref)
            
            # Display document references
            for doc_id, refs in docs.items():
                st.markdown(f"**Document:** {doc_id}")
                st.markdown(f"**Occurrences:** {len(refs)}")
        else:
            st.info("No document references available.")
    
    # Entity relationships
    st.markdown("#### Relationships")
    
    if not entity_relationships:
        st.info("No relationships found for this entity.")
    else:
        # Create tabs for incoming and outgoing relationships
        tab1, tab2 = st.tabs(["As Subject", "As Object"])
        
        # Tab 1: Entity as subject
        with tab1:
            subject_rels = [rel for rel in entity_relationships if rel.get("subject") == entity_name]
            
            if not subject_rels:
                st.info("No relationships where this entity is the subject.")
            else:
                # Create relationship summary
                rel_summaries = []
                for rel in subject_rels:
                    rel_summaries.append({
                        "Relationship": rel.get("type", "Unknown"),
                        "Object": rel.get("object", "Unknown"),
                        "Confidence": f"{rel.get('confidence', 0):.2f}"
                    })
                
                # Display relationship summary
                st.dataframe(
                    pd.DataFrame(rel_summaries),
                    hide_index=True,
                    use_container_width=True
                )
        
        # Tab 2: Entity as object
        with tab2:
            object_rels = [rel for rel in entity_relationships if rel.get("object") == entity_name]
            
            if not object_rels:
                st.info("No relationships where this entity is the object.")
            else:
                # Create relationship summary
                rel_summaries = []
                for rel in object_rels:
                    rel_summaries.append({
                        "Subject": rel.get("subject", "Unknown"),
                        "Relationship": rel.get("type", "Unknown"),
                        "Confidence": f"{rel.get('confidence', 0):.2f}"
                    })
                
                # Display relationship summary
                st.dataframe(
                    pd.DataFrame(rel_summaries),
                    hide_index=True,
                    use_container_width=True
                )
    
    # Entity mentions in context
    st.markdown("#### Mentions in Context")
    
    mentions = entity_data.get("mentions", [])
    if not mentions:
        st.info("No context examples available.")
    else:
        # Show a sample of mentions (up to 5)
        for i, mention in enumerate(mentions[:5]):
            with st.expander(f"Mention {i+1} in {mention.get('document_id', 'unknown')}", expanded=i==0):
                st.markdown(mention.get("sentence", "No context available."))

def chat_interface(query_handler, on_submit=None):
    """
    Display a chat interface for querying the system.
    
    Args:
        query_handler: Query handler object
        on_submit: Callback function for query submission
    """
    st.markdown("""
    <h2 style='color: #1E3A8A; margin-bottom: 0.5rem;'>Conversational Interface</h2>
    <p style='color: #64748B; margin-bottom: 1rem;'>
        Ask questions about your documents and get contextual answers.
    </p>
    """, unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display context if available
            if "context" in message and message["context"]:
                with st.expander("View Source Context", expanded=False):
                    for i, ctx in enumerate(message["context"]):
                        st.markdown(f"**Source {i+1}:** {ctx['metadata'].get('file_name', 'Unknown')}")
                        if ctx['metadata'].get('page_num'):
                            st.markdown(f"**Page:** {ctx['metadata'].get('page_num')}")
                        st.markdown("**Excerpt:**")
                        st.markdown(ctx['text'])
                        st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            # Process query
            with st.spinner("Generating response..."):
                answer, context = query_handler.process_query(prompt)
            
            # Update placeholder with response
            message_placeholder.markdown(answer)
            
            # Add assistant message to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "context": context
            })
            
            # Call onsubmit callback if provided
            if on_submit:
                on_submit(prompt, answer, context)
