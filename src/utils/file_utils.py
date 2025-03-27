"""
File utility functions for handling temporary files.
"""
import os
import tempfile
from pathlib import Path
from typing import List, Any
import logging

logger = logging.getLogger(__name__)

def save_uploaded_files(uploaded_files: List[Any]) -> List[Path]:
    """
    Save uploaded files to temporary location.
    
    Args:
        uploaded_files: List of uploaded file objects (from Streamlit)
        
    Returns:
        list: List of paths to saved files
    """
    temp_paths = []
    
    for uploaded_file in uploaded_files:
        # Create temp file with same extension
        file_extension = os.path.splitext(uploaded_file.name)[1]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        
        # Write content
        temp_file.write(uploaded_file.getvalue())
        temp_file.close()
        
        # Add to list
        temp_paths.append(Path(temp_file.name))
        
        logger.info(f"Saved uploaded file '{uploaded_file.name}' to '{temp_file.name}'")
    
    return temp_paths

def clear_temp_files(temp_paths: List[Path]) -> None:
    """
    Delete temporary files after processing.
    
    Args:
        temp_paths: List of temporary file paths
    """
    if not temp_paths:
        return
        
    logger.info(f"Cleaning up {len(temp_paths)} temporary files")
    
    for path in temp_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Removed temporary file: {path}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {path}: {e}")
