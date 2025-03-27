"""
File utility functions for the Anti-Corruption RAG system.
"""
import os
import tempfile
import shutil
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, BinaryIO

import streamlit as st

# Add parent directory to sys.path to enable imports from project root
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.core.config import CONFIG, TEMP_DIR
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def save_uploaded_file(uploaded_file: BinaryIO, destination_dir: Optional[Path] = None) -> Path:
    """
    Save an uploaded file to a temporary location and return the path.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        destination_dir: Directory to save the file (defaults to TEMP_DIR)
        
    Returns:
        Path: Path to the saved file
    """
    if destination_dir is None:
        destination_dir = TEMP_DIR
    
    # Ensure directory exists
    os.makedirs(destination_dir, exist_ok=True)
    
    # Get original file extension
    file_ext = Path(uploaded_file.name).suffix
    
    # Create destination path with unique name
    unique_id = uuid.uuid4().hex
    dest_path = destination_dir / f"{unique_id}{file_ext}"
    
    # Save the file
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    logger.info(f"Saved uploaded file '{uploaded_file.name}' to '{dest_path}'")
    
    return dest_path

def save_uploaded_files(uploaded_files: List[BinaryIO]) -> List[Path]:
    """
    Save multiple uploaded files to temporary locations.
    
    Args:
        uploaded_files: List of Streamlit uploaded file objects
        
    Returns:
        list: List of paths to saved files
    """
    return [save_uploaded_file(f) for f in uploaded_files]

def clear_temp_files(file_paths: List[Path] = None):
    """
    Clear temporary files.
    
    Args:
        file_paths: List of file paths to clear (defaults to all files in TEMP_DIR)
    """
    if file_paths:
        # Clear specific files
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed temporary file: {file_path}")
            except Exception as e:
                logger.error(f"Error removing file {file_path}: {e}")
    else:
        # Clear all files in temp directory
        try:
            if os.path.exists(TEMP_DIR):
                for filename in os.listdir(TEMP_DIR):
                    file_path = os.path.join(TEMP_DIR, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        
                logger.info(f"Cleared all files in {TEMP_DIR}")
        except Exception as e:
            logger.error(f"Error clearing temp directory: {e}")

def is_valid_file_type(file_path: str) -> bool:
    """
    Check if a file has a valid extension for processing.
    
    Args:
        file_path: Path to the file
        
    Returns:
        bool: True if the file has a valid extension, False otherwise
    """
    valid_extensions = ['.pdf', '.docx', '.doc', '.xlsx', '.xls', '.csv', '.txt']
    file_extension = Path(file_path).suffix.lower()
    
    return file_extension in valid_extensions

def get_file_size_mb(file_path: Path) -> float:
    """
    Get the size of a file in megabytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        float: Size of the file in MB
    """
    return os.path.getsize(file_path) / (1024 * 1024)

def check_file_size_limit(file_path: Path, max_size_mb: Optional[float] = None) -> bool:
    """
    Check if a file is within the size limit.
    
    Args:
        file_path: Path to the file
        max_size_mb: Maximum file size in MB (defaults to CONFIG value)
        
    Returns:
        bool: True if the file is within the limit, False otherwise
    """
    if max_size_mb is None:
        max_size_mb = CONFIG["ui"]["max_upload_size_mb"]
    
    file_size_mb = get_file_size_mb(file_path)
    
    return file_size_mb <= max_size_mb
