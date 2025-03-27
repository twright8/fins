#!/usr/bin/env python3
"""
Utility script to clear model cache and temporary files.
This can be useful when you want to force re-downloading models
or resolve cache-related issues.
"""
import os
import shutil
import sys
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def get_cache_dirs():
    """
    Get standard cache directories used by the system.
    
    Returns:
        dict: Dictionary of cache directories and their descriptions
    """
    home = Path.home()
    
    return {
        "huggingface": home / ".cache" / "huggingface",
        "torch": home / ".cache" / "torch",
        "flair": home / ".flair",
        "transformers": home / ".cache" / "transformers",
        "tmp_data": Path(__file__).resolve().parent / "temp"
    }

def clear_cache(cache_type="all", dry_run=False):
    """
    Clear specified cache type.
    
    Args:
        cache_type (str): Type of cache to clear (all, huggingface, torch, flair, transformers, tmp)
        dry_run (bool): If True, only print what would be deleted without actually deleting
    """
    cache_dirs = get_cache_dirs()
    
    if cache_type == "all":
        targets = cache_dirs
    elif cache_type == "tmp":
        targets = {"tmp_data": cache_dirs["tmp_data"]}
    elif cache_type in cache_dirs:
        targets = {cache_type: cache_dirs[cache_type]}
    else:
        logger.error(f"Unknown cache type: {cache_type}")
        return False
    
    success = True
    
    for name, path in targets.items():
        try:
            if path.exists():
                size = get_directory_size(path)
                size_str = format_size(size)
                
                if dry_run:
                    logger.info(f"Would delete {name} cache: {path} ({size_str})")
                else:
                    logger.info(f"Deleting {name} cache: {path} ({size_str})")
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                    logger.info(f"Successfully cleared {name} cache")
            else:
                logger.info(f"{name.capitalize()} cache not found at {path}")
        except Exception as e:
            logger.error(f"Error clearing {name} cache: {e}")
            success = False
    
    return success

def get_directory_size(path):
    """
    Get the total size of a directory in bytes.
    
    Args:
        path (Path): Path to the directory
        
    Returns:
        int: Size in bytes
    """
    total_size = 0
    
    if not path.exists():
        return 0
    
    if path.is_file():
        return path.stat().st_size
    
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_path = Path(dirpath) / filename
            if not file_path.is_symlink():
                total_size += file_path.stat().st_size
    
    return total_size

def format_size(size_bytes):
    """
    Format bytes to human-readable format.
    
    Args:
        size_bytes (int): Size in bytes
        
    Returns:
        str: Human-readable size
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ("B", "KB", "MB", "GB", "TB")
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024
        i += 1
    
    return f"{size_bytes:.2f} {size_names[i]}"

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Clear model cache and temporary files")
    parser.add_argument(
        "--cache", 
        choices=["all", "huggingface", "torch", "flair", "transformers", "tmp"], 
        default="all",
        help="Type of cache to clear"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print(" Model Cache Clearing Utility ".center(60, "="))
    print("=" * 60 + "\n")
    
    if args.dry_run:
        print("DRY RUN MODE: No files will actually be deleted\n")
    
    # Print current cache sizes
    cache_dirs = get_cache_dirs()
    print("Current cache sizes:")
    for name, path in cache_dirs.items():
        size = get_directory_size(path)
        print(f"  - {name.ljust(12)}: {format_size(size).rjust(10)} | {path}")
    print()
    
    # Confirm if not dry-run
    if not args.dry_run:
        confirm = input(f"Are you sure you want to clear the {args.cache} cache? This cannot be undone. (y/N): ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            return
    
    # Clear the cache
    success = clear_cache(args.cache, args.dry_run)
    
    print("\n" + "=" * 60)
    if success:
        if args.dry_run:
            print(" Cache clearing simulation completed successfully ".center(60, "="))
        else:
            print(" Cache cleared successfully ".center(60, "="))
    else:
        print(" Some errors occurred during cache clearing ".center(60, "="))
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
