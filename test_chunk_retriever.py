"""
Test script to verify the fix for the threading issue in the chunk retriever.
"""
import threading
import time
import sys
from pathlib import Path

# Add parent directory to sys.path to enable imports from project root
sys.path.append(str(Path(__file__).resolve().parent))
from src.core.config import CONFIG

# Function similar to QdrantRetriever.get_collection_info in app.py
def get_collection_info():
    try:
        print("Getting collection info...")
        # Simulate work
        time.sleep(1)
        return {
            'exists': True,
            'name': 'test_collection',
            'points_count': 100,
            'vector_size': 768,
            'distance': 'cosine'
        }
    except Exception as e:
        print(f"Error getting collection info: {e}")
        return {'exists': False, 'error': str(e)}

def main():
    """Test functions that mimic the problematic code."""
    print("Starting test...")
    
    # Test with threading
    print("Testing get_collection_info with threading...")
    collection_info_result = [None]
    error_result = [None]
    
    def thread_get_info():
        try:
            collection_info_result[0] = get_collection_info()
        except Exception as e:
            error_result[0] = str(e)
            print(f"Error getting collection info: {e}")
    
    # Start thread to get collection info
    thread = threading.Thread(target=thread_get_info)
    thread.start()
    thread.join(timeout=5.0)  # 5 second timeout
    
    # Check results
    if error_result[0] is not None:
        print(f"Got error: {error_result[0]}")
    elif collection_info_result[0] is None:
        print("Operation timed out")
    else:
        print(f"Result: {collection_info_result[0]}")
    
    print("Test completed.")

if __name__ == "__main__":
    main()
