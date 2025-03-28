"""
Test script to verify Qdrant client functionality.
"""
import sys
from pathlib import Path

# Add parent directory to sys.path to enable imports from project root
sys.path.append(str(Path(__file__).resolve().parent))
from src.core.config import CONFIG

def test_qdrant_connection():
    """
    Test connection to Qdrant and basic filtering.
    """
    try:
        print("Importing Qdrant client...")
        from qdrant_client import QdrantClient, models
        
        print("Connecting to Qdrant...")
        client = QdrantClient(
            host=CONFIG["qdrant"]["host"],
            port=CONFIG["qdrant"]["port"]
        )
        
        print("Getting collections...")
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        print(f"Available collections: {collection_names}")
        
        if CONFIG["qdrant"]["collection_name"] in collection_names:
            print(f"Found target collection: {CONFIG['qdrant']['collection_name']}")
            
            # Get collection info
            collection_info = client.get_collection(CONFIG["qdrant"]["collection_name"])
            print(f"Collection info: {collection_info.points_count} points, vector size: {collection_info.config.params.vectors.size}")
            
            # Try a simple scroll without filter
            print("Testing scroll without filter...")
            scroll_result = client.scroll(
                collection_name=CONFIG["qdrant"]["collection_name"],
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            
            points = scroll_result[0]
            print(f"Retrieved {len(points)} points")
            
            if points:
                print(f"Sample point ID: {points[0].id}")
                print(f"Sample payload keys: {list(points[0].payload.keys())}")
            
            # Now try with a filter
            print("Testing scroll with filter...")
            filter_conditions = []
            
            # Create a simple filter
            filter_obj = models.Filter(
                must=[
                    models.FieldCondition(
                        key="text",
                        match=models.MatchValue(value="the")
                    )
                ]
            )
            
            scroll_result = client.scroll(
                collection_name=CONFIG["qdrant"]["collection_name"],
                limit=1,
                with_payload=True,
                with_vectors=False,
                filter=filter_obj
            )
            
            points = scroll_result[0]
            print(f"Retrieved {len(points)} points with filter")
            
            if points:
                print(f"Sample filtered point ID: {points[0].id}")
                
            print("Test completed successfully!")
        else:
            print(f"Target collection {CONFIG['qdrant']['collection_name']} not found.")
            print("Please process documents first to create the collection.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    test_qdrant_connection()
