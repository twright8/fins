"""
Test script to validate the threading fix.
"""
import threading
import time

def test_thread_function():
    """Test thread function to verify threading works."""
    print("Thread started")
    time.sleep(1)
    print("Thread finished")

def get_info():
    """Function similar to the one that had the error."""
    try:
        print("Starting get_info function")
        # Simulate some work
        time.sleep(0.5)
        print("Completed get_info function")
        return {"status": "success"}
    except Exception as e:
        print(f"Error in get_info: {e}")
        return {"status": "error", "message": str(e)}

def main():
    """Main function to test threading."""
    print("Starting test...")
    
    # Test basic threading first
    t1 = threading.Thread(target=test_thread_function)
    t1.start()
    t1.join()
    
    # Now test the function similar to what had the error
    print("Testing get_info with threading...")
    result = [None]
    
    def thread_get_info():
        try:
            result[0] = get_info()
        except Exception as e:
            print(f"Thread error: {e}")
    
    t2 = threading.Thread(target=thread_get_info)
    t2.start()
    t2.join()
    
    print(f"Result: {result[0]}")
    print("Test completed.")

if __name__ == "__main__":
    main()
