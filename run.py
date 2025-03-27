#!/usr/bin/env python3
"""
Run script for the Anti-Corruption RAG system.
"""
import os
import subprocess
import sys
import time
import argparse
import webbrowser
from pathlib import Path

def check_qdrant_status(host="localhost", port=6334):
    """Check if Qdrant is running."""
    import requests
    try:
        response = requests.get(f"http://{host}:{port}/healthz")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def start_qdrant():
    """Start Qdrant with Docker if not already running."""
    # Check if Qdrant is already running
    if check_qdrant_status():
        print("✅ Qdrant is already running")
        return True
    
    print("Starting Qdrant with Docker...")
    
    # Create data directory if it doesn't exist
    data_dir = Path(__file__).resolve().parent / "data" / "qdrant_data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Start Qdrant container
    try:
        cmd = [
            "docker", "run", "-d",
            "--name", "qdrant-server",
            "-p", "6333:6333",
            "-p", "6334:6334",
            "-v", f"{data_dir}:/qdrant/storage",
            "qdrant/qdrant:latest"
        ]
        
        subprocess.run(cmd, check=True)
        
        # Wait for Qdrant to start
        print("Waiting for Qdrant to start...")
        for _ in range(10):
            if check_qdrant_status():
                print("✅ Qdrant is now running")
                return True
            time.sleep(2)
        
        print("❌ Qdrant failed to start in time")
        return False
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting Qdrant: {e}")
        
        # Check if container already exists but is stopped
        try:
            subprocess.run(["docker", "start", "qdrant-server"], check=True)
            print("✅ Started existing Qdrant container")
            return True
        except subprocess.CalledProcessError:
            print("❌ Could not start existing Qdrant container")
            return False
    
    except Exception as e:
        print(f"❌ Unexpected error starting Qdrant: {e}")
        return False

def start_streamlit():
    """Start the Streamlit application."""
    print("Starting Streamlit application...")
    
    streamlit_path = Path(__file__).resolve().parent / "src" / "ui" / "app.py"
    
    # Open the URL in a browser
    url = "http://localhost:8501"
    webbrowser.open(url)
    
    # Start Streamlit
    cmd = ["streamlit", "run", str(streamlit_path)]
    
    try:
        # Use subprocess.run for Python 3.5+
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nStopping application...")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run the Anti-Corruption RAG system")
    parser.add_argument("--no-qdrant", action="store_true", help="Don't start Qdrant")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    
    args = parser.parse_args()
    
    # Start Qdrant if needed
    if not args.no_qdrant:
        if not start_qdrant():
            print("❌ Failed to start Qdrant. Use --no-qdrant to skip this step.")
            return
    
    # Disable browser opening if requested
    if args.no_browser:
        os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
        os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    
    # Start Streamlit
    start_streamlit()

if __name__ == "__main__":
    main()
