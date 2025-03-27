#!/bin/bash

# Script to run the application in local development mode

# Create necessary directories
mkdir -p data/bm25_indices data/extracted data/ocr_cache logs temp

# Check if Qdrant is running locally or with Docker
if curl -s http://localhost:6334/healthz > /dev/null; then
    echo "✅ Qdrant is already running"
else
    echo "Starting Qdrant with Docker..."
    
    # Create Qdrant data directory
    mkdir -p data/qdrant_data
    
    # Check if Qdrant container exists
    if docker ps -a | grep -q qdrant-server; then
        # Start existing container if it exists
        docker start qdrant-server
    else
        # Run new container
        docker run -d \
            --name qdrant-server \
            -p 6333:6333 \
            -p 6334:6334 \
            -v "$(pwd)/data/qdrant_data:/qdrant/storage" \
            qdrant/qdrant:latest
    fi
    
    # Wait for Qdrant to start
    echo "Waiting for Qdrant to start..."
    for i in {1..10}; do
        if curl -s http://localhost:6334/healthz > /dev/null; then
            echo "✅ Qdrant is now running"
            break
        fi
        sleep 2
    done
fi

# Start the Streamlit application
echo "Starting Streamlit application..."
streamlit run src/ui/app.py
