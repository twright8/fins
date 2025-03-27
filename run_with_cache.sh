#!/bin/bash
# This script runs the Streamlit app with explicitly configured cache directories
# to ensure consistent caching behavior when running locally

# Define cache directories
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers
export SENTENCE_TRANSFORMERS_HOME=~/.cache/torch/sentence_transformers
export TORCH_HOME=~/.cache/torch

# Ensure cache directories exist
mkdir -p $HF_HOME
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $SENTENCE_TRANSFORMERS_HOME
mkdir -p $TORCH_HOME

# Set other environment variables for better logging and performance
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=info
export HF_HUB_DISABLE_PROGRESS_BARS=0
export TOKENIZERS_PARALLELISM=true

# Print cache locations
echo "=== Cache Directories ==="
echo "HF_HOME: $HF_HOME"
echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "SENTENCE_TRANSFORMERS_HOME: $SENTENCE_TRANSFORMERS_HOME"
echo "TORCH_HOME: $TORCH_HOME"
echo "=========================="

# Run the Streamlit app
streamlit run src/ui/app.py "$@"
