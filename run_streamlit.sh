#!/bin/bash

# Run script for Streamlit with enhanced logging
# This ensures that all Python output is unbuffered,
# Hugging Face progress bars are shown, and various
# libraries output at INFO level for better visibility

# Set environment variables for better logging
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_PROGRESS_BARS=0
export TRANSFORMERS_VERBOSITY=info
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8

# Run Streamlit with additional flags
echo "Starting Streamlit with enhanced logging..."
echo "All output will be displayed in real-time"
echo "Press Ctrl+C to stop"
echo

# Start Streamlit
streamlit run src/ui/app.py --server.runOnSave=true
