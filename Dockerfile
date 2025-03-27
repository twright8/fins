FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set cache directories
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
ENV SENTENCE_TRANSFORMERS_HOME=/root/.cache/torch/sentence_transformers
ENV TORCH_HOME=/root/.cache/torch

# Set other environment variables for better logging
ENV TRANSFORMERS_VERBOSITY=info
ENV HF_HUB_DISABLE_PROGRESS_BARS=0
ENV TOKENIZERS_PARALLELISM=true

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    wget \
    git \
    curl \
    tesseract-ocr \
    libtesseract-dev \
    lsb-release \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt &&\
    pip install --no-cache-dir maverick-coref --no-deps


# Copy app code
COPY . .

# Create directories
RUN mkdir -p data/bm25_indices data/extracted data/ocr_cache logs temp

# Set permissions
RUN chmod -R 777 data logs temp

# Expose port for Streamlit
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
