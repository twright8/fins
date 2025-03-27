# Anti-Corruption RAG System

A modular Retrieval-Augmented Generation (RAG) system for anti-corruption document analysis. The system enables users to upload documents, process them to extract entities and relationships, and query the information using natural language.

## Features

- Document processing with support for PDF, Word, TXT, CSV/XLSX files
- Semantic chunking with coreference resolution
- Named entity recognition and relationship extraction
- Hybrid search combining BM25 and vector embeddings
- Conversational interface powered by LLM
- Entity visualization and exploration
- Dockerized deployment with GPU support

## System Architecture

The system consists of the following main components:

1. **Document Processing Pipeline**
   - Document loading and parsing
   - Semantic chunking
   - Coreference resolution
   - Entity extraction and relationship classification
   - BM25 and vector indexing

2. **Retrieval System**
   - Hybrid search (BM25 + vector)
   - Reranking
   - Context fusion

3. **Generation System**
   - LLM-based response generation
   - Context-aware answers

4. **User Interface**
   - Document upload and processing
   - Document exploration
   - Entity visualization
   - Conversational query interface

## Technical Stack

- **Backend**: Python, Flair, Maverick, Infinity Embeddings, Qdrant, Aphrodite
- **Frontend**: Streamlit
- **Infrastructure**: Docker, CUDA

## Installation

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/anti-corruption-rag.git
cd anti-corruption-rag
```

2. Start the application using Docker Compose:

```bash
docker-compose up -d
```

This will start the Qdrant vector database and the main application.

3. Access the web interface at http://localhost:8501

### Local Development Setup

If you prefer to run the application locally without Docker:

1. Install the requirements:

```bash
pip install -r requirements.txt
```

2. Start Qdrant (you can use Docker for this part only):

```bash
docker run -d -p 6333:6333 -p 6334:6334 -v $(pwd)/data/qdrant_data:/qdrant/storage qdrant/qdrant
```

3. Run the Streamlit application:

```bash
streamlit run src/ui/app.py
```

## Usage

1. **Upload Documents**
   - Click on "Upload & Process" in the sidebar
   - Upload one or more documents (PDF, DOCX, TXT, CSV, XLSX)
   - Click "Process Documents" to start analysis

2. **Explore Documents**
   - After processing, navigate to "Explore Documents" to view document content
   - Explore extracted entities and relationships

3. **Query Documents**
   - Use the "Query Documents" section to ask questions about your uploaded documents
   - View source evidence for the generated answers

## License

This project is licensed under the MIT License - see the LICENSE file for details.
