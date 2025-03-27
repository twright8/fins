#!/usr/bin/env python3
"""
Testing script for the Anti-Corruption RAG system components.
Use this script to test individual components of the system.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.config import CONFIG
from src.processing.document_loader import DocumentLoader
from src.processing.document_chunker import DocumentChunker
from src.processing.coreference_resolver import CoreferenceResolver
from src.processing.entity_extractor import EntityExtractor
from src.processing.indexer import Indexer
from src.core.retriever import Retriever
from src.core.generator import Generator
from src.core.query_handler import QueryHandler

def test_document_loader(file_path):
    """
    Test the document loader component.
    
    Args:
        file_path: Path to a document to load
    """
    print(f"Testing document loader with file: {file_path}")
    
    loader = DocumentLoader()
    
    start_time = time.time()
    document = loader.load_document(file_path)
    elapsed_time = time.time() - start_time
    
    print(f"Document loaded in {elapsed_time:.2f} seconds")
    print(f"Document ID: {document.get('document_id', 'Unknown')}")
    print(f"File type: {document.get('file_type', 'Unknown')}")
    
    content = document.get('content', [])
    print(f"Content items: {len(content)}")
    
    metadata = document.get('metadata', {})
    print("Metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    # Print a sample of content
    if content:
        print("\nSample content:")
        first_content = content[0].get('text', '')
        print(first_content[:500] + "..." if len(first_content) > 500 else first_content)
    
    return document

def test_document_chunking(document):
    """
    Test the document chunker component.
    
    Args:
        document: Document data from document loader
    """
    print("\nTesting document chunker")
    
    chunker = DocumentChunker()
    
    start_time = time.time()
    chunks = chunker.chunk_document(document)
    elapsed_time = time.time() - start_time
    
    print(f"Document chunked in {elapsed_time:.2f} seconds")
    print(f"Created {len(chunks)} chunks")
    
    # Print statistics
    chunk_sizes = [len(chunk.get('text', '')) for chunk in chunks]
    avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
    
    print(f"Average chunk size: {avg_size:.1f} characters")
    print(f"Min chunk size: {min(chunk_sizes) if chunk_sizes else 0} characters")
    print(f"Max chunk size: {max(chunk_sizes) if chunk_sizes else 0} characters")
    
    # Print a sample chunk
    if chunks:
        print("\nSample chunk:")
        first_chunk = chunks[0].get('text', '')
        print(first_chunk[:500] + "..." if len(first_chunk) > 500 else first_chunk)
    
    return chunks

def test_coreference_resolution(chunks):
    """
    Test the coreference resolution component.
    
    Args:
        chunks: List of document chunks
    """
    print("\nTesting coreference resolution")
    
    resolver = CoreferenceResolver()
    
    start_time = time.time()
    resolved_chunks = resolver.process_chunks(chunks[:5])  # Process a few chunks for demonstration
    elapsed_time = time.time() - start_time
    
    print(f"Coreference resolution applied in {elapsed_time:.2f} seconds")
    print(f"Processed {len(resolved_chunks)} chunks")
    
    # Print a sample before and after
    if resolved_chunks:
        print("\nSample before and after coreference resolution:")
        original = chunks[0].get('text', '')
        resolved = resolved_chunks[0].get('text', '')
        
        print("Original:")
        print(original[:500] + "..." if len(original) > 500 else original)
        print("\nResolved:")
        print(resolved[:500] + "..." if len(resolved) > 500 else resolved)
    
    return resolved_chunks

def test_entity_extraction(chunks):
    """
    Test the entity extraction component.
    
    Args:
        chunks: List of document chunks
    """
    print("\nTesting entity extraction")
    
    extractor = EntityExtractor()
    
    start_time = time.time()
    processed_chunks, entities, relationships = extractor.process_chunks(chunks[:5])  # Process a few chunks for demonstration
    elapsed_time = time.time() - start_time
    
    print(f"Entity extraction completed in {elapsed_time:.2f} seconds")
    print(f"Extracted {len(entities)} entities and {len(relationships)} relationships")
    
    # Print entity types and counts
    entity_types = {}
    for entity in entities:
        entity_type = entity.get('type', 'unknown')
        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
    
    print("\nEntity types:")
    for entity_type, count in entity_types.items():
        print(f"  {entity_type}: {count}")
    
    # Print relationship types and counts
    relationship_types = {}
    for rel in relationships:
        rel_type = rel.get('type', 'unknown')
        relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
    
    print("\nRelationship types:")
    for rel_type, count in relationship_types.items():
        print(f"  {rel_type}: {count}")
    
    # Print some sample entities
    if entities:
        print("\nSample entities:")
        for entity in entities[:5]:
            print(f"  {entity.get('text', 'Unknown')} ({entity.get('type', 'unknown')}): {entity.get('confidence', 0):.3f}")
    
    # Print some sample relationships
    if relationships:
        print("\nSample relationships:")
        for rel in relationships[:5]:
            subject = rel.get('subject', 'Unknown')
            rel_type = rel.get('type', 'unknown')
            obj = rel.get('object', 'Unknown')
            print(f"  {subject} --[{rel_type}]--> {obj}: {rel.get('confidence', 0):.3f}")
    
    return processed_chunks, entities, relationships

def test_indexing(chunks):
    """
    Test the indexing component.
    
    Args:
        chunks: List of document chunks
    """
    print("\nTesting indexing")
    
    indexer = Indexer()
    
    start_time = time.time()
    success = indexer.index_chunks(chunks[:20])  # Index a subset of chunks for demonstration
    elapsed_time = time.time() - start_time
    
    print(f"Indexing completed in {elapsed_time:.2f} seconds")
    print(f"Indexing success: {success}")
    
    return success

def test_retrieval(query):
    """
    Test the retrieval component.
    
    Args:
        query: Query string
    """
    print(f"\nTesting retrieval with query: '{query}'")
    
    retriever = Retriever()
    
    start_time = time.time()
    results = retriever.retrieve(query)
    elapsed_time = time.time() - start_time
    
    print(f"Retrieval completed in {elapsed_time:.2f} seconds")
    print(f"Retrieved {len(results)} results")
    
    # Print some sample results
    if results:
        print("\nTop results:")
        for i, result in enumerate(results[:3]):
            print(f"\nResult {i+1} (score: {result.get('score', 0):.3f}):")
            
            # Print metadata
            metadata = result.get('metadata', {})
            if metadata:
                print(f"  Document: {metadata.get('file_name', 'Unknown')}")
                if 'page_num' in metadata:
                    print(f"  Page: {metadata.get('page_num')}")
            
            # Print text sample
            text = result.get('text', '')
            print(f"  Text: {text[:300]}..." if len(text) > 300 else f"  Text: {text}")
    
    return results

def test_generation(query, context=None):
    """
    Test the generation component.
    
    Args:
        query: Query string
        context: Optional context for generation
    """
    print(f"\nTesting generation with query: '{query}'")
    
    generator = Generator()
    
    # Load the model
    if not generator._load_model():
        print("Failed to load model")
        return None
    
    # Generate response
    start_time = time.time()
    
    if context:
        response = generator.generate_with_context(query, context)
    else:
        # Create a simple prompt
        prompt = f"Answer the following question concisely: {query}"
        response = generator.generate(prompt)
    
    elapsed_time = time.time() - start_time
    
    print(f"Generation completed in {elapsed_time:.2f} seconds")
    print(f"Generated {len(response)} characters")
    
    print("\nGenerated response:")
    print(response)
    
    return response

def test_full_query(query):
    """
    Test the full query pipeline.
    
    Args:
        query: Query string
    """
    print(f"\nTesting full query pipeline with: '{query}'")
    
    query_handler = QueryHandler()
    
    start_time = time.time()
    answer, context = query_handler.process_query(query)
    elapsed_time = time.time() - start_time
    
    print(f"Query processing completed in {elapsed_time:.2f} seconds")
    print(f"Retrieved {len(context)} context chunks")
    
    print("\nGenerated answer:")
    print(answer)
    
    return answer, context

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test Anti-Corruption RAG components")
    parser.add_argument("--document", "-d", help="Path to a document to process")
    parser.add_argument("--query", "-q", help="Query to test retrieval and generation")
    parser.add_argument("--component", "-c", choices=[
        "loader", "chunker", "coref", "entity", "indexer", "retriever", "generator", "query", "all"
    ], default="all", help="Component to test")
    
    args = parser.parse_args()
    
    # Test specified component
    if args.component in ["loader", "chunker", "coref", "entity", "indexer", "all"] and not args.document:
        parser.error("--document is required for these component tests")
    
    if args.component in ["retriever", "generator", "query", "all"] and not args.query:
        parser.error("--query is required for these component tests")
    
    # Run the tests
    if args.component == "loader" or args.component == "all":
        document = test_document_loader(args.document)
    
    if args.component == "chunker" or args.component == "all":
        if args.component == "chunker":
            document = test_document_loader(args.document)
        
        chunks = test_document_chunking(document)
    
    if args.component == "coref" or args.component == "all":
        if args.component == "coref":
            document = test_document_loader(args.document)
            chunks = test_document_chunking(document)
        
        resolved_chunks = test_coreference_resolution(chunks)
    
    if args.component == "entity" or args.component == "all":
        if args.component == "entity":
            document = test_document_loader(args.document)
            chunks = test_document_chunking(document)
            resolved_chunks = test_coreference_resolution(chunks)
        
        processed_chunks, entities, relationships = test_entity_extraction(resolved_chunks)
    
    if args.component == "indexer" or args.component == "all":
        if args.component == "indexer":
            document = test_document_loader(args.document)
            chunks = test_document_chunking(document)
        
        success = test_indexing(chunks)
    
    if args.component == "retriever" or args.component == "all":
        results = test_retrieval(args.query)
    
    if args.component == "generator" or args.component == "all":
        if args.component == "generator" and args.component != "all":
            # Test with simple prompt
            test_generation(args.query)
        elif args.component == "all":
            # Use retrieved results as context
            test_generation(args.query, results)
    
    if args.component == "query" or args.component == "all":
        answer, context = test_full_query(args.query)

if __name__ == "__main__":
    main()
