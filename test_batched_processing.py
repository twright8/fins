#!/usr/bin/env python3
"""
Test script for the batched processing implementation.
"""
import sys
import os
import time
import uuid
from pathlib import Path
import multiprocessing

# Add parent directory to sys.path to enable imports from project root
sys.path.append(str(Path(__file__).resolve().parent))
from src.core.config import CONFIG, DATA_DIR
from src.processing.coreference_resolver import CoreferenceResolver
from src.processing.entity_extractor import EntityExtractor

def test_batch_processing():
    """
    Test the batched processing implementation.
    """
    print("=" * 80)
    print("Testing Batched Processing".center(80))
    print("=" * 80)
    
    # Create a test document with multiple chunks
    chunks = []
    
    # Create some test chunks
    test_texts = [
        "Jason is 30 years old and earns thirty thousand pounds per year. "
        "He lives in London and works as a Software Engineer. His employee ID is 12345.",
        
        "Susan, aged 9, does not have a job and earns zero pounds. "
        "She is a student in elementary school and lives in Manchester.",
        
        "Michael, a 45-year-old Doctor, earns ninety-five thousand pounds annually. "
        "He resides in Birmingham and has an employee ID of 67890.",
        
        "Emily is a 28-year-old Data Scientist who earns seventy-two thousand pounds. "
        "She is based in Edinburgh and her employee ID is 54321.",
        
        "John Smith and his wife Jane visited Paris last summer. "
        "They enjoyed the trip and plan to return next year with their children.",
        
        "Apple Inc. was founded by Steve Jobs and Steve Wozniak in 1976. "
        "It is now one of the largest technology companies in the world based in Cupertino, California.",
        
        "The United Nations was established in 1945 after World War II. "
        "It aims to maintain international peace and security. "
        "The organization's headquarters is in New York City.",
        
        "Tesla, the electric vehicle manufacturer, was named after Nikola Tesla. "
        "It was founded by Elon Musk, JB Straubel, and others. "
        "The company produces electric cars, battery storage, and solar products."
    ]
    
    # Create chunks with appropriate metadata
    for i, text in enumerate(test_texts):
        chunks.append({
            'chunk_id': f"chunk_{i}",
            'document_id': 'test_doc',
            'text': text,
            'file_name': 'test_document.txt',
            'page_num': i // 2 + 1,  # 2 chunks per page
        })
    
    # Duplicate the chunks to create a larger test set
    chunks = chunks * 2  # 16 chunks
    
    # Test different batch sizes
    batch_sizes = [1, 4, 8, 16]
    
    # Test CoreferenceResolver with different batch sizes
    print("\nTesting CoreferenceResolver with different batch sizes:")
    print("-" * 80)
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Create a CoreferenceResolver instance
        coref_resolver = CoreferenceResolver()
        
        # Process chunks with timing
        start_time = time.time()
        resolved_chunks = coref_resolver.process_chunks(chunks, batch_size=batch_size)
        elapsed_time = time.time() - start_time
        
        print(f"Processed {len(chunks)} chunks in {elapsed_time:.2f} seconds")
        print(f"Average time per chunk: {elapsed_time / len(chunks):.4f} seconds")
        
        # Print sample of resolved text
        print("\nSample resolved text:")
        print(resolved_chunks[0]['text'][:200] + "...")
    
    # Test EntityExtractor
    print("\n\nTesting EntityExtractor with batch processing:")
    print("-" * 80)
    
    # Create an EntityExtractor instance
    entity_extractor = EntityExtractor()
    
    # Process chunks with timing
    start_time = time.time()
    processed_chunks, entities, relationships = entity_extractor.process_chunks(chunks)
    elapsed_time = time.time() - start_time
    
    print(f"Processed {len(chunks)} chunks in {elapsed_time:.2f} seconds")
    print(f"Average time per chunk: {elapsed_time / len(chunks):.4f} seconds")
    print(f"Extracted {len(entities)} unique entities and {len(relationships)} relationships")
    
    # Print sample of entities
    print("\nSample entities:")
    for i, entity in enumerate(entities[:5]):
        print(f"{i+1}. {entity['text']} ({entity['type']}) - Confidence: {entity['confidence']:.4f}")
    
    # Print sample of relationships
    if relationships:
        print("\nSample relationships:")
        for i, rel in enumerate(relationships[:5]):
            print(f"{i+1}. {rel['subject']} -> {rel['type']} -> {rel['object']} - Confidence: {rel['confidence']:.4f}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    # Ensure clean CUDA initialization for multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, continue
        pass
    
    test_batch_processing()
