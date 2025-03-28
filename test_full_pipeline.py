#!/usr/bin/env python3
"""
Test script for the full document processing pipeline with batched implementations.
"""
import sys
import os
import time
import multiprocessing
from pathlib import Path
import traceback

# Add parent directory to sys.path to enable imports from project root
sys.path.append(str(Path(__file__).resolve().parent))
from src.core.config import CONFIG, DATA_DIR
from src.processing.pipeline import process_documents

def test_pipeline():
    """
    Test the full document processing pipeline with a sample document.
    """
    print("=" * 80)
    print("Testing Full Document Processing Pipeline with Batched Implementation".center(80))
    print("=" * 80)
    
    # Create a sample document
    sample_file = Path(__file__).parent / "temp" / "sample_document.txt"
    os.makedirs(sample_file.parent, exist_ok=True)
    
    # Create sample content with named entities and relationships
    sample_content = """# Anti-Corruption Investigation Report

## Executive Summary

This document outlines a comprehensive investigation into alleged corruption at Acme Corporation.
The investigation was led by John Smith, Director of Internal Audit, with assistance from the legal team
headed by Sarah Johnson.

## Timeline of Events

On January 15, 2023, an anonymous whistleblower contacted the Ethics Hotline to report suspicious
financial transactions involving Michael Williams, the Vice President of Procurement.
The whistleblower alleged that Williams was accepting bribes from Atlas Suppliers Ltd., a company
based in London, UK, owned by Richard Davies.

Internal Auditor James Chen conducted a preliminary investigation between February 1-15, 2023.
His findings indicated that procurement contracts worth $3.5 million were awarded to Atlas Suppliers
without proper competitive bidding procedures.

## Financial Analysis

Financial records show payments from Acme to Atlas Suppliers Ltd. increased by 240% in fiscal year 2022-2023.
The records also revealed unusual consulting fees paid by Atlas to offshore company Blue Ocean Ventures Inc.,
registered in the Cayman Islands.

The audit trail indicates that Blue Ocean Ventures transferred approximately $250,000 to a private account
at United Bank in Switzerland, allegedly connected to Michael Williams through his wife, Jennifer Williams.

## Key Individuals Involved

1. Michael Williams - VP of Procurement, Acme Corporation
2. Richard Davies - CEO, Atlas Suppliers Ltd.
3. Jennifer Williams - Spouse of Michael Williams
4. Maria Rodriguez - CFO, Acme Corporation (flagged unusual payments but was overruled)
5. Thomas Barker - Legal Counsel, Atlas Suppliers Ltd.

## Evidence Collection

The investigation team, led by FBI Agent Lisa Thompson, collected 234 emails, 56 financial documents,
and testimony from 12 witnesses. Digital forensics conducted by tech expert David Zhang recovered
deleted communications between Williams and Davies discussing "special arrangements" and "consulting fees."

The team also conducted interviews with Acme employees in London, New York, and Singapore offices.

## Recommendations

Based on the findings, we recommend:

1. Immediate termination of Michael Williams
2. Legal action against Atlas Suppliers Ltd. for fraud
3. Recovery of misappropriated funds
4. Implementation of enhanced procurement controls
5. Full cooperation with authorities in the UK and US

## Submitted by

The report was prepared by the Special Investigation Team:
- John Smith, Director of Internal Audit
- Sarah Johnson, Head of Legal
- James Chen, Senior Auditor
- Lisa Thompson, FBI Liaison
- David Zhang, Digital Forensics Expert

Submitted to the Board of Directors on March 30, 2023.
"""
    
    # Write sample document to file
    with open(sample_file, 'w') as f:
        f.write(sample_content)
    
    print(f"Created sample document: {sample_file}")
    
    # Create a status queue for communication
    status_queue = multiprocessing.Queue()
    
    # Define a queue listener function
    def queue_listener(queue):
        while True:
            try:
                msg = queue.get()
                
                if msg[0] == 'end':
                    break
                
                if msg[0] == 'progress':
                    progress, status_text = msg[1], msg[2]
                    print(f"Progress: {progress:.1%} - {status_text}")
                
                elif msg[0] == 'status':
                    print(f"Status: {msg[1]}")
                
                elif msg[0] == 'error':
                    print(f"Error: {msg[1]}")
                
                elif msg[0] == 'success':
                    print(f"Success: {msg[1]}")
            
            except (EOFError, KeyboardInterrupt):
                break
            
            except Exception as e:
                print(f"Error in queue listener: {e}")
                break
    
    # Start queue listener in a separate thread
    from threading import Thread
    listener_thread = Thread(target=queue_listener, args=(status_queue,))
    listener_thread.daemon = True
    listener_thread.start()
    
    # Process the document
    try:
        print("\nStarting document processing...")
        start_time = time.time()
        
        success = process_documents([str(sample_file)], status_queue, True)
        
        elapsed_time = time.time() - start_time
        
        if success:
            print(f"\nDocument processing completed successfully in {elapsed_time:.2f} seconds!")
            
            # Check for extracted entities
            entities_file = DATA_DIR / "extracted" / "entities.json"
            relationships_file = DATA_DIR / "extracted" / "relationships.json"
            
            if entities_file.exists():
                import json
                with open(entities_file, 'r') as f:
                    entities = json.load(f)
                print(f"\nExtracted {len(entities)} entities")
                
                # Print top 10 entities by type
                entity_types = {}
                for entity in entities:
                    entity_type = entity.get('type', 'unknown')
                    if entity_type not in entity_types:
                        entity_types[entity_type] = []
                    entity_types[entity_type].append(entity)
                
                print("\nEntities by type:")
                for entity_type, type_entities in entity_types.items():
                    print(f"- {entity_type}: {len(type_entities)} entities")
                    # Print up to 5 examples of each type
                    for i, entity in enumerate(sorted(type_entities, key=lambda e: e.get('confidence', 0), reverse=True)[:5]):
                        print(f"  - {entity['text']} (confidence: {entity.get('confidence', 0):.4f})")
            else:
                print("\nNo entities were extracted")
            
            if relationships_file.exists():
                import json
                with open(relationships_file, 'r') as f:
                    relationships = json.load(f)
                print(f"\nExtracted {len(relationships)} relationships")
                
                # Print top 10 relationships by type
                rel_types = {}
                for rel in relationships:
                    rel_type = rel.get('type', 'unknown')
                    if rel_type not in rel_types:
                        rel_types[rel_type] = []
                    rel_types[rel_type].append(rel)
                
                print("\nRelationships by type:")
                for rel_type, type_rels in rel_types.items():
                    print(f"- {rel_type}: {len(type_rels)} relationships")
                    # Print up to 5 examples of each type
                    for i, rel in enumerate(sorted(type_rels, key=lambda r: r.get('confidence', 0), reverse=True)[:5]):
                        print(f"  - {rel['subject']} -> {rel['object']} (confidence: {rel.get('confidence', 0):.4f})")
            else:
                print("\nNo relationships were extracted")
            
        else:
            print(f"\nDocument processing failed after {elapsed_time:.2f} seconds")
    
    except Exception as e:
        print(f"Error in test: {e}")
        traceback.print_exc()
    
    # Signal the listener to stop
    status_queue.put(('end', None))
    listener_thread.join()
    
    print("\nTest completed!")

if __name__ == "__main__":
    # Ensure clean CUDA initialization for multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, continue
        pass
    
    test_pipeline()
