#!/usr/bin/env python3
"""
Test script for the fixed entity extraction.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Direct test using Flair
def test_direct_flair():
    """Test direct usage of Flair as shown in the example."""
    print("\n=== Testing Direct Flair Usage ===")
    
    test_text = """Jason is 30 years old and earns thirty thousand pounds per year.
    He lives in London and works as a Software Engineer. His employee ID is 12345.
    Susan, aged 9, does not have a job and earns zero pounds.
    She is a student in elementary school and lives in Manchester.
    Michael, a 45-year-old Doctor, earns ninety-five thousand pounds annually.
    He resides in Birmingham and has an employee ID of 67890.
    Emily is a 28-year-old Data Scientist who earns seventy-two thousand pounds.
    She is based in Edinburgh and her employee ID is 54321."""
    
    print("Loading Flair models...")
    from flair.nn import Classifier
    from flair.splitter import SegtokSentenceSplitter
    
    # Initialize splitter and split text
    splitter = SegtokSentenceSplitter()
    sentences = splitter.split(test_text)
    print(f"Split text into {len(sentences)} sentences")
    
    # Load NER tagger and predict
    print("Loading NER model...")
    tagger = Classifier.load("flair/ner-english-ontonotes-fast")
    
    print("Predicting entities...")
    tagger.predict(sentences)
    
    # Extract entities
    ent = []
    for sentence in sentences:
        ent.append(sentence.get_labels('ner'))
    
    # Load relation extractor and predict
    print("Loading relation extractor...")
    extractor = Classifier.load('relations')
    
    print("Predicting relationships...")
    for sentence in sentences:
        extractor.predict(sentence)
    
    # Extract relationships
    rels = []
    for sentence in sentences:
        rels.append(sentence.get_labels("relation"))
    
    # Print entities found
    print("\nEntities found:")
    total_entities = 0
    for i, sentence_entities in enumerate(ent):
        if sentence_entities:
            print(f"Sentence {i+1}: {len(sentence_entities)} entities")
            total_entities += len(sentence_entities)
            for j, entity in enumerate(sentence_entities[:3]):  # Show up to 3 per sentence
                print(f"  - {entity.data_point.text} ({entity.data_point.tag}): {entity.score:.4f}")
            if len(sentence_entities) > 3:
                print(f"  - ... and {len(sentence_entities) - 3} more")
    
    print(f"\nTotal entities found: {total_entities}")
    
    # Print relationships found
    print("\nRelationships found:")
    total_rels = 0
    for i, sentence_rels in enumerate(rels):
        if sentence_rels:
            print(f"Sentence {i+1}: {len(sentence_rels)} relationships")
            total_rels += len(sentence_rels)
            for j, rel in enumerate(sentence_rels[:3]):  # Show up to 3 per sentence
                if hasattr(rel.data_point, 'first') and hasattr(rel.data_point, 'second'):
                    print(f"  - {rel.data_point.first.text} -> {rel.data_point.tag} -> {rel.data_point.second.text}: {rel.score:.4f}")
            if len(sentence_rels) > 3:
                print(f"  - ... and {len(sentence_rels) - 3} more")
    
    print(f"\nTotal relationships found: {total_rels}")

# Test using our updated EntityExtractor
def test_entity_extractor():
    """Test the updated EntityExtractor."""
    print("\n=== Testing Updated EntityExtractor ===")
    
    from src.processing.entity_extractor import EntityExtractor
    
    # Create test chunks
    test_chunk = {
        'chunk_id': 'test_chunk_1',
        'document_id': 'test_doc',
        'text': """Jason is 30 years old and earns thirty thousand pounds per year.
        He lives in London and works as a Software Engineer. His employee ID is 12345.
        Susan, aged 9, does not have a job and earns zero pounds.
        She is a student in elementary school and lives in Manchester.
        Michael, a 45-year-old Doctor, earns ninety-five thousand pounds annually.
        He resides in Birmingham and has an employee ID of 67890.
        Emily is a 28-year-old Data Scientist who earns seventy-two thousand pounds.
        She is based in Edinburgh and her employee ID is 54321."""
    }
    
    # Create EntityExtractor instance
    extractor = EntityExtractor()
    
    # Process the chunk
    print("Processing chunk with EntityExtractor...")
    processed_chunks, entities, relationships = extractor.process_chunks([test_chunk])
    
    # Print results
    print(f"\nEntities found: {len(entities)}")
    for i, entity in enumerate(entities[:10]):  # Show up to 10 entities
        print(f"{i+1}. {entity['text']} ({entity['type']}): {entity['confidence']:.4f}")
    if len(entities) > 10:
        print(f"... and {len(entities) - 10} more")
    
    print(f"\nRelationships found: {len(relationships)}")
    for i, rel in enumerate(relationships[:10]):  # Show up to 10 relationships
        print(f"{i+1}. {rel['subject']} -> {rel['type']} -> {rel['object']}: {rel['confidence']:.4f}")
    if len(relationships) > 10:
        print(f"... and {len(relationships) - 10} more")
    
    # Check if the processed chunk has entity annotations
    if processed_chunks and 'entity_count' in processed_chunks[0]:
        print(f"\nProcessed chunk has {processed_chunks[0]['entity_count']} entity annotations")
        print(f"Sample of annotated text:")
        lines = processed_chunks[0]['text'].split('\n')
        for i, line in enumerate(lines[:3]):  # Show the first 3 lines of annotated text
            print(f"{i+1}. {line[:100]}{'...' if len(line) > 100 else ''}")

if __name__ == "__main__":
    # Test direct Flair usage
    test_direct_flair()
    
    # Test the updated EntityExtractor
    test_entity_extractor()
