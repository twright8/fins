#!/usr/bin/env python3
"""
Test script to verify NLTK sent_tokenize functionality.
"""
import time
import nltk

# Try to load punkt data or download if needed
try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK punkt data already downloaded")
except LookupError:
    print("Downloading NLTK punkt data...")
    nltk.download('punkt')
    print("NLTK punkt data downloaded successfully")

# Sample text to test - same as in the test_entity_extraction.py
test_text = """Jason is 30 years old and earns thirty thousand pounds per year.
He lives in London and works as a Software Engineer. His employee ID is 12345.
Susan, aged 9, does not have a job and earns zero pounds.
She is a student in elementary school and lives in Manchester.
Michael, a 45-year-old Doctor, earns ninety-five thousand pounds annually.
He resides in Birmingham and has an employee ID of 67890.
Emily is a 28-year-old Data Scientist who earns seventy-two thousand pounds.
She is based in Edinburgh and her employee ID is 54321."""

# Test the Flair splitter performance (if available)
try:
    from flair.splitter import SegtokSentenceSplitter
    
    print("\nTesting Flair's SegtokSentenceSplitter:")
    splitter = SegtokSentenceSplitter()
    
    start_time = time.time()
    sentences_flair = splitter.split(test_text)
    flair_time = time.time() - start_time
    
    print(f"Flair split the text into {len(sentences_flair)} sentences in {flair_time:.6f} seconds")
except ImportError:
    print("\nFlair library not available, skipping Flair test")

# Test the NLTK sent_tokenize performance
print("\nTesting NLTK's sent_tokenize:")
start_time = time.time()
sentences_nltk = nltk.sent_tokenize(test_text)
nltk_time = time.time() - start_time

print(f"NLTK split the text into {len(sentences_nltk)} sentences in {nltk_time:.6f} seconds")

# Print the sentences from NLTK
print("\nSentences detected by NLTK:")
for i, sentence in enumerate(sentences_nltk):
    print(f"{i+1}. {sentence}")

# Compare performance if Flair is available
if 'flair_time' in locals():
    print(f"\nPerformance comparison:")
    print(f"- Flair: {flair_time:.6f} seconds")
    print(f"- NLTK:  {nltk_time:.6f} seconds")
    print(f"- NLTK is {flair_time/nltk_time:.2f}x faster than Flair")
