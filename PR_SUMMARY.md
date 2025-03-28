# Implemented Configurable Parallelism in Processing Pipeline

This PR introduces parallelism using `concurrent.futures.ThreadPoolExecutor` to several stages of the document processing pipeline to improve throughput. The implementation follows the developer brief and adds configurable parallelism to different pipeline stages.

## Changes

### 1. Configuration Updates (config.yaml)
- Added new section under `document_processing` for `parallelism_workers` with the following settings:
  - `document_loading`: 2 workers (for loading documents concurrently)
  - `chunker_pages`: 4 workers (for processing pages/items within DocumentChunker concurrently)
  - `coref_batches`: 2 workers (for processing coref batches concurrently)
  - `entity_splitting`: 4 workers (for splitting chunk texts into sentences concurrently)
  - `qdrant_upsert_batches`: 4 workers (for upserting batches to Qdrant concurrently)

### 2. Document Loading (pipeline.py)
- Implemented parallel document loading in `process_documents`
- Added a `_load_single_document` helper function
- Used `ThreadPoolExecutor` to load multiple documents concurrently
- Maintained proper error handling and progress reporting

### 3. Document Chunking (document_chunker.py)
- Implemented parallel processing of content items in `chunk_document`
- Added a `_process_single_content_item` helper function
- Used `ThreadPoolExecutor` to process content items concurrently
- Maintained chunk ordering and progress reporting

### 4. Coreference Resolution (coreference_resolver.py)
- Implemented parallel batch processing in `process_chunks`
- Added a `_process_coref_batch` helper function
- Used `ThreadPoolExecutor` to process batches concurrently
- Maintained proper aggregation of results across batches

### 5. Entity Extraction (entity_extractor.py)
- Implemented parallel sentence splitting in `process_chunks`
- Added a `_split_chunk_text_nltk` helper function
- Used `ThreadPoolExecutor` to split chunks into sentences concurrently
- Note: Did not parallelize the NER prediction itself, as that is already batched internally

### 6. Qdrant Upsert (indexer.py)
- Implemented parallel batch upserting in `_index_with_vectors`
- Added a `_upsert_batch_to_qdrant` helper function
- Used `ThreadPoolExecutor` to upsert batches concurrently
- Maintained proper error handling and progress reporting

## Performance Considerations

- **Threading vs. Processing**: Used `ThreadPoolExecutor` as specified in the brief to avoid memory duplication of shared models and clients.
- **Error Handling**: Implemented robust error handling in all helpers and around futures.
- **Progress Reporting**: Adapted progress reporting to account for parallel tasks completed.
- **Worker Count**: The worker counts in config.yaml are starting points and can be tuned based on hardware capabilities and workload.

## Testing Notes

When testing this implementation, consider the following:

1. **Hardware Utilization**: Monitor CPU/GPU utilization and memory usage to ensure resources are being used efficiently.
2. **Worker Count Tuning**: Experiment with different worker counts based on your hardware. Too many workers can lead to resource contention.
3. **Edge Cases**: Test with various document sizes and types to ensure parallelism works correctly for different workloads.
4. **Error Handling**: Test with problematic files or content to ensure errors in parallel tasks don't crash the entire pipeline.

## Future Improvements

Potential future improvements could include:

1. Dynamic worker count adjustment based on available system resources
2. More granular parallelism for other pipeline steps
3. Adding an option to switch between thread-based and process-based parallelism for certain stages
