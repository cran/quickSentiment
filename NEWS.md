# quickSentiment 0.3.6

## MAJOR ARCHITECTURAL UPGRADES
Fault-Tolerant Text Preprocessing: Completely rebuilt pre_process() to ensure pipeline resilience on massive, messy datasets. The function now uses a chunked processing architecture (default chunk_size = 5000). If a malformed string crashes a chunk, the function automatically degrades to row-by-row isolation for that specific block, neutralizing the corrupted string to NA_character_ and seamlessly resuming C++ vectorized speed for the rest of the dataset.

The Quarantine Attribute: When pre_process() isolates a fatal text encoding or parsing error, it now attaches a quarantine dataframe as an attribute to the output vector (attr(cleaned_vector, "quarantine")). This allows production pipelines to log the exact row_index, original_text, and error_message of failed rows without halting the overarching R session.

## NEW FEATURES
Advanced Memory Management: Exposed the chunk_size argument in pre_process(), allowing advanced users to optimize the speed-to-memory trade-off based on their specific hardware and data quality.

Automated Data Sanitization: The main pipeline() function now safely detects and simultaneously drops NA values in both the text vector and sentiment label vectors prior to training, preventing downstream failures caused by quarantined text rows.
