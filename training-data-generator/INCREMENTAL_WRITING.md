# Incremental Writing Implementation

## Overview

The JSONL generators have been updated to implement incremental writing instead of buffering all data in memory. This critical fix prevents:

- Memory exhaustion on large datasets (100K+ items)
- 12-48 hour delays before output appears
- Data loss on job interruption

## Changes Made

### 1. JSONLCodeGenerator (`src/generators/jsonl_code_gen.py`)

**Previous Behavior:**
- Buffered all conversations in memory (`self.conversations` list)
- Only wrote to disk in `finalize()` when job completes
- For 330K items, this could take 12-48 hours with no output

**New Behavior:**
- Opens file handle in `__init__` for immediate writing
- Flushes buffer to disk every 100 conversations (configurable)
- Writes incrementally throughout job execution
- File appears and grows within seconds of job start

**Key Changes:**
```python
# New parameters
buffer_size: int = 100  # Configurable buffer size
_file_handle: file handle kept open during job
_total_written: int  # Track total items written

# New methods
_open_file()  # Opens file immediately
_flush()      # Writes buffer to disk incrementally

# Modified methods
add()         # Now flushes when buffer is full
finalize()    # Flushes remaining items and closes file
```

### 2. JSONLConversationGenerator (`src/generators/jsonl_conv_gen.py`)

**Same changes as JSONLCodeGenerator:**
- Incremental writing with configurable buffer
- File handle kept open throughout job
- Automatic flushing every 100 items
- Proper cleanup on error or completion

### 3. HuggingFace Downloader Fix (`scripts/download_huggingface_datasets.py`)

**Issue Fixed:**
- Streaming datasets getting stuck at "Processing: 0 items"
- Pre-filtering with `dataset.filter()` can cause iterator to hang

**Solution:**
- Removed pre-filtering on streaming dataset
- Moved language filtering inline during iteration
- Progress bar now updates correctly
- Iterator advances properly

**Changed:**
```python
# OLD (causes hang):
if self.config.language_field:
    dataset = dataset.filter(self._filter_python)

for example in dataset:
    # process...

# NEW (works correctly):
for example in dataset:
    # Filter inline
    if self.config.language_field:
        if not self._filter_python(example):
            continue
    # process...
```

## Usage

### Basic Usage (Unchanged)

```python
from src.generators.jsonl_code_gen import JSONLCodeGenerator
from src.config.schema import CodeOutputOptions

generator = JSONLCodeGenerator(
    output_path="output.jsonl",
    options=CodeOutputOptions(),
    logger=logger
)

for item in data_source:
    generator.add(item)  # Automatically flushes every 100 items

generator.finalize()  # Flushes remaining items and closes file
```

### Custom Buffer Size

```python
# Smaller buffer for faster writes (more I/O overhead)
generator = JSONLCodeGenerator(
    output_path="output.jsonl",
    buffer_size=50  # Flush every 50 items
)

# Larger buffer for better performance (less I/O overhead)
generator = JSONLCodeGenerator(
    output_path="output.jsonl",
    buffer_size=500  # Flush every 500 items
)
```

## Benefits

### 1. Immediate Feedback
- Output file appears within seconds
- Progress can be monitored with `wc -l output.jsonl`
- No waiting 12-48 hours to see results

### 2. Memory Efficiency
- Constant memory usage regardless of dataset size
- Buffer size = 100 items (vs. 330K+ items before)
- Suitable for processing millions of items

### 3. Crash Recovery
- Data is saved incrementally
- Job interruption only loses buffer contents (max 100 items)
- Can resume from last written item

### 4. Better Monitoring
- File size grows in real-time
- Can tail output: `tail -f output.jsonl`
- Progress tracking is accurate

## Performance Characteristics

### Buffer Size Impact

| Buffer Size | Flush Frequency | I/O Overhead | Memory Usage |
|-------------|-----------------|--------------|--------------|
| 10          | Every 10 items  | High         | Very Low     |
| 100         | Every 100 items | Low          | Low          |
| 1000        | Every 1000 items| Very Low     | Medium       |

**Recommendation:** Default of 100 is optimal for most use cases.

### Timing Comparison

**Before (buffering all data):**
- 330K items × 1KB each = ~330MB in memory
- First output: 12-48 hours
- Memory: 330MB+

**After (incremental writing):**
- 100 items × 1KB each = ~100KB in memory
- First output: seconds
- Memory: 100KB (constant)

## Testing

Run the test suite:

```bash
uv run python test_incremental_writing.py
```

The test verifies:
1. File is created immediately
2. File grows after each flush
3. Total items written is correct
4. JSON format is valid
5. No data loss

## Backward Compatibility

The changes are fully backward compatible:

- Default behavior unchanged (buffer_size=100)
- API unchanged (same `add()` and `finalize()` methods)
- Output format identical (same JSONL structure)
- Can be used as drop-in replacement

## Error Handling

### File I/O Errors
- Logged with context
- File handle closed on error
- Exception re-raised for caller handling

### Cleanup
- `__del__` method ensures file handle is closed
- Works even if `finalize()` not called
- Safe for process interruption

## Monitoring Progress

### Real-time Line Count
```bash
watch -n 1 "wc -l output.jsonl"
```

### Tail Output
```bash
tail -f output.jsonl | jq .
```

### File Size
```bash
watch -n 1 "ls -lh output.jsonl"
```

### Estimated Completion
```bash
# Items per second
echo "scale=2; $(wc -l < output.jsonl) / $(stat -c %Y output.jsonl - $(stat -c %W output.jsonl))" | bc

# Estimated time to completion (if total known)
# (total_items - current_items) / items_per_second
```

## Best Practices

1. **Use default buffer size** unless you have specific requirements
2. **Monitor progress** with `wc -l` or `watch`
3. **Check logs** for flush messages at DEBUG level
4. **Handle interrupts** gracefully (data is saved incrementally)
5. **Verify output** after job completes with line count

## Implementation Details

### Thread Safety
Current implementation is NOT thread-safe. For parallel writing:
- Use separate generator instances per thread
- Write to different output files
- Merge files after completion

### Flush Timing
Flushes occur:
1. Every N items (buffer_size)
2. On `finalize()` call
3. On generator destruction (`__del__`)

### Disk I/O
- Uses Python's buffered I/O (`open()` with buffering)
- Explicit `flush()` ensures data reaches disk
- No additional buffering needed

## Future Enhancements

Potential improvements:
1. Add progress callback for custom monitoring
2. Support for compressed output (gzip streaming)
3. Automatic checkpointing for resume
4. Parallel writing with thread-safe queue
5. Async I/O for better performance

## Related Files

- `/home/ling/workarea/nanochat/training-data-generator/src/generators/jsonl_code_gen.py`
- `/home/ling/workarea/nanochat/training-data-generator/src/generators/jsonl_conv_gen.py`
- `/home/ling/workarea/nanochat/training-data-generator/scripts/download_huggingface_datasets.py`
- `/home/ling/workarea/nanochat/training-data-generator/test_incremental_writing.py`
