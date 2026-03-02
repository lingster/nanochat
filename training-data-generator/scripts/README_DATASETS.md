# HuggingFace Dataset Downloader

Download large-scale Python code datasets from HuggingFace Hub to help reach 50GB+ of training data.

## Overview

This script downloads Python code datasets from HuggingFace Hub and converts them to parquet format matching the nanochat pretraining format (`{text: string}`).

## Features

- **Resume capability**: Interrupted downloads can be resumed
- **Deduplication**: SHA-256 hash-based deduplication within datasets
- **Data validation**: Filters for code quality (length, structure, content)
- **Progress tracking**: Real-time progress with file sizes
- **Shard generation**: Automatic 100MB parquet shards
- **Size limits**: Optional max download size

## Available Datasets

### 1. The Stack (Python) - `the-stack-python`

**Description**: Deduplicated Python code from The Stack dataset

**HuggingFace**: `bigcode/the-stack-dedup`

**Estimated Size**: ~35 GB

**Content**: High-quality Python code from various GitHub repositories

**Best for**: Large-scale pretraining on diverse Python codebases

```bash
uv run python scripts/download_huggingface_datasets.py \
    --dataset the-stack-python \
    --max-size-gb 30 \
    --output-dir /mnt/archive/nanochat/huggingface_data
```

### 2. CodeParrot Clean - `codeparrot-clean`

**Description**: Cleaned and filtered Python code from GitHub

**HuggingFace**: `codeparrot/codeparrot-clean`

**Estimated Size**: ~8 GB

**Content**: Python code with quality filtering applied

**Best for**: Medium-scale pretraining with higher quality standards

```bash
uv run python scripts/download_huggingface_datasets.py \
    --dataset codeparrot-clean \
    --output-dir /mnt/archive/nanochat/huggingface_data
```

### 3. GitHub Code Python - `github-code-python`

**Description**: Python code directly from GitHub repositories

**HuggingFace**: `codeparrot/github-code`

**Estimated Size**: ~15 GB

**Content**: Raw Python files from popular GitHub projects

**Best for**: Diverse code patterns and real-world examples

```bash
uv run python scripts/download_huggingface_datasets.py \
    --dataset github-code-python \
    --max-size-gb 15 \
    --output-dir /mnt/archive/nanochat/huggingface_data
```

### 4. Python Code Instructions - `python-code-instructions`

**Description**: Python code with instruction-following format

**HuggingFace**: `iamtarun/python_code_instructions_18k_alpaca`

**Estimated Size**: ~0.5 GB

**Content**: Python code paired with natural language instructions

**Best for**: Instruction-following and code generation tasks

```bash
uv run python scripts/download_huggingface_datasets.py \
    --dataset python-code-instructions \
    --output-dir /mnt/archive/nanochat/huggingface_data
```

## Usage

### Quick Start

1. **List available datasets**:
```bash
uv run python scripts/download_huggingface_datasets.py --list
```

2. **Download a dataset**:
```bash
uv run python scripts/download_huggingface_datasets.py \
    --dataset the-stack-python \
    --max-size-gb 30 \
    --output-dir /mnt/archive/nanochat/huggingface_data
```

3. **Resume interrupted download**:
```bash
# Just run the same command again
uv run python scripts/download_huggingface_datasets.py \
    --dataset the-stack-python \
    --max-size-gb 30 \
    --output-dir /mnt/archive/nanochat/huggingface_data
```

### Command-Line Options

```bash
--dataset DATASET          Dataset to download (required)
                          Choices: the-stack-python, codeparrot-clean,
                                  github-code-python, python-code-instructions

--max-size-gb SIZE        Maximum download size in GB (optional)
                          If not specified, downloads entire dataset

--output-dir PATH         Output directory for parquet files
                          Default: /mnt/archive/nanochat/huggingface_data

--no-resume              Start fresh, ignore previous state
                          Default: resume from last run

--log-level LEVEL        Logging level
                          Choices: DEBUG, INFO, WARNING, ERROR
                          Default: INFO

--list                   List available datasets and exit
```

### Examples

**Download 30GB from The Stack**:
```bash
uv run python scripts/download_huggingface_datasets.py \
    --dataset the-stack-python \
    --max-size-gb 30 \
    --output-dir /mnt/archive/nanochat/huggingface_data
```

**Download entire CodeParrot dataset**:
```bash
uv run python scripts/download_huggingface_datasets.py \
    --dataset codeparrot-clean \
    --output-dir /mnt/archive/nanochat/huggingface_data
```

**Download with debug logging**:
```bash
uv run python scripts/download_huggingface_datasets.py \
    --dataset github-code-python \
    --max-size-gb 15 \
    --log-level DEBUG \
    --output-dir /mnt/archive/nanochat/huggingface_data
```

**Start fresh (ignore cached state)**:
```bash
uv run python scripts/download_huggingface_datasets.py \
    --dataset the-stack-python \
    --no-resume \
    --output-dir /mnt/archive/nanochat/huggingface_data
```

## Output Format

### Directory Structure

```
/mnt/archive/nanochat/huggingface_data/
├── the-stack-python/
│   ├── shard_00000.parquet
│   ├── shard_00001.parquet
│   ├── ...
│   └── .the-stack-python_state.json  # Resume state
├── codeparrot-clean/
│   ├── shard_00000.parquet
│   └── ...
└── github-code-python/
    ├── shard_00000.parquet
    └── ...
```

### Parquet Schema

All output files use the same schema as nanochat pretraining data:

```python
{
    "text": string  # Python code content
}
```

### Shard Size

- Each shard is approximately **100 MB**
- Compressed with **ZSTD** compression
- Row group size: **1024 rows**

### Example: Reading Parquet Files

```python
import pyarrow.parquet as pq

# Read a single shard
table = pq.read_table("/mnt/archive/nanochat/huggingface_data/the-stack-python/shard_00000.parquet")

# Iterate over rows
for row in table.to_pylist():
    print(row['text'])
```

## Resume/Cache Functionality

### How It Works

The script automatically tracks:
- **Content hashes**: SHA-256 hashes of all processed items
- **Download progress**: Total items and bytes downloaded
- **Shard index**: Current shard being written

State is saved in: `<output_dir>/<dataset_name>/.{dataset_name}_state.json`

### Resume Behavior

**Default (resume enabled)**:
- Skips already-downloaded content based on hash
- Continues from last shard index
- Preserves deduplication state

**Start fresh (--no-resume)**:
- Ignores previous state
- Overwrites existing files
- Starts from shard 0

### Example

```bash
# First run - downloads 5GB
uv run python scripts/download_huggingface_datasets.py \
    --dataset the-stack-python \
    --max-size-gb 30

# Interrupted after 5GB (Ctrl+C)
^C

# Resume - continues from 5GB
uv run python scripts/download_huggingface_datasets.py \
    --dataset the-stack-python \
    --max-size-gb 30

# Output:
# "Resuming from previous run: 50,000 items, 5.00 GB"
```

## Data Quality

### Filtering Criteria

The script applies the following quality filters:

1. **Length validation**:
   - Minimum: 100 characters
   - Maximum: 1,000,000 characters

2. **Content validation**:
   - At least 30% alphanumeric characters
   - At least 3 lines of code
   - No null bytes

3. **Deduplication**:
   - SHA-256 hash-based deduplication
   - Tracks all seen hashes in memory

### Cleaning Operations

- **Whitespace normalization**: Collapses multiple spaces
- **Null byte removal**: Removes `\x00` characters
- **Trimming**: Strips leading/trailing whitespace

## Size Estimates

### Reaching 50GB Target

To reach 50GB of Python training data, consider:

| Dataset                   | Command                                                    | Size  |
|---------------------------|-----------------------------------------------------------|-------|
| The Stack (Python)        | `--dataset the-stack-python --max-size-gb 35`             | 35 GB |
| CodeParrot Clean          | `--dataset codeparrot-clean`                              | 8 GB  |
| GitHub Code Python        | `--dataset github-code-python --max-size-gb 7`            | 7 GB  |
| **Total**                 |                                                           | **50 GB** |

### Download Time Estimates

Times vary based on network speed and HuggingFace API rate limits:

- **Fast network (100 Mbps)**: ~1-2 hours per 10GB
- **Medium network (50 Mbps)**: ~2-4 hours per 10GB
- **Slow network (10 Mbps)**: ~8-10 hours per 10GB

**Pro tip**: Use resume functionality to download in chunks over multiple sessions.

## Troubleshooting

### Network Errors

**Problem**: Connection timeout or rate limiting

**Solution**:
```bash
# The script will automatically retry failed requests
# If interrupted, just re-run the same command to resume
uv run python scripts/download_huggingface_datasets.py \
    --dataset the-stack-python \
    --max-size-gb 30
```

### HuggingFace Authentication

**Problem**: Dataset requires authentication

**Solution**:
```bash
# Log in to HuggingFace
huggingface-cli login

# Then run download
uv run python scripts/download_huggingface_datasets.py \
    --dataset the-stack-python
```

### Memory Issues

**Problem**: Script uses too much memory

**Solution**:
- The script uses streaming mode to minimize memory
- Deduplication hashes are kept in memory (unavoidable)
- For very large downloads (>100GB), consider splitting into chunks with `--max-size-gb`

### Disk Space

**Problem**: Running out of disk space

**Solution**:
```bash
# Check available space
df -h /mnt/archive/nanochat

# Use --max-size-gb to limit download
uv run python scripts/download_huggingface_datasets.py \
    --dataset the-stack-python \
    --max-size-gb 20
```

## Integration with nanochat

### Using Downloaded Data

After downloading, use the parquet files with nanochat's training pipeline:

1. **Update dataset configuration** in nanochat to point to downloaded shards
2. **Combine with existing data** by placing shards in nanochat's data directory
3. **Train tokenizer** on combined data first
4. **Run pretraining** with all available shards

### Example Integration

```bash
# Download data
uv run python scripts/download_huggingface_datasets.py \
    --dataset the-stack-python \
    --max-size-gb 30 \
    --output-dir /mnt/archive/nanochat/huggingface_data

# Copy to nanochat data directory
cp /mnt/archive/nanochat/huggingface_data/the-stack-python/*.parquet \
   ~/.cache/nanochat/base_data/

# Train with nanochat
cd /path/to/nanochat
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20
```

## Performance Tips

1. **Use fast storage**: SSD/NVMe recommended for output directory
2. **Monitor progress**: Use default INFO logging for progress updates
3. **Batch downloads**: Download multiple datasets in sequence
4. **Resume often**: Don't worry about interruptions, just resume
5. **Parallel downloads**: Run multiple instances for different datasets

## Dependencies

Required packages (already in `requirements.txt`):

- `datasets` - HuggingFace datasets library
- `pyarrow` - Parquet file I/O
- `tqdm` - Progress bars
- `xxhash` - Fast hashing (optional, falls back to hashlib)

## Advanced Usage

### Customizing Dataset Configs

To add a new dataset, edit the script and add to `DATASET_CONFIGS`:

```python
DATASET_CONFIGS = {
    "my-custom-dataset": DatasetConfig(
        name="my-custom-dataset",
        hf_name="organization/dataset-name",
        subset=None,  # or "subset_name"
        split="train",
        text_field="code",  # field containing text
        language_field="lang",  # optional: field for language filtering
        description="My custom Python dataset",
        estimated_size_gb=10.0,
    ),
}
```

### Programmatic Usage

```python
from pathlib import Path
import logging
from download_huggingface_datasets import HuggingFaceDownloader, setup_logging

# Set up logging
logger = setup_logging("INFO")

# Create downloader
downloader = HuggingFaceDownloader(
    dataset_name="the-stack-python",
    output_dir=Path("/mnt/archive/nanochat/huggingface_data/the-stack-python"),
    max_size_gb=30.0,
    resume=True,
    logger=logger,
)

# Download
downloader.download()
```

## License

MIT License - Same as nanochat project

## Support

For issues or questions:
- Check logs with `--log-level DEBUG`
- Review HuggingFace dataset documentation
- Open an issue in the nanochat repository
