# Simple Python Training Data Download Guide

This guide provides two proven, simple approaches for downloading 50GB+ of Python training data, bypassing complex extraction pipelines.

## Quick Start

### Option A: Direct Python Download (Recommended for Most Users)
```bash
# Install datasets library
uv add datasets

# Download 8GB of clean Python code
uv run python scripts/simple_download.py --dataset codeparrot-clean --max-gb 8

# Download 35GB from The Stack
uv run python scripts/simple_download.py --dataset the-stack --max-gb 35

# Download 15GB from GitHub Code
uv run python scripts/simple_download.py --dataset github-code --max-gb 15
```

### Option B: HuggingFace CLI Download (Maximum Reliability)
```bash
# One-command download and conversion
./scripts/download_with_cli.sh codeparrot-clean 8

# Or with custom output directory
OUTPUT_DIR=/mnt/archive/python ./scripts/download_with_cli.sh the-stack 35
```

---

## Option A: Direct Python Download (`simple_download.py`)

### When to Use
- You want a pure Python solution
- You need fine-grained control over the download
- You want to resume interrupted downloads
- You prefer streaming downloads (no large disk space needed)

### Pros
✅ Pure Python, no external CLI tools
✅ Streams data directly to parquet (minimal disk usage)
✅ Built-in resume capability
✅ Progress tracking saved automatically
✅ Memory efficient (streaming mode)
✅ Handles large datasets without running out of disk space

### Cons
❌ Requires datasets library installation
❌ Slightly slower than CLI approach
❌ May have issues with some HuggingFace authentication

### Available Datasets

| Dataset | Key | Size Available | Quality | Description |
|---------|-----|----------------|---------|-------------|
| **CodeParrot Clean** | `codeparrot-clean` | ~8GB | ⭐⭐⭐⭐⭐ | Clean, deduplicated Python from GitHub |
| **The Stack Dedup** | `the-stack` | ~35GB | ⭐⭐⭐⭐ | Python subset of The Stack (deduplicated) |
| **GitHub Code** | `github-code` | ~15GB | ⭐⭐⭐⭐ | Python code from GitHub repositories |

### Usage Examples

#### Basic Download
```bash
# Download 8GB of codeparrot-clean
uv run python scripts/simple_download.py \
    --dataset codeparrot-clean \
    --max-gb 8
```

#### Custom Output Directory
```bash
# Download to specific location
uv run python scripts/simple_download.py \
    --dataset the-stack \
    --output-dir /mnt/archive/python_data \
    --max-gb 35
```

#### Resume Interrupted Download
```bash
# Just rerun the same command - progress is saved automatically
uv run python scripts/simple_download.py \
    --dataset codeparrot-clean \
    --max-gb 8
```

#### Fresh Start (No Resume)
```bash
# Start from scratch
uv run python scripts/simple_download.py \
    --dataset codeparrot-clean \
    --max-gb 8 \
    --no-resume
```

### Output Format
- Files: `<dataset>_shard_NNNNN.parquet`
- Schema: `{text: string}`
- Compression: Snappy
- Shard size: ~100MB each
- Location: `/mnt/archive/nanochat/` (default)

### Expected Download Times

| Dataset | Size | Connection | Time |
|---------|------|------------|------|
| codeparrot-clean | 8GB | 100 Mbps | ~15 min |
| codeparrot-clean | 8GB | 1 Gbps | ~2 min |
| the-stack | 35GB | 100 Mbps | ~60 min |
| the-stack | 35GB | 1 Gbps | ~8 min |
| github-code | 15GB | 100 Mbps | ~25 min |
| github-code | 15GB | 1 Gbps | ~4 min |

*Times are approximate and include conversion overhead*

### Troubleshooting

#### Problem: "No module named 'datasets'"
```bash
# Solution: Install datasets library
uv add datasets pyarrow tqdm
```

#### Problem: "Connection timeout" or "Network error"
```bash
# Solution 1: Check internet connection
ping huggingface.co

# Solution 2: Set HF token if dataset requires authentication
export HF_TOKEN=your_token_here
# or
huggingface-cli login
```

#### Problem: Download is very slow
```bash
# Solution: Try Option B (CLI approach) instead
# It uses optimized parallel downloads
./scripts/download_with_cli.sh codeparrot-clean 8
```

#### Problem: "Disk full" error
```bash
# Solution: The script uses streaming, but ensure you have enough space
# Each 100MB parquet shard is written as data streams
# Minimum: 1.2x the target size (e.g., 10GB for 8GB download)

# Check available space
df -h ~/.cache/nanochat/
```

#### Problem: Progress file corrupted
```bash
# Solution: Delete progress file and restart
rm /mnt/archive/nanochat/.codeparrot-clean_progress.json

# Then restart download
uv run python scripts/simple_download.py --dataset codeparrot-clean --max-gb 8
```

---

## Option B: HuggingFace CLI Download (`download_with_cli.sh`)

### When to Use
- You want maximum download speed
- You have plenty of disk space
- You prefer battle-tested CLI tools
- You want parallel downloads

### Pros
✅ Fastest download speed (parallel workers)
✅ Most reliable (uses official HF CLI)
✅ Simple one-command operation
✅ Can resume interrupted downloads
✅ Works with all HuggingFace datasets
✅ Better for large datasets (50GB+)

### Cons
❌ Requires more disk space (downloads then converts)
❌ Two-step process (download + convert)
❌ Requires bash shell
❌ Additional CLI tool installation

### Usage Examples

#### Basic Download
```bash
# Download 8GB of codeparrot-clean
./scripts/download_with_cli.sh codeparrot-clean 8
```

#### Custom Output Directory
```bash
# Set custom output location
OUTPUT_DIR=/mnt/archive/python_data ./scripts/download_with_cli.sh the-stack 35
```

#### Custom Download Location
```bash
# Control both download and output directories
DOWNLOAD_DIR=/tmp/downloads \
OUTPUT_DIR=/mnt/archive/python_data \
./scripts/download_with_cli.sh github-code 15
```

### Output Format
- Same as Option A
- Files: `shard_NNNNN.parquet`
- Schema: `{text: string}`
- Compression: Snappy
- Shard size: ~100MB each

### Expected Download Times

| Dataset | Size | Connection | Download | Convert | Total |
|---------|------|------------|----------|---------|-------|
| codeparrot-clean | 8GB | 100 Mbps | ~10 min | ~3 min | ~13 min |
| codeparrot-clean | 8GB | 1 Gbps | ~1.5 min | ~3 min | ~4.5 min |
| the-stack | 35GB | 100 Mbps | ~45 min | ~10 min | ~55 min |
| the-stack | 35GB | 1 Gbps | ~6 min | ~10 min | ~16 min |

*CLI approach is faster than Option A due to parallel downloads*

### Disk Space Requirements

| Operation | Space Needed |
|-----------|--------------|
| Download only | ~1.5x target size |
| Conversion | ~2x target size (peak) |
| Final output | ~1x target size |

**Example:** For 35GB download:
- Download: ~52GB
- Peak (during conversion): ~70GB
- Final: ~35GB
- **Recommendation:** Have at least 2.5x target size available

### Troubleshooting

#### Problem: "huggingface-cli: command not found"
```bash
# Solution: The script will auto-install, but you can also:
pip install huggingface_hub[cli]
```

#### Problem: "Authentication required"
```bash
# Solution: Login to HuggingFace
huggingface-cli login
# Enter your token from https://huggingface.co/settings/tokens
```

#### Problem: Download failed partway through
```bash
# Solution: Just rerun - CLI supports resume
./scripts/download_with_cli.sh codeparrot-clean 8
```

#### Problem: Conversion script not found
```bash
# Solution: Ensure you're in the project root
cd /path/to/training-data-generator

# Verify converter exists
ls -la scripts/convert_to_parquet.py

# Run with full path if needed
bash /full/path/to/scripts/download_with_cli.sh codeparrot-clean 8
```

#### Problem: Out of disk space during conversion
```bash
# Solution 1: Clean up download directory after conversion
# The script will suggest this, or manually:
rm -rf /mnt/archive/nanochat/downloads/codeparrot-clean

# Solution 2: Use different disk for downloads
DOWNLOAD_DIR=/mnt/bigdisk/downloads \
OUTPUT_DIR=/mnt/archive/python \
./scripts/download_with_cli.sh codeparrot-clean 8
```

#### Problem: Conversion is slow
```bash
# Solution: The converter processes files sequentially
# For faster conversion, you can split the work:

# Option 1: Convert smaller batches
python scripts/convert_to_parquet.py \
    --input /mnt/archive/nanochat/downloads/codeparrot-clean \
    --output /mnt/archive/nanochat \
    --max-gb 4  # Process in smaller chunks
```

---

## Comparison: Which Option to Choose?

### Choose Option A (simple_download.py) if:
- ✅ You have limited disk space
- ✅ You prefer pure Python solutions
- ✅ You want streaming downloads
- ✅ You're downloading smaller datasets (<20GB)
- ✅ You want automatic resume without manual intervention

### Choose Option B (download_with_cli.sh) if:
- ✅ You have plenty of disk space
- ✅ You want maximum speed
- ✅ You're downloading large datasets (>30GB)
- ✅ You're comfortable with bash scripts
- ✅ You want the most reliable downloads

### Performance Comparison

| Metric | Option A (Python) | Option B (CLI) |
|--------|-------------------|----------------|
| **Download Speed** | Good | Excellent |
| **Disk Usage** | Low (streaming) | High (2x peak) |
| **Resume** | Automatic | Manual rerun |
| **Reliability** | Very Good | Excellent |
| **Setup** | Simple | Very Simple |
| **Memory Usage** | Low | Medium |

---

## Complete 50GB Download Strategy

### Strategy 1: Mix of Datasets (Recommended)
```bash
# Total: ~58GB (high quality, diverse)

# 1. CodeParrot Clean (8GB) - Highest quality
uv run python scripts/simple_download.py --dataset codeparrot-clean --max-gb 8

# 2. The Stack (35GB) - Large, diverse
./scripts/download_with_cli.sh the-stack 35

# 3. GitHub Code (15GB) - Additional diversity
uv run python scripts/simple_download.py --dataset github-code --max-gb 15
```

### Strategy 2: Single Large Dataset
```bash
# Option: The Stack only (fastest, most consistent)
./scripts/download_with_cli.sh the-stack 50
```

### Strategy 3: Incremental Download
```bash
# Start small, expand as needed

# Phase 1: Quick start (8GB, ~15 min)
uv run python scripts/simple_download.py --dataset codeparrot-clean --max-gb 8

# Phase 2: Expand (25GB, ~45 min)
./scripts/download_with_cli.sh the-stack 25

# Phase 3: Complete (17GB more, ~30 min)
uv run python scripts/simple_download.py --dataset github-code --max-gb 17
```

---

## Verifying Downloaded Data

### Check File Count and Size
```bash
# Count parquet files
ls -1 /mnt/archive/nanochat/*.parquet | wc -l

# Check total size
du -sh /mnt/archive/nanochat/

# Check individual shard sizes
ls -lh /mnt/archive/nanochat/*.parquet | head -10
```

### Inspect Parquet Contents
```python
# Quick check with Python
import pyarrow.parquet as pq

# Read first shard
table = pq.read_table('/mnt/archive/nanochat/codeparrot-clean_shard_00000.parquet')
print(f"Rows: {table.num_rows}")
print(f"Schema: {table.schema}")
print(f"Sample: {table.to_pandas().head()}")
```

### Validate Data Quality
```bash
# Create quick validation script
cat > validate_data.py << 'EOF'
import pyarrow.parquet as pq
from pathlib import Path

data_dir = Path("/mnt/archive/nanochat")
files = sorted(data_dir.glob("*.parquet"))

total_rows = 0
total_bytes = 0

print(f"Checking {len(files)} files...")

for f in files[:10]:  # Check first 10
    table = pq.read_table(f)
    rows = table.num_rows
    size = f.stat().st_size
    total_rows += rows
    total_bytes += size
    print(f"  {f.name}: {rows:,} rows, {size/1024/1024:.1f} MB")

print(f"\nTotal (first 10): {total_rows:,} rows, {total_bytes/1024/1024:.1f} MB")
EOF

python validate_data.py
```

---

## Integration with nanochat Training

### Using Downloaded Data for Training
```bash
# 1. Ensure data is in correct location
export NANOCHAT_BASE_DIR=~/.cache/nanochat

# 2. Verify data
ls -lh /mnt/archive/nanochat/*.parquet | head

# 3. Update nanochat to use Python data
# Modify dataset.py to point to python_data instead of base_data

# 4. Start training
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20
```

### Custom Dataset Integration
```python
# Example: Add to nanochat/dataset.py

def load_python_data(data_dir: str = "/mnt/archive/nanochat"):
    """Load Python training data from parquet shards."""
    data_dir = Path(data_dir).expanduser()
    files = sorted(data_dir.glob("*.parquet"))

    # Load with datasets library
    from datasets import load_dataset

    dataset = load_dataset(
        "parquet",
        data_files=[str(f) for f in files],
        split="train",
        streaming=True
    )

    return dataset
```

---

## Advanced Usage

### Parallel Downloads (Option A)
```bash
# Download multiple datasets in parallel
uv run python scripts/simple_download.py --dataset codeparrot-clean --max-gb 8 &
uv run python scripts/simple_download.py --dataset github-code --max-gb 15 &
wait
```

### Custom Dataset Configuration
```python
# Edit simple_download.py to add custom dataset

DATASETS = {
    "my-custom-dataset": {
        "name": "organization/dataset-name",
        "split": "train",
        "streaming": True,
        "text_field": "code",  # Field containing text
    },
}
```

### Batch Processing
```bash
# Download all datasets in one command
for dataset in codeparrot-clean the-stack github-code; do
    echo "Downloading $dataset..."
    uv run python scripts/simple_download.py --dataset $dataset --max-gb 20
done
```

---

## FAQ

### Q: Which approach is faster?
**A:** Option B (CLI) is typically 20-30% faster due to parallel downloads, but requires more disk space.

### Q: Can I combine both approaches?
**A:** Yes! Use Option A for smaller datasets and Option B for large ones.

### Q: How do I know if download completed successfully?
**A:** Both scripts print completion statistics. Check:
- Total shards created
- Total size matches target
- No error messages in output

### Q: Can I pause and resume downloads?
**A:**
- **Option A:** Yes, automatic resume on rerun
- **Option B:** Yes, rerun same command

### Q: What if I run out of disk space mid-download?
**A:**
- **Option A:** Stop will save progress, free space, continue
- **Option B:** Stop, free space, rerun (downloads resume)

### Q: How do I download more than 50GB?
**A:** Just increase `--max-gb` parameter:
```bash
# Download 100GB from The Stack
./scripts/download_with_cli.sh the-stack 100
```

### Q: Can I filter by programming language?
**A:** Yes, for The Stack. Edit `simple_download.py`:
```python
"the-stack": {
    "name": "bigcode/the-stack-dedup",
    "data_dir": "data/python",  # Change to other language
}
```

### Q: How do I verify data isn't corrupted?
**A:** Use parquet tools:
```bash
# Install parquet-tools
pip install parquet-tools

# Check each file
for f in /mnt/archive/nanochat/*.parquet; do
    parquet-tools inspect $f >/dev/null && echo "✓ $f" || echo "✗ $f"
done
```

---

## Support and Troubleshooting

### Getting Help
1. Check error messages in terminal output
2. Review this README for common issues
3. Verify disk space: `df -h`
4. Check internet connection: `ping huggingface.co`
5. Review HuggingFace dataset page for known issues

### Common Error Messages

| Error | Solution |
|-------|----------|
| "No module named 'datasets'" | `uv add datasets` |
| "Connection timeout" | Check internet, retry with resume |
| "Authentication required" | `huggingface-cli login` |
| "Disk quota exceeded" | Free disk space, use different directory |
| "Permission denied" | Check directory permissions, use `sudo` if needed |
| "Dataset not found" | Verify dataset name, check HuggingFace Hub |

### Performance Tips
- Use SSD for output directory (10x faster writes)
- Close other network-heavy applications
- Use wired connection instead of WiFi
- Consider using `nice` to lower priority: `nice -n 19 ./script.sh`
- For multiple downloads, run them sequentially on slow connections

---

## License and Attribution

These scripts use HuggingFace datasets. Each dataset has its own license:
- **CodeParrot Clean:** Apache 2.0
- **The Stack:** Multiple licenses (check subset)
- **GitHub Code:** Multiple licenses (check subset)

Always review dataset licenses before use in production.

---

## Changelog

### 2026-02-13
- Initial release
- Two proven download approaches
- Support for 3 major Python datasets
- Resume capability
- Progress tracking
- Comprehensive documentation
