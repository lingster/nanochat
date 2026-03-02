# Quick Start: Download 50GB Python Training Data

## 1. Verify Setup (30 seconds)
```bash
cd /home/ling/workarea/nanochat/training-data-generator
python scripts/test_download_setup.py
```

## 2. Choose Your Approach

### Option A: Python Streaming (Recommended)
**Best if you have limited disk space**

```bash
# Download 8GB clean Python code (~15 min)
uv run python scripts/simple_download.py --dataset codeparrot-clean --max-gb 8

# Download 35GB from The Stack (~60 min)
uv run python scripts/simple_download.py --dataset the-stack --max-gb 35

# Download 15GB GitHub code (~25 min)
uv run python scripts/simple_download.py --dataset github-code --max-gb 15
```

### Option B: HuggingFace CLI (Fastest)
**Best if you have plenty of disk space**

```bash
# Download 8GB (~13 min)
./scripts/download_with_cli.sh codeparrot-clean 8

# Download 35GB (~55 min)
./scripts/download_with_cli.sh the-stack 35

# Download 15GB (~20 min)
./scripts/download_with_cli.sh github-code 15
```

## 3. Complete 50GB Strategy

### Strategy 1: High Quality Mix (58GB total, ~100 min)
```bash
uv run python scripts/simple_download.py --dataset codeparrot-clean --max-gb 8
./scripts/download_with_cli.sh the-stack 35
uv run python scripts/simple_download.py --dataset github-code --max-gb 15
```

### Strategy 2: Single Large Dataset (50GB, ~55 min)
```bash
./scripts/download_with_cli.sh the-stack 50
```

## 4. Verify Download
```bash
# Check size
du -sh /mnt/archive/nanochat/

# Count files
ls -1 /mnt/archive/nanochat/*.parquet | wc -l

# Inspect sample
python -c "
import pyarrow.parquet as pq
table = pq.read_table('$(ls /mnt/archive/nanochat/*.parquet | head -1)')
print(f'Rows: {table.num_rows:,}')
print(f'Schema: {table.schema}')
print(f'Sample: {table.to_pandas().head(1)["text"].values[0][:200]}...')
"
```

## Quick Reference

| Feature | Option A (Python) | Option B (CLI) |
|---------|-------------------|----------------|
| **Speed** | Good | Excellent |
| **Disk Space** | Low (1.2x) | High (2x peak) |
| **Resume** | Automatic | Rerun command |
| **Best For** | <20GB datasets | >30GB datasets |

## Need Help?

- **Full docs:** `scripts/SIMPLE_DOWNLOAD_README.md` (602 lines)
- **Summary:** `SIMPLE_DOWNLOAD_SUMMARY.md`
- **Troubleshooting:** See README sections

## Common Issues

```bash
# Missing datasets library
uv add datasets

# Authentication needed
huggingface-cli login

# Out of disk space
df -h /mnt/archive/nanochat/  # Check available space
OUTPUT_DIR=/mnt/archive/python ./scripts/download_with_cli.sh the-stack 35

# Slow download
# Try Option B instead of Option A (20-30% faster)
```

## Output Location

Default: `/mnt/archive/nanochat/`

Custom location:
```bash
# Option A
uv run python scripts/simple_download.py \
    --dataset codeparrot-clean \
    --output-dir /custom/path \
    --max-gb 8

# Option B
OUTPUT_DIR=/custom/path ./scripts/download_with_cli.sh codeparrot-clean 8
```

---

**Total time to 50GB: ~100 minutes (100 Mbps) or ~30 minutes (1 Gbps)**
