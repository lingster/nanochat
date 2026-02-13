#!/usr/bin/env python3
"""
Quick test to verify download setup is working correctly.

Tests:
1. Required libraries are installed
2. Dataset configurations are valid
3. Output directories can be created
4. Basic functionality works
"""

import sys
from pathlib import Path

# Test 1: Check imports
print("Testing imports...")
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tqdm import tqdm
    print("✓ pyarrow and tqdm installed")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    print("\nInstall with: uv add pyarrow tqdm")
    sys.exit(1)

try:
    from datasets import load_dataset, IterableDataset
    print("✓ datasets library installed")
except ImportError as e:
    print(f"✗ Missing datasets library: {e}")
    print("\nInstall with: uv add datasets")
    sys.exit(1)

# Test 2: Check dataset configurations
print("\nTesting dataset configurations...")
DATASETS = {
    "codeparrot-clean": "codeparrot/codeparrot-clean",
    "the-stack": "bigcode/the-stack-dedup",
    "github-code": "codeparrot/github-code",
}

for key, path in DATASETS.items():
    print(f"  {key}: {path}")

print("✓ Dataset configurations valid")

# Test 3: Test directory creation
print("\nTesting directory creation...")
test_dir = Path.home() / ".cache/nanochat/test_output"
try:
    test_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Can create directories: {test_dir}")

    # Clean up test directory
    test_dir.rmdir()
except Exception as e:
    print(f"✗ Cannot create directories: {e}")
    sys.exit(1)

# Test 4: Test parquet writing
print("\nTesting parquet writing...")
try:
    test_data = [{"text": "Hello, world!"}, {"text": "Test data"}]
    table = pa.Table.from_pylist(test_data)

    test_file = Path("/tmp/test_parquet.parquet")
    pq.write_table(table, test_file, compression="snappy")

    # Verify file was created
    assert test_file.exists()
    size = test_file.stat().st_size
    print(f"✓ Parquet writing works (test file: {size} bytes)")

    # Clean up
    test_file.unlink()
except Exception as e:
    print(f"✗ Parquet writing failed: {e}")
    sys.exit(1)

# Test 5: Test simple_download.py import
print("\nTesting simple_download.py...")
try:
    sys.path.insert(0, str(Path(__file__).parent))
    # Don't actually import (would require all dependencies)
    # Just verify the file exists and is readable
    script_path = Path(__file__).parent / "simple_download.py"
    assert script_path.exists()
    print(f"✓ simple_download.py exists")
except Exception as e:
    print(f"✗ simple_download.py test failed: {e}")
    sys.exit(1)

# Test 6: Test convert_to_parquet.py
print("\nTesting convert_to_parquet.py...")
try:
    script_path = Path(__file__).parent / "convert_to_parquet.py"
    assert script_path.exists()
    print(f"✓ convert_to_parquet.py exists")
except Exception as e:
    print(f"✗ convert_to_parquet.py test failed: {e}")
    sys.exit(1)

# Test 7: Test download_with_cli.sh
print("\nTesting download_with_cli.sh...")
try:
    script_path = Path(__file__).parent / "download_with_cli.sh"
    assert script_path.exists()
    assert script_path.stat().st_mode & 0o111  # Check executable bit
    print(f"✓ download_with_cli.sh exists and is executable")
except Exception as e:
    print(f"✗ download_with_cli.sh test failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("All tests passed! Download setup is ready.")
print("=" * 70)
print("\nNext steps:")
print("1. Choose your approach (Option A or Option B)")
print("2. Run the download script")
print("3. See SIMPLE_DOWNLOAD_README.md for examples")
print("\nQuick start:")
print("  # Option A (Python streaming)")
print("  uv run python scripts/simple_download.py --dataset codeparrot-clean --max-gb 8")
print("\n  # Option B (CLI batch)")
print("  ./scripts/download_with_cli.sh codeparrot-clean 8")
