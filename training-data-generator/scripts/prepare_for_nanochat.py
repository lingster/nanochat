#!/usr/bin/env python3
"""
Prepare collected Python training data for nanochat training.

This script:
1. Processes GitHub Code files (rename 'content' column to 'text')
2. Combines CodeParrot, GitHub Code, and Web Scraping data
3. Creates properly named shards (shard_00000.parquet, etc.)
4. Reserves last shard for validation
5. Outputs to nanochat's expected data directory

Usage:
    python prepare_for_nanochat.py --output-dir ~/.cache/nanochat/custom_python_data
"""

import os
import sys
import argparse
import glob
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

def prepare_github_code_shard(input_file, output_file):
    """
    Read GitHub Code parquet, extract 'content' column, rename to 'text', write out.
    """
    print(f"Processing: {os.path.basename(input_file)}")

    # Read the file
    table = pq.read_table(input_file)

    # Extract content column and rename to text
    if 'content' in table.column_names:
        content_col = table.column('content')
        # Create new table with just the text column
        new_table = pa.table({'text': content_col})

        # Write to output
        pq.write_table(new_table, output_file, compression='snappy')

        rows = len(new_table)
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  ✓ {rows:,} rows, {size_mb:.1f} MB")
        return rows
    else:
        print(f"  ✗ Error: 'content' column not found")
        return 0

def copy_with_rename(input_file, output_file):
    """Copy parquet file (already has 'text' column)"""
    print(f"Copying: {os.path.basename(input_file)}")
    table = pq.read_table(input_file)
    pq.write_table(table, output_file, compression='snappy')
    rows = len(table)
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  ✓ {rows:,} rows, {size_mb:.1f} MB")
    return rows

def main():
    parser = argparse.ArgumentParser(description='Prepare collected data for nanochat training')
    parser.add_argument('--source-dir', type=str, default='/mnt/archive/nanochat',
                       help='Source directory with collected data')
    parser.add_argument('--output-dir', type=str,
                       default=os.path.expanduser('~/.cache/nanochat/custom_python_data'),
                       help='Output directory for nanochat training data')
    parser.add_argument('--val-shards', type=int, default=1,
                       help='Number of shards to reserve for validation (default: 1)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"Preparing Python Training Data for nanochat")
    print(f"{'='*70}")
    print(f"Source: {args.source_dir}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*70}\n")

    # Track statistics
    total_rows = 0
    total_size = 0
    shard_idx = 0

    # Process data sources in order
    sources = [
        ('CodeParrot Clean', f'{args.source_dir}/codeparrot-clean-parquet/*.parquet', 'copy'),
        ('GitHub Code', f'{args.source_dir}/downloads/github-code/data/*.parquet', 'github'),
        ('Web Scraping', f'{args.source_dir}/web_scraped_data/*.parquet', 'copy'),
    ]

    for source_name, pattern, mode in sources:
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"⚠️  No files found for {source_name}: {pattern}")
            continue

        print(f"\n📁 Processing {source_name} ({len(files)} files)")
        print(f"{'─'*70}")

        for input_file in files:
            output_file = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.parquet")

            try:
                if mode == 'github':
                    rows = prepare_github_code_shard(input_file, output_file)
                else:  # mode == 'copy'
                    rows = copy_with_rename(input_file, output_file)

                total_rows += rows
                total_size += os.path.getsize(output_file)
                shard_idx += 1

            except Exception as e:
                print(f"  ✗ Error processing {os.path.basename(input_file)}: {e}")
                continue

    print(f"\n{'='*70}")
    print(f"📊 SUMMARY")
    print(f"{'='*70}")
    print(f"Total shards created: {shard_idx}")
    print(f"Total rows: {total_rows:,}")
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    print(f"Training shards: {shard_idx - args.val_shards} (shard_00000 to shard_{shard_idx - args.val_shards - 1:05d})")
    print(f"Validation shards: {args.val_shards} (shard_{shard_idx - args.val_shards:05d} to shard_{shard_idx - 1:05d})")
    print(f"{'='*70}")

    # Verify the data is loadable
    print(f"\n🔍 Verifying data format...")
    try:
        test_file = os.path.join(args.output_dir, "shard_00000.parquet")
        table = pq.read_table(test_file)
        assert 'text' in table.column_names, "Missing 'text' column"
        print(f"✓ Format verified: {len(table):,} rows, columns: {table.column_names}")

        # Show sample
        sample = table.slice(0, 1).to_pydict()['text'][0]
        print(f"\n📝 Sample text (first 200 chars):")
        print(f"   {sample[:200]}...")

    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return 1

    print(f"\n✅ Data preparation complete!")
    print(f"\n💡 Next steps:")
    print(f"   1. Set NANOCHAT_BASE_DIR environment variable:")
    print(f"      export NANOCHAT_BASE_DIR={os.path.dirname(args.output_dir)}")
    print(f"   2. Update speedrun script to use 'custom_python_data' instead of 'base_data'")
    print(f"   3. Run: bash speedrun_2xa6000_python.sh")
    print()

    return 0

if __name__ == '__main__':
    sys.exit(main())
