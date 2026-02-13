#!/usr/bin/env python3
"""
Convert downloaded HuggingFace datasets to parquet shards.

This script reads various HuggingFace dataset formats and converts them
to standardized parquet shards suitable for training.

Usage:
    python scripts/convert_to_parquet.py \\
        --input /mnt/archive/nanochat/downloads/codeparrot-clean \\
        --output /mnt/archive/nanochat \\
        --max-gb 8
"""

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Iterator, Dict, Any, Optional

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("\nInstall with: pip install pyarrow tqdm")
    sys.exit(1)


SHARD_SIZE_MB = 100
BYTES_PER_MB = 1024 * 1024

# Common text fields in different datasets
TEXT_FIELDS = ["text", "content", "code", "data"]


class ParquetConverter:
    """Convert downloaded datasets to training parquet format."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        max_gb: float,
        field_name: Optional[str] = None,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_bytes = int(max_gb * 1024 * BYTES_PER_MB)
        self.shard_size = SHARD_SIZE_MB * BYTES_PER_MB
        self.field_name = field_name

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Find all input files
        self.input_files = self._find_input_files()
        if not self.input_files:
            raise ValueError(f"No data files found in {input_dir}")

    def _find_input_files(self) -> list[Path]:
        """Find all parquet, jsonl, and compressed json files in input directory."""
        files = []

        # Look for parquet files
        files.extend(self.input_dir.rglob("*.parquet"))

        # Look for jsonl files
        files.extend(self.input_dir.rglob("*.jsonl"))
        files.extend(self.input_dir.rglob("*.json"))

        # Look for compressed json files
        files.extend(self.input_dir.rglob("*.json.gz"))
        files.extend(self.input_dir.rglob("*.jsonl.gz"))

        return sorted(files)

    def _read_parquet_batches(self, file_path: Path) -> Iterator[Dict[str, Any]]:
        """Read parquet file in batches."""
        try:
            table = pq.read_table(file_path)

            # Detect text field (use user-specified field or auto-detect)
            text_field = None
            if self.field_name and self.field_name in table.column_names:
                text_field = self.field_name
            else:
                for field in TEXT_FIELDS:
                    if field in table.column_names:
                        text_field = field
                        break

            if text_field is None:
                print(f"Warning: No text field found in {file_path}, skipping")
                return

            # Yield rows
            for batch in table.to_batches():
                df = batch.to_pandas()
                for _, row in df.iterrows():
                    text = row.get(text_field, "")
                    if text:
                        yield {"text": str(text)}

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    def _read_jsonl_batches(self, file_path: Path) -> Iterator[Dict[str, Any]]:
        """Read JSONL file line by line."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line)

                        # Find text field (use user-specified field or auto-detect)
                        text = None
                        if self.field_name and self.field_name in data:
                            text = data[self.field_name]
                        else:
                            for field in TEXT_FIELDS:
                                if field in data:
                                    text = data[field]
                                    break

                        if text:
                            yield {"text": str(text)}

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    def _read_gzip_jsonl_batches(self, file_path: Path) -> Iterator[Dict[str, Any]]:
        """Read compressed JSONL file line by line."""
        try:
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line)

                        # Find text field (use user-specified field or auto-detect)
                        text = None
                        if self.field_name and self.field_name in data:
                            text = data[self.field_name]
                        else:
                            for field in TEXT_FIELDS:
                                if field in data:
                                    text = data[field]
                                    break

                        if text:
                            yield {"text": str(text)}

                    except json.JSONDecodeError as e:
                        if line_num <= 10:
                            print(f"Warning: JSON decode error in {file_path.name} line {line_num}: {e}")
                        continue

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    def _get_file_iterator(self, file_path: Path) -> Iterator[Dict[str, Any]]:
        """Get iterator for file based on extension."""
        if file_path.suffix == ".parquet":
            return self._read_parquet_batches(file_path)
        elif file_path.suffix == ".gz":
            # Compressed json/jsonl files
            return self._read_gzip_jsonl_batches(file_path)
        elif file_path.suffix in [".jsonl", ".json"]:
            return self._read_jsonl_batches(file_path)
        else:
            print(f"Warning: Unknown file type {file_path}, skipping")
            return iter([])

    def _write_shard(
        self,
        rows: list[Dict[str, str]],
        shard_num: int,
    ) -> int:
        """Write rows to parquet shard."""
        if not rows:
            return 0

        shard_path = self.output_dir / f"shard_{shard_num:05d}.parquet"

        try:
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, shard_path, compression="snappy")
            return shard_path.stat().st_size

        except Exception as e:
            print(f"\nError writing shard {shard_num}: {e}")
            if shard_path.exists():
                shard_path.unlink()
            raise

    def convert(self) -> bool:
        """Convert all input files to parquet shards."""
        print(f"\n{'=' * 70}")
        print(f"Converting to Parquet Shards")
        print(f"Input dir:  {self.input_dir}")
        print(f"Output dir: {self.output_dir}")
        print(f"Max size:   {self.max_bytes / BYTES_PER_MB / 1024:.2f} GB")
        print(f"Shard size: {SHARD_SIZE_MB} MB")
        print(f"Files:      {len(self.input_files)}")
        if self.field_name:
            print(f"Field:      {self.field_name}")
        print(f"{'=' * 70}\n")

        total_bytes = 0
        total_rows = 0
        shard_num = 0
        current_rows = []
        current_bytes = 0

        # Progress bar
        with tqdm(
            total=self.max_bytes,
            unit="B",
            unit_scale=True,
            desc="Converting",
        ) as pbar:

            try:
                for file_idx, file_path in enumerate(self.input_files, 1):
                    if total_bytes >= self.max_bytes:
                        break

                    # Update progress description to show current file
                    pbar.set_description(f"Converting [{file_idx}/{len(self.input_files)}] {file_path.name}")

                    # Process file
                    for item in self._get_file_iterator(file_path):
                        text = item["text"]
                        text_bytes = len(text.encode("utf-8"))

                        # Add to current batch
                        current_rows.append(item)
                        current_bytes += text_bytes

                        # Write shard if full
                        if current_bytes >= self.shard_size:
                            bytes_written = self._write_shard(current_rows, shard_num)
                            total_bytes += bytes_written
                            total_rows += len(current_rows)
                            pbar.update(bytes_written)

                            shard_num += 1
                            current_rows = []
                            current_bytes = 0

                            # Check if we've hit max size
                            if total_bytes >= self.max_bytes:
                                break

                # Write remaining rows
                if current_rows and total_bytes < self.max_bytes:
                    bytes_written = self._write_shard(current_rows, shard_num)
                    total_bytes += bytes_written
                    total_rows += len(current_rows)
                    pbar.update(bytes_written)
                    shard_num += 1

            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                return False

            except Exception as e:
                print(f"\nError during conversion: {e}")
                return False

        print(f"\n{'=' * 70}")
        print(f"Conversion complete!")
        print(f"Total shards: {shard_num}")
        print(f"Total size:   {total_bytes / BYTES_PER_MB / 1024:.2f} GB")
        print(f"Total rows:   {total_rows:,}")
        print(f"Output:       {self.output_dir}")
        print(f"{'=' * 70}\n")

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace datasets to parquet shards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert codeparrot-clean (compressed JSON files)
    python scripts/convert_to_parquet.py \\
        --input /mnt/archive/nanochat/downloads/codeparrot-clean \\
        --output /mnt/archive/nanochat \\
        --max-gb 8

    # Convert The Stack with custom field name
    python scripts/convert_to_parquet.py \\
        --input /mnt/archive/nanochat/downloads/the-stack \\
        --output /mnt/archive/nanochat \\
        --max-gb 35 \\
        --field-name code

    # Convert parquet or jsonl files (auto-detect format)
    python scripts/convert_to_parquet.py \\
        --input /path/to/dataset \\
        --output /path/to/output \\
        --max-gb 10
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory containing downloaded dataset",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for parquet shards",
    )

    parser.add_argument(
        "--max-gb",
        type=float,
        required=True,
        help="Maximum data to convert in GB",
    )

    parser.add_argument(
        "--field-name",
        type=str,
        help="JSON field name to extract (default: auto-detect from 'text', 'content', 'code', 'data')",
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input.exists():
        print(f"Error: Input directory does not exist: {args.input}")
        sys.exit(1)

    # Create converter
    try:
        converter = ParquetConverter(
            input_dir=args.input,
            output_dir=args.output,
            max_gb=args.max_gb,
            field_name=args.field_name,
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Run conversion
    success = converter.convert()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
