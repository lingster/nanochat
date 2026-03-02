#!/usr/bin/env python3
"""
Simple, proven direct download script for Python training data.

Downloads Python datasets from HuggingFace Hub and converts to parquet format
with minimal dependencies and maximum reliability.

Usage:
    uv run python scripts/simple_download.py --dataset codeparrot-clean --max-gb 8
    uv run python scripts/simple_download.py --dataset the-stack --max-gb 35
    uv run python scripts/simple_download.py --dataset github-code --max-gb 15
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Iterator, Dict, Any
import json

try:
    from datasets import load_dataset, IterableDataset
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("\nInstall with: uv add datasets pyarrow tqdm")
    sys.exit(1)


# Dataset configurations
DATASETS = {
    "codeparrot-clean": {
        "name": "codeparrot/codeparrot-clean",
        "split": "train",
        "streaming": True,
        "text_field": "content",
    },
    "the-stack": {
        "name": "bigcode/the-stack-dedup",
        "split": "train",
        "streaming": True,
        "text_field": "content",
        "data_dir": "data/python",  # Python subset only
    },
    "github-code": {
        "name": "codeparrot/github-code",
        "split": "train",
        "streaming": True,
        "text_field": "code",
        "data_dir": "Python",
    },
}

SHARD_SIZE_MB = 100
BYTES_PER_MB = 1024 * 1024


class SimpleDownloader:
    """Simple, reliable dataset downloader with parquet conversion."""

    def __init__(
        self,
        dataset_key: str,
        output_dir: Path,
        max_gb: float,
        resume: bool = True,
    ):
        if dataset_key not in DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_key}. "
                f"Available: {list(DATASETS.keys())}"
            )

        self.dataset_key = dataset_key
        self.config = DATASETS[dataset_key]
        self.output_dir = output_dir
        self.max_bytes = int(max_gb * 1024 * BYTES_PER_MB)
        self.resume = resume
        self.shard_size = SHARD_SIZE_MB * BYTES_PER_MB

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Resume tracking
        self.progress_file = self.output_dir / f".{dataset_key}_progress.json"
        self.current_shard = 0
        self.total_bytes = 0

        if resume and self.progress_file.exists():
            self._load_progress()

    def _load_progress(self):
        """Load progress from previous run."""
        try:
            with open(self.progress_file, "r") as f:
                progress = json.load(f)
                self.current_shard = progress.get("current_shard", 0)
                self.total_bytes = progress.get("total_bytes", 0)
                print(f"Resuming from shard {self.current_shard} "
                      f"({self.total_bytes / BYTES_PER_MB / 1024:.2f} GB)")
        except Exception as e:
            print(f"Warning: Could not load progress: {e}")
            self.current_shard = 0
            self.total_bytes = 0

    def _save_progress(self):
        """Save current progress."""
        try:
            with open(self.progress_file, "w") as f:
                json.dump(
                    {
                        "current_shard": self.current_shard,
                        "total_bytes": self.total_bytes,
                    },
                    f,
                )
        except Exception as e:
            print(f"Warning: Could not save progress: {e}")

    def _get_shard_path(self, shard_num: int) -> Path:
        """Get path for shard file."""
        return self.output_dir / f"{self.dataset_key}_shard_{shard_num:05d}.parquet"

    def _stream_to_shard(
        self,
        iterator: Iterator[Dict[str, Any]],
        shard_num: int,
        pbar: tqdm,
    ) -> tuple[int, int]:
        """
        Stream data to a single parquet shard.

        Returns:
            (bytes_written, num_rows)
        """
        shard_path = self._get_shard_path(shard_num)

        # Skip if already exists and resuming
        if self.resume and shard_path.exists():
            size = shard_path.stat().st_size
            pbar.update(size)
            return size, 0

        text_field = self.config["text_field"]
        rows = []
        shard_bytes = 0

        try:
            for item in iterator:
                # Extract text content
                text = item.get(text_field, "")
                if not text:
                    continue

                # Estimate size (approximate)
                text_bytes = len(text.encode("utf-8"))
                shard_bytes += text_bytes

                rows.append({"text": text})

                # Check if shard is full
                if shard_bytes >= self.shard_size:
                    break

            # Write shard if we have data
            if rows:
                table = pa.Table.from_pylist(rows)
                pq.write_table(table, shard_path, compression="snappy")

                actual_size = shard_path.stat().st_size
                pbar.update(actual_size)

                return actual_size, len(rows)

            return 0, 0

        except Exception as e:
            print(f"\nError writing shard {shard_num}: {e}")
            # Clean up partial file
            if shard_path.exists():
                shard_path.unlink()
            raise

    def download(self):
        """Download and convert dataset to parquet shards."""
        print(f"\n{'=' * 70}")
        print(f"Downloading: {self.config['name']}")
        print(f"Output dir: {self.output_dir}")
        print(f"Max size: {self.max_bytes / BYTES_PER_MB / 1024:.2f} GB")
        print(f"Shard size: {SHARD_SIZE_MB} MB")
        print(f"{'=' * 70}\n")

        # Load dataset in streaming mode
        print("Loading dataset stream...")
        try:
            load_kwargs = {
                "path": self.config["name"],
                "split": self.config["split"],
                "streaming": True,
            }

            # Add data_dir if specified (for subsets)
            if "data_dir" in self.config:
                load_kwargs["data_dir"] = self.config["data_dir"]

            dataset = load_dataset(**load_kwargs)

            if not isinstance(dataset, IterableDataset):
                print("Error: Dataset is not iterable. Streaming may not be supported.")
                return False

        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("\nTroubleshooting:")
            print("1. Check internet connection")
            print("2. Verify dataset name is correct")
            print("3. Try: huggingface-cli login")
            return False

        # Create progress bar
        with tqdm(
            total=self.max_bytes,
            initial=self.total_bytes,
            unit="B",
            unit_scale=True,
            desc="Downloading",
        ) as pbar:

            iterator = iter(dataset)
            shard_num = self.current_shard
            total_rows = 0

            while self.total_bytes < self.max_bytes:
                try:
                    bytes_written, rows = self._stream_to_shard(
                        iterator, shard_num, pbar
                    )

                    if bytes_written == 0:
                        print("\nDataset exhausted before reaching max size")
                        break

                    self.total_bytes += bytes_written
                    total_rows += rows
                    shard_num += 1
                    self.current_shard = shard_num

                    # Save progress periodically
                    if shard_num % 10 == 0:
                        self._save_progress()

                except KeyboardInterrupt:
                    print("\n\nInterrupted by user. Progress saved.")
                    self._save_progress()
                    return False

                except Exception as e:
                    print(f"\nError: {e}")
                    self._save_progress()
                    return False

        # Final progress save
        self._save_progress()

        print(f"\n{'=' * 70}")
        print(f"Download complete!")
        print(f"Total shards: {shard_num}")
        print(f"Total size: {self.total_bytes / BYTES_PER_MB / 1024:.2f} GB")
        print(f"Total rows: {total_rows:,}")
        print(f"Output: {self.output_dir}")
        print(f"{'=' * 70}\n")

        # Clean up progress file on successful completion
        if self.total_bytes >= self.max_bytes and self.progress_file.exists():
            self.progress_file.unlink()

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Simple Python training data downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download 8GB of codeparrot-clean
    uv run python scripts/simple_download.py --dataset codeparrot-clean --max-gb 8

    # Download 35GB of The Stack (Python only)
    uv run python scripts/simple_download.py --dataset the-stack --max-gb 35

    # Download to custom directory
    uv run python scripts/simple_download.py --dataset github-code \\
        --output-dir /custom/path --max-gb 15

    # Resume interrupted download
    uv run python scripts/simple_download.py --dataset codeparrot-clean --max-gb 8

Available datasets:
    codeparrot-clean  : Clean Python code from GitHub (~8GB available)
    the-stack         : The Stack Dedup Python subset (~35GB available)
    github-code       : GitHub Code Python (~15GB available)
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASETS.keys()),
        help="Dataset to download",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/archive/nanochat"),
        help="Output directory for parquet files (default: /mnt/archive/nanochat)",
    )

    parser.add_argument(
        "--max-gb",
        type=float,
        required=True,
        help="Maximum data to download in GB",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start from scratch instead of resuming",
    )

    args = parser.parse_args()

    # Create downloader
    downloader = SimpleDownloader(
        dataset_key=args.dataset,
        output_dir=args.output_dir,
        max_gb=args.max_gb,
        resume=not args.no_resume,
    )

    # Run download
    success = downloader.download()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
