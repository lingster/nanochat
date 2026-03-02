#!/usr/bin/env python3
"""Download and convert HuggingFace code datasets to parquet format.

This script downloads large-scale Python code datasets from HuggingFace Hub
and converts them to parquet format matching the nanochat pretraining format.

Usage:
    uv run python scripts/download_huggingface_datasets.py \
        --dataset the-stack-python \
        --max-size-gb 30 \
        --output-dir /mnt/archive/nanochat/huggingface_data

Features:
    - Resume capability if interrupted
    - Progress tracking with file sizes
    - Data cleaning and deduplication
    - Automatic shard generation (100MB each)
    - Content validation and filtering
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm


@dataclass
class DatasetConfig:
    """Configuration for a specific HuggingFace dataset."""

    name: str
    hf_name: str
    subset: Optional[str]
    split: str
    text_field: str
    language_field: Optional[str] = None
    description: str = ""
    estimated_size_gb: float = 0.0


# Dataset configurations
DATASET_CONFIGS = {
    "the-stack-python": DatasetConfig(
        name="the-stack-python",
        hf_name="bigcode/the-stack-dedup",
        subset="data",
        split="train",
        text_field="content",
        language_field="lang",
        description="Python code from The Stack (deduplicated)",
        estimated_size_gb=35.0,
    ),
    "codeparrot-clean": DatasetConfig(
        name="codeparrot-clean",
        hf_name="codeparrot/codeparrot-clean",
        subset=None,
        split="train",
        text_field="content",
        description="Cleaned Python code from GitHub",
        estimated_size_gb=8.0,
    ),
    "github-code-python": DatasetConfig(
        name="github-code-python",
        hf_name="codeparrot/github-code",
        subset=None,
        split="train",
        text_field="code",
        language_field="language",
        description="Python code from GitHub repositories",
        estimated_size_gb=15.0,
    ),
    "python-code-instructions": DatasetConfig(
        name="python-code-instructions",
        hf_name="iamtarun/python_code_instructions_18k_alpaca",
        subset=None,
        split="train",
        text_field="output",
        description="Python code instruction-following dataset",
        estimated_size_gb=0.5,
    ),
}


class StateManager:
    """Manage download state for resumption."""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Load state from disk."""
        if self.state_file.exists():
            with open(self.state_file, "r") as f:
                return json.load(f)
        return {
            "processed_hashes": [],
            "total_items": 0,
            "total_bytes": 0,
            "current_shard": 0,
        }

    def save_state(self):
        """Save state to disk."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def is_processed(self, content_hash: str) -> bool:
        """Check if content has been processed."""
        return content_hash in self.state["processed_hashes"]

    def mark_processed(self, content_hash: str):
        """Mark content as processed."""
        if content_hash not in self.state["processed_hashes"]:
            self.state["processed_hashes"].append(content_hash)

    def get_current_shard(self) -> int:
        """Get current shard index."""
        return self.state.get("current_shard", 0)

    def increment_shard(self):
        """Increment shard counter."""
        self.state["current_shard"] = self.state.get("current_shard", 0) + 1

    def update_stats(self, items: int, bytes_count: int):
        """Update download statistics."""
        self.state["total_items"] = self.state.get("total_items", 0) + items
        self.state["total_bytes"] = self.state.get("total_bytes", 0) + bytes_count


class DataCleaner:
    """Clean and validate text data."""

    MIN_LENGTH = 100
    MAX_LENGTH = 1_000_000

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = " ".join(text.split())

        # Remove null bytes
        text = text.replace("\x00", "")

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    @staticmethod
    def is_valid(text: str) -> bool:
        """Check if text meets quality criteria."""
        if not text:
            return False

        # Length check
        if len(text) < DataCleaner.MIN_LENGTH or len(text) > DataCleaner.MAX_LENGTH:
            return False

        # Check for sufficient alphanumeric content
        alphanumeric = sum(c.isalnum() for c in text)
        if alphanumeric / len(text) < 0.3:
            return False

        # Check for reasonable line structure (for code)
        lines = text.split("\n")
        if len(lines) < 3:
            return False

        return True

    @staticmethod
    def compute_hash(text: str) -> str:
        """Compute content hash for deduplication."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ParquetWriter:
    """Write data to parquet shards."""

    def __init__(
        self,
        output_dir: Path,
        shard_size_mb: int = 100,
        compression: str = "zstd",
    ):
        self.output_dir = output_dir
        self.shard_size_bytes = shard_size_mb * 1024 * 1024
        self.compression = compression
        self.current_shard_data: List[str] = []
        self.current_shard_bytes = 0

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add_text(self, text: str):
        """Add text to current shard."""
        text_bytes = len(text.encode("utf-8"))
        self.current_shard_data.append(text)
        self.current_shard_bytes += text_bytes

    def should_flush(self) -> bool:
        """Check if current shard should be flushed."""
        return self.current_shard_bytes >= self.shard_size_bytes

    def flush(self, shard_index: int) -> int:
        """Write current shard to disk and return number of items written."""
        if not self.current_shard_data:
            return 0

        # Create parquet table
        table = pa.table({"text": self.current_shard_data})

        # Write to file
        output_file = self.output_dir / f"shard_{shard_index:05d}.parquet"
        pq.write_table(
            table,
            output_file,
            compression=self.compression,
            row_group_size=1024,
        )

        # Get stats
        num_items = len(self.current_shard_data)
        file_size = output_file.stat().st_size

        # Reset shard data
        self.current_shard_data = []
        self.current_shard_bytes = 0

        return num_items


class HuggingFaceDownloader:
    """Download and process HuggingFace datasets."""

    def __init__(
        self,
        dataset_name: str,
        output_dir: Path,
        max_size_gb: Optional[float] = None,
        resume: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.max_size_bytes = (
            int(max_size_gb * 1024 * 1024 * 1024) if max_size_gb else None
        )
        self.resume = resume
        self.logger = logger or logging.getLogger(__name__)

        # Get dataset config
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {', '.join(DATASET_CONFIGS.keys())}"
            )
        self.config = DATASET_CONFIGS[dataset_name]

        # Initialize components
        self.cleaner = DataCleaner()
        self.writer = ParquetWriter(output_dir)

        # State management
        state_file = output_dir / f".{dataset_name}_state.json"
        self.state = StateManager(state_file)

        # Statistics
        self.stats = {
            "downloaded": 0,
            "skipped_duplicate": 0,
            "skipped_invalid": 0,
            "total_bytes": 0,
        }

    def _filter_python(self, example: Dict) -> bool:
        """Filter for Python content if language field exists."""
        if self.config.language_field:
            lang = example.get(self.config.language_field, "").lower()
            return lang in ["python", "py"]
        return True

    def _extract_text(self, example: Dict) -> Optional[str]:
        """Extract text content from example."""
        text = example.get(self.config.text_field)
        if not text:
            return None

        # Clean text
        text = self.cleaner.clean_text(text)

        # Validate
        if not self.cleaner.is_valid(text):
            return None

        return text

    def _should_continue(self) -> bool:
        """Check if download should continue based on size limits."""
        if self.max_size_bytes is None:
            return True
        return self.state.state["total_bytes"] < self.max_size_bytes

    def download(self):
        """Download and process dataset."""
        self.logger.info(f"Downloading dataset: {self.config.name}")
        self.logger.info(f"HuggingFace dataset: {self.config.hf_name}")
        self.logger.info(f"Output directory: {self.output_dir}")

        if self.resume and self.state.state["total_items"] > 0:
            self.logger.info(
                f"Resuming from previous run: "
                f"{self.state.state['total_items']} items, "
                f"{self.state.state['total_bytes'] / (1024**3):.2f} GB"
            )

        # Load dataset with streaming
        self.logger.info("Loading dataset from HuggingFace Hub...")
        try:
            if self.config.subset:
                dataset = load_dataset(
                    self.config.hf_name,
                    data_dir=self.config.subset,
                    split=self.config.split,
                    streaming=True,
                )
            else:
                dataset = load_dataset(
                    self.config.hf_name,
                    split=self.config.split,
                    streaming=True,
                )
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise

        # Process dataset
        self.logger.info("Processing dataset...")
        shard_index = self.state.get_current_shard()
        seen_hashes: Set[str] = set(self.state.state["processed_hashes"])

        # Note: We don't pre-filter for Python on streaming datasets
        # because it can cause the iterator to hang. Instead, we filter
        # inline during iteration for better progress tracking.

        with tqdm(desc="Processing", unit=" items") as pbar:
            for example in dataset:
                # Check size limit
                if not self._should_continue():
                    self.logger.info("Reached size limit, stopping download")
                    break

                # Filter for Python content inline (if language field exists)
                if self.config.language_field:
                    if not self._filter_python(example):
                        self.stats["skipped_invalid"] += 1
                        pbar.update(1)
                        continue

                # Extract text
                text = self._extract_text(example)
                if not text:
                    self.stats["skipped_invalid"] += 1
                    pbar.update(1)
                    continue

                # Compute hash for deduplication
                content_hash = self.cleaner.compute_hash(text)

                # Skip duplicates
                if content_hash in seen_hashes:
                    self.stats["skipped_duplicate"] += 1
                    continue

                # Add to shard
                self.writer.add_text(text)
                seen_hashes.add(content_hash)
                self.state.mark_processed(content_hash)

                # Update stats
                text_bytes = len(text.encode("utf-8"))
                self.stats["downloaded"] += 1
                self.stats["total_bytes"] += text_bytes

                # Update progress
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "shard": shard_index,
                        "size_gb": self.stats["total_bytes"] / (1024**3),
                    }
                )

                # Flush shard if needed
                if self.writer.should_flush():
                    items_written = self.writer.flush(shard_index)
                    self.state.update_stats(items_written, text_bytes)
                    self.state.increment_shard()
                    self.state.save_state()
                    shard_index += 1

                    self.logger.info(
                        f"Wrote shard {shard_index - 1}: {items_written} items"
                    )

                # Periodic state save
                if self.stats["downloaded"] % 100 == 0:
                    self.state.save_state()

        # Flush remaining data
        if self.writer.current_shard_data:
            items_written = self.writer.flush(shard_index)
            self.state.update_stats(items_written, self.writer.current_shard_bytes)
            self.logger.info(f"Wrote final shard {shard_index}: {items_written} items")

        # Save final state
        self.state.save_state()

        # Print summary
        self._print_summary()

    def _print_summary(self):
        """Print download summary."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Download Summary")
        self.logger.info("=" * 60)
        self.logger.info(f"Dataset: {self.config.name}")
        self.logger.info(f"Items downloaded: {self.stats['downloaded']:,}")
        self.logger.info(f"Items skipped (invalid): {self.stats['skipped_invalid']:,}")
        self.logger.info(
            f"Items skipped (duplicate): {self.stats['skipped_duplicate']:,}"
        )
        self.logger.info(
            f"Total size: {self.stats['total_bytes'] / (1024**3):.2f} GB"
        )
        self.logger.info(f"Output directory: {self.output_dir}")

        # Count shards
        shards = list(self.output_dir.glob("shard_*.parquet"))
        self.logger.info(f"Shards created: {len(shards)}")
        self.logger.info("=" * 60)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("hf_downloader")
    logger.setLevel(log_level)

    # Console handler
    handler = logging.StreamHandler()
    handler.setLevel(log_level)

    # Format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


def list_datasets():
    """List available datasets."""
    print("\nAvailable Datasets:")
    print("=" * 80)

    for name, config in DATASET_CONFIGS.items():
        print(f"\nDataset: {name}")
        print(f"  HuggingFace: {config.hf_name}")
        print(f"  Description: {config.description}")
        print(f"  Estimated Size: {config.estimated_size_gb:.1f} GB")
        print(f"  Text Field: {config.text_field}")
        if config.language_field:
            print(f"  Language Field: {config.language_field}")

    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download Python code datasets from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download The Stack (Python) up to 30GB
  python scripts/download_huggingface_datasets.py \\
      --dataset the-stack-python \\
      --max-size-gb 30 \\
      --output-dir /mnt/archive/nanochat/huggingface_data

  # Download CodeParrot without size limit
  python scripts/download_huggingface_datasets.py \\
      --dataset codeparrot-clean \\
      --output-dir /mnt/archive/nanochat/huggingface_data

  # List available datasets
  python scripts/download_huggingface_datasets.py --list
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to download",
    )

    parser.add_argument(
        "--max-size-gb",
        type=float,
        help="Maximum download size in GB (default: no limit)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/archive/nanochat/huggingface_data",
        help="Output directory for parquet files (default: /mnt/archive/nanochat/huggingface_data)",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore previous state (default: resume)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit",
    )

    args = parser.parse_args()

    # List datasets if requested
    if args.list:
        list_datasets()
        sys.exit(0)

    # Validate arguments
    if not args.dataset:
        parser.error("--dataset is required (or use --list to see available datasets)")

    # Set up logging
    logger = setup_logging(args.log_level)

    # Create output directory
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("HuggingFace Dataset Downloader")
    logger.info("=" * 60)

    try:
        # Create downloader
        downloader = HuggingFaceDownloader(
            dataset_name=args.dataset,
            output_dir=output_dir,
            max_size_gb=args.max_size_gb,
            resume=not args.no_resume,
            logger=logger,
        )

        # Download
        downloader.download()

        logger.info("\nDownload completed successfully!")

    except KeyboardInterrupt:
        logger.warning("\nDownload interrupted by user")
        logger.info("Run the same command again to resume")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
