#!/usr/bin/env python3
"""
Python Dataset Builder - Main Orchestration Script

This script orchestrates the creation of a Python-focused training dataset for
the nanochat LLM by:
1. Scraping documentation from PyPI packages
2. Creating datasets from GitHub PR history
3. Combining and formatting data for training

Usage:
    python -m tools.python_dataset_builder.build_dataset --config config.yaml
    python -m tools.python_dataset_builder.build_dataset --pypi-only
    python -m tools.python_dataset_builder.build_dataset --github-only
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.python_dataset_builder.pypi_scraper import (
    scrape_pypi_docs,
    create_documentation_conversations,
    save_scraped_pages,
    save_conversations_jsonl as save_pypi_conversations,
)
from tools.python_dataset_builder.github_pr_dataset import (
    create_pr_dataset,
    save_prs_json,
    save_conversations_jsonl as save_pr_conversations,
)

logger = logging.getLogger(__name__)


def setup_logging(config: dict):
    """Configure logging based on config."""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper())

    handlers = [logging.StreamHandler()]

    if log_config.get("file"):
        handlers.append(logging.FileHandler(log_config["file"]))

    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if not log_config.get("timestamps", True):
        format_str = "%(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers
    )


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def expand_path(path: str) -> Path:
    """Expand ~ and environment variables in path."""
    return Path(os.path.expandvars(os.path.expanduser(path)))


def deduplicate_conversations(
    conversations: list[dict],
    threshold: float = 0.95
) -> list[dict]:
    """
    Remove near-duplicate conversations based on content hash.

    Uses simple hash-based deduplication. For threshold < 1.0,
    would need more sophisticated similarity comparison.
    """
    seen_hashes = set()
    unique_conversations = []

    for conv in conversations:
        # Create hash from message content
        content = json.dumps(conv["messages"], sort_keys=True)
        content_hash = hashlib.md5(content.encode()).hexdigest()

        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_conversations.append(conv)

    logger.info(
        f"Deduplication: {len(conversations)} -> {len(unique_conversations)} "
        f"({len(conversations) - len(unique_conversations)} removed)"
    )

    return unique_conversations


def split_train_test(
    conversations: list[dict],
    train_ratio: float = 0.9,
    seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Split conversations into train and test sets."""
    random.seed(seed)
    shuffled = conversations.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train = shuffled[:split_idx]
    test = shuffled[split_idx:]

    logger.info(f"Split: {len(train)} train, {len(test)} test")

    return train, test


def save_final_dataset(
    conversations: list[dict],
    output_dir: Path,
    formats: list[str],
    parquet_compression: str = "zstd",
    parquet_compression_level: int = 3
):
    """Save the final dataset in specified formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if "jsonl" in formats:
        # Save as JSONL (compatible with customjson.py task)
        jsonl_path = output_dir / "conversations.jsonl"
        with open(jsonl_path, "w") as f:
            for conv in conversations:
                messages = conv["messages"]
                f.write(json.dumps(messages) + "\n")
        logger.info(f"Saved JSONL: {jsonl_path}")

        # Also save with metadata for reference
        full_jsonl_path = output_dir / "conversations_with_metadata.jsonl"
        with open(full_jsonl_path, "w") as f:
            for conv in conversations:
                f.write(json.dumps(conv) + "\n")
        logger.info(f"Saved full JSONL: {full_jsonl_path}")

    if "parquet" in formats:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            # Convert to columnar format
            data = {
                "messages": [json.dumps(c["messages"]) for c in conversations],
                "source": [c.get("metadata", {}).get("source", "unknown") for c in conversations],
            }

            table = pa.table(data)
            parquet_path = output_dir / "conversations.parquet"

            pq.write_table(
                table,
                parquet_path,
                compression=parquet_compression,
                compression_level=parquet_compression_level
            )
            logger.info(f"Saved Parquet: {parquet_path}")

        except ImportError:
            logger.warning("pyarrow not installed, skipping parquet output")


def save_dataset_info(
    output_dir: Path,
    stats: dict,
    config: dict
):
    """Save dataset metadata and statistics."""
    info = {
        "created_at": datetime.now().isoformat(),
        "statistics": stats,
        "config_summary": {
            "pypi_packages": len(config.get("pypi_docs", {}).get("packages", [])),
            "github_repos": len(config.get("github_prs", {}).get("repositories", [])),
        }
    }

    info_path = output_dir / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    logger.info(f"Saved dataset info: {info_path}")


async def run_pypi_scraper(config: dict, output_dir: Path) -> list[dict]:
    """Run the PyPI documentation scraper."""
    pypi_config = config.get("pypi_docs", {})

    if not pypi_config.get("enabled", True):
        logger.info("PyPI scraping disabled in config")
        return []

    packages = pypi_config.get("packages", [])
    if not packages:
        logger.warning("No packages configured for PyPI scraping")
        return []

    logger.info(f"Scraping documentation for {len(packages)} packages...")

    # Run scraper
    pages = await scrape_pypi_docs(
        packages=packages,
        scraper_config=pypi_config.get("scraper", {}),
        content_config=pypi_config.get("content", {})
    )

    # Save raw pages
    pypi_output_dir = output_dir / pypi_config.get("output_subdir", "pypi_docs")
    save_scraped_pages(pages, pypi_output_dir / "scraped_pages.json")

    # Create conversations
    conversations = create_documentation_conversations(pages)

    # Save intermediate conversations
    save_pypi_conversations(conversations, pypi_output_dir / "conversations.jsonl")

    logger.info(f"PyPI scraping complete: {len(pages)} pages, {len(conversations)} conversations")

    return conversations


async def run_github_pr_creator(config: dict, output_dir: Path) -> list[dict]:
    """Run the GitHub PR dataset creator."""
    pr_config = config.get("github_prs", {})

    if not pr_config.get("enabled", True):
        logger.info("GitHub PR dataset creation disabled in config")
        return []

    repositories = pr_config.get("repositories", [])
    if not repositories:
        logger.warning("No repositories configured for GitHub PR dataset")
        return []

    logger.info(f"Creating PR dataset from {len(repositories)} repositories...")

    # Get GitHub token
    github_token = pr_config.get("token") or os.environ.get("GITHUB_TOKEN")
    if not github_token:
        logger.warning(
            "No GitHub token provided. API rate limits will be restricted. "
            "Set GITHUB_TOKEN environment variable or add 'token' to config."
        )

    # Run PR dataset creator
    prs, conversations = await create_pr_dataset(
        repositories=repositories,
        filters=pr_config.get("filters", {}),
        api_config=pr_config.get("api", {}),
        github_token=github_token
    )

    # Save raw PRs
    pr_output_dir = output_dir / pr_config.get("output_subdir", "github_prs")
    save_prs_json(prs, pr_output_dir / "pull_requests.json")

    # Save intermediate conversations
    save_pr_conversations(conversations, pr_output_dir / "conversations.jsonl")

    logger.info(f"GitHub PR dataset complete: {len(prs)} PRs, {len(conversations)} conversations")

    return conversations


async def main():
    parser = argparse.ArgumentParser(
        description="Build Python-focused training dataset for nanochat LLM"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--pypi-only",
        action="store_true",
        help="Only run PyPI documentation scraping"
    )
    parser.add_argument(
        "--github-only",
        action="store_true",
        help="Only run GitHub PR dataset creation"
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Skip deduplication"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Load config
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        print("Create a config.yaml file or specify path with --config")
        sys.exit(1)

    config = load_config(args.config)

    # Override logging if verbose
    if args.verbose:
        config.setdefault("logging", {})["level"] = "DEBUG"

    setup_logging(config)
    logger.info(f"Loaded config from {args.config}")

    # Determine output directory
    output_config = config.get("output", {})
    output_dir = args.output or expand_path(
        output_config.get("base_dir", "~/.cache/nanochat/python_dataset")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Collect conversations from all sources
    all_conversations = []
    stats = {
        "pypi_docs": 0,
        "github_prs": 0,
    }

    # Run PyPI scraper
    if not args.github_only:
        try:
            pypi_conversations = await run_pypi_scraper(config, output_dir)
            all_conversations.extend(pypi_conversations)
            stats["pypi_docs"] = len(pypi_conversations)
        except Exception as e:
            logger.error(f"PyPI scraping failed: {e}")
            if args.pypi_only:
                raise

    # Run GitHub PR creator
    if not args.pypi_only:
        try:
            pr_conversations = await run_github_pr_creator(config, output_dir)
            all_conversations.extend(pr_conversations)
            stats["github_prs"] = len(pr_conversations)
        except Exception as e:
            logger.error(f"GitHub PR dataset creation failed: {e}")
            if args.github_only:
                raise

    if not all_conversations:
        logger.warning("No conversations generated!")
        return

    logger.info(f"Total conversations: {len(all_conversations)}")

    # Deduplication
    dataset_config = config.get("dataset", {})
    if dataset_config.get("deduplicate", True) and not args.no_dedupe:
        all_conversations = deduplicate_conversations(
            all_conversations,
            threshold=dataset_config.get("dedup_threshold", 0.95)
        )

    # Shuffle
    seed = dataset_config.get("random_seed", 42)
    random.seed(seed)
    random.shuffle(all_conversations)

    # Split train/test
    train_ratio = dataset_config.get("train_ratio", 0.9)
    train_conversations, test_conversations = split_train_test(
        all_conversations,
        train_ratio=train_ratio,
        seed=seed
    )

    # Update stats
    stats["total_before_dedupe"] = stats["pypi_docs"] + stats["github_prs"]
    stats["total_after_dedupe"] = len(all_conversations)
    stats["train_size"] = len(train_conversations)
    stats["test_size"] = len(test_conversations)

    # Save final datasets
    formats = output_config.get("formats", ["jsonl"])

    # Save train set
    train_dir = output_dir / "train"
    save_final_dataset(
        train_conversations,
        train_dir,
        formats,
        output_config.get("parquet_compression", "zstd"),
        output_config.get("parquet_compression_level", 3)
    )

    # Save test set
    test_dir = output_dir / "test"
    save_final_dataset(
        test_conversations,
        test_dir,
        formats,
        output_config.get("parquet_compression", "zstd"),
        output_config.get("parquet_compression_level", 3)
    )

    # Save combined (for convenience)
    combined_dir = output_dir / "combined"
    save_final_dataset(
        all_conversations,
        combined_dir,
        formats,
        output_config.get("parquet_compression", "zstd"),
        output_config.get("parquet_compression_level", 3)
    )

    # Save dataset info
    save_dataset_info(output_dir, stats, config)

    # Print summary
    print("\n" + "=" * 60)
    print("PYTHON DATASET BUILD COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nStatistics:")
    print(f"  PyPI docs conversations: {stats['pypi_docs']}")
    print(f"  GitHub PR conversations: {stats['github_prs']}")
    print(f"  Total (after dedupe):    {stats['total_after_dedupe']}")
    print(f"  Train set:               {stats['train_size']}")
    print(f"  Test set:                {stats['test_size']}")
    print(f"\nFiles created:")
    print(f"  {train_dir / 'conversations.jsonl'}")
    print(f"  {test_dir / 'conversations.jsonl'}")
    print(f"  {combined_dir / 'conversations.jsonl'}")
    print("\nTo use this dataset for training, update your task configuration")
    print("to point to the generated conversations.jsonl file.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
