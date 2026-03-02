#!/usr/bin/env python
"""Utility to convert a folder of plain-text files into nanochat pretraining shards."""

import argparse
import os
import re
from pathlib import Path
from typing import Iterable, List

import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.common import get_base_dir


def discover_text_files(root: Path, recursive: bool = True) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for path in root.glob(pattern):
        if path.is_file() and path.suffix.lower() in {".txt", ""}:
            yield path


def detect_start_index(output_dir: Path) -> int:
    regex = re.compile(r"shard_(\d{5})")
    max_index = -1
    for path in output_dir.glob("shard_*.parquet"):
        match = regex.match(path.stem)
        if match:
            max_index = max(max_index, int(match.group(1)))
    # also handle shards already marked as val
    for path in output_dir.glob("shard_*_val.parquet"):
        match = regex.match(path.stem.split("_val")[0])
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def write_parquet_shard(texts: List[str], path: Path, row_group_size: int, compression_level: int) -> None:
    table = pa.Table.from_arrays([pa.array(texts, type=pa.string())], names=["text"])
    pq.write_table(
        table,
        path,
        row_group_size=row_group_size,
        compression="zstd",
        compression_level=compression_level,
        use_dictionary=False,
        write_statistics=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert text files into nanochat-style Parquet shards.")
    parser.add_argument("input_dir", type=Path, help="Directory containing raw text files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write shards (defaults to ~/.cache/nanochat/base_data).",
    )
    parser.add_argument(
        "--target-chars",
        type=int,
        default=250_000_000,
        help="Approximate number of characters per shard (default: 250M).",
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=1024,
        help="Parquet row group size (default: 1024).",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=3,
        help="Zstandard compression level (default: 3).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Numeric index for the first shard. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive traversal of input_dir.",
    )
    parser.add_argument(
        "--reserve-val-shard",
        action="store_true",
        help="Rename the final shard to *_val.parquet so nanochat treats it as validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir = args.output_dir
    if output_dir is None:
        base_dir = Path(get_base_dir())
        output_dir = base_dir / "base_data"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    start_index = args.start_index if args.start_index is not None else detect_start_index(output_dir)

    text_files = sorted(discover_text_files(input_dir, recursive=not args.no_recursive))
    if not text_files:
        raise RuntimeError(f"No text files found under {input_dir}")

    current_texts: List[str] = []
    current_chars = 0
    shard_counter = start_index
    written_paths: List[Path] = []
    total_chars = 0

    for file_path in text_files:
        with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
            text = handle.read()
        if not text:
            continue
        current_texts.append(text)
        text_len = len(text)
        current_chars += text_len
        total_chars += text_len

        if current_chars >= args.target_chars and len(current_texts) >= args.row_group_size:
            shard_path = output_dir / f"shard_{shard_counter:05d}.parquet"
            write_parquet_shard(current_texts, shard_path, args.row_group_size, args.compression_level)
            print(f"Wrote {shard_path} | docs: {len(current_texts)} | chars: {current_chars}")
            written_paths.append(shard_path)
            current_texts = []
            current_chars = 0
            shard_counter += 1

    # Flush remainder (even if smaller than target)
    if current_texts:
        shard_path = output_dir / f"shard_{shard_counter:05d}.parquet"
        write_parquet_shard(current_texts, shard_path, args.row_group_size, args.compression_level)
        print(f"Wrote {shard_path} | docs: {len(current_texts)} | chars: {current_chars}")
        written_paths.append(shard_path)
        shard_counter += 1

    if args.reserve_val_shard and written_paths:
        last_path = written_paths.pop()
        val_path = last_path.with_name(f"{last_path.stem}_val.parquet")
        last_path.rename(val_path)
        print(f"Reserved validation shard: {val_path}")
        written_paths.append(val_path)

    print("---")
    print(f"Input files processed: {len(text_files)}")
    print(f"Total characters: {total_chars}")
    print(f"Shards written: {len(written_paths)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
