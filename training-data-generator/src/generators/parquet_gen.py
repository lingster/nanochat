"""Parquet file generator."""

import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import List

from .base import BaseGenerator
from ..config.schema import ParquetOutputOptions


class ParquetGenerator(BaseGenerator):
    """Generator for Parquet files."""

    def __init__(self, output_path: str, options: ParquetOutputOptions = None, logger=None):
        """Initialize Parquet generator.

        Args:
            output_path: Path template for output files (can include {index})
            options: Parquet generation options
            logger: Optional logger
        """
        super().__init__(output_path, options, logger)
        self.options = options or ParquetOutputOptions()

        # Buffer for accumulating data
        self.buffer = []
        self.current_size_mb = 0
        self.shard_index = 0
        self.output_files = []

    def add(self, item: dict):
        """Add a text item to be written.

        Args:
            item: Dictionary with 'text' key
        """
        text = item.get('text', '')

        if not text:
            return

        # Validate length
        if len(text) < self.options.min_text_length:
            self.log('debug', f"Text too short ({len(text)} chars), skipping")
            return

        if len(text) > self.options.max_text_length:
            self.log('debug', f"Text too long ({len(text)} chars), truncating")
            text = text[:self.options.max_text_length]

        # Add to buffer
        self.buffer.append({'text': text})

        # Estimate size (rough approximation)
        self.current_size_mb += len(text) / (1024 * 1024)

        # Check if we should write a shard
        if self.current_size_mb >= self.options.shard_size_mb:
            self._write_shard()

    def finalize(self) -> List[str]:
        """Write remaining data and return output file paths.

        Returns:
            List of output file paths
        """
        # Write any remaining data
        if self.buffer:
            self._write_shard()

        self.log('info', f"Generated {len(self.output_files)} Parquet shard(s)")
        return self.output_files

    def _write_shard(self):
        """Write current buffer to a Parquet shard."""
        if not self.buffer:
            return

        # Generate output filename
        if '{index' in str(self.output_path):
            output_file = str(self.output_path).format(index=self.shard_index)
        else:
            # Append shard number if template doesn't have it
            stem = self.output_path.stem
            suffix = self.output_path.suffix
            output_file = self.output_path.parent / f"{stem}_shard_{self.shard_index:05d}{suffix}"

        output_file = Path(output_file)

        self.log('info', f"Writing shard {self.shard_index} with {len(self.buffer)} items to {output_file}")

        # Create schema
        schema = pa.schema([
            ('text', pa.string())
        ])

        # Convert buffer to Arrow table
        table = pa.Table.from_pylist(self.buffer, schema=schema)

        # Write Parquet file
        pq.write_table(
            table,
            output_file,
            compression=self.options.compression,
            row_group_size=self.options.row_group_size
        )

        self.output_files.append(str(output_file))
        self.shard_index += 1

        # Clear buffer
        self.buffer = []
        self.current_size_mb = 0
