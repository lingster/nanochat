"""File system scanner extractor."""

import os
from pathlib import Path
from typing import Iterator
import chardet

from .base import BaseExtractor, ExtractedContent
from ..config.schema import FileSourceOptions


class FileScanner(BaseExtractor):
    """Extractor for local file system."""

    # Text file extensions
    TEXT_EXTENSIONS = {
        '.txt', '.md', '.rst', '.tex', '.markdown',
        '.py', '.js', '.java', '.cpp', '.c', '.h', '.hpp',
        '.rs', '.go', '.rb', '.php', '.html', '.xml', '.json',
        '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
        '.sh', '.bash', '.zsh', '.fish',
        '.css', '.scss', '.sass', '.less',
        '.sql', '.r', '.m', '.swift', '.kt', '.scala',
    }

    def __init__(self, paths: list, options: FileSourceOptions, logger=None, state=None):
        """Initialize file scanner.

        Args:
            paths: List of file paths or glob patterns
            options: File scanning options
            logger: Optional logger
            state: Optional JobState for resume functionality
        """
        super().__init__(logger)
        self.paths = paths
        self.options = options
        self.state = state

    def extract(self) -> Iterator[ExtractedContent]:
        """Extract content from files.

        Yields:
            ExtractedContent objects
        """
        for path_pattern in self.paths:
            # Handle glob patterns
            if '*' in path_pattern:
                base_path = Path(path_pattern.split('*')[0])
                if not base_path.exists():
                    self.log('warning', f"Base path doesn't exist: {base_path}")
                    continue

                pattern = path_pattern[len(str(base_path)):]
                for file_path in base_path.glob(pattern.lstrip('/')):
                    if file_path.is_file():
                        content = self._read_file(file_path)
                        if content:
                            yield content
            else:
                # Single file or directory
                path = Path(path_pattern)
                if not path.exists():
                    self.log('warning', f"Path doesn't exist: {path}")
                    continue

                if path.is_file():
                    content = self._read_file(path)
                    if content:
                        yield content
                elif path.is_dir():
                    yield from self._scan_directory(path)

    def _scan_directory(self, directory: Path) -> Iterator[ExtractedContent]:
        """Recursively scan directory for text files.

        Args:
            directory: Directory to scan

        Yields:
            ExtractedContent objects
        """
        if self.options.recursive:
            file_iter = directory.rglob('*')
        else:
            file_iter = directory.glob('*')

        for file_path in file_iter:
            if file_path.is_file():
                content = self._read_file(file_path)
                if content:
                    yield content

    def _read_file(self, file_path: Path) -> ExtractedContent:
        """Read a single file.

        Args:
            file_path: Path to file

        Returns:
            ExtractedContent object or None if failed
        """
        # Check if already processed
        file_id = str(file_path.absolute())
        if self.state and self.state.is_processed('files', file_id):
            self.log('debug', f"File already processed, skipping: {file_path}")
            if self.state:
                self.state.increment_skipped()
            return None

        # Check if text file
        if not self._is_text_file(file_path):
            self.log('debug', f"Skipping non-text file: {file_path}")
            return None

        try:
            self.log('info', f"Reading: {file_path}")

            # Try to read with specified encoding
            try:
                with open(file_path, 'r', encoding=self.options.encoding) as f:
                    text = f.read()
            except UnicodeDecodeError:
                # Fallback: detect encoding
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                detected = chardet.detect(raw_data)
                encoding = detected['encoding'] or 'utf-8'
                text = raw_data.decode(encoding, errors='ignore')

            metadata = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_extension': file_path.suffix,
                'file_size': file_path.stat().st_size,
            }

            # Mark as processed in state
            if self.state:
                self.state.mark_processed('files', file_id)

            return ExtractedContent(
                text=text,
                metadata=metadata,
                source=str(file_path)
            )

        except Exception as e:
            self.log('error', f"Failed to read {file_path}: {e}")
            return None

    def _is_text_file(self, file_path: Path) -> bool:
        """Check if file is a text file.

        Args:
            file_path: Path to file

        Returns:
            True if text file
        """
        # Check extension
        if file_path.suffix.lower() in self.TEXT_EXTENSIONS:
            return True

        # Check if file is too large (skip files > 100MB)
        if file_path.stat().st_size > 100 * 1024 * 1024:
            return False

        # Try to detect if binary
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                # Check for null bytes (binary indicator)
                if b'\x00' in chunk:
                    return False
            return True
        except Exception:
            return False
