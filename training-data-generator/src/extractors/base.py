"""Base extractor interface."""

from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any
from dataclasses import dataclass


@dataclass
class ExtractedContent:
    """Container for extracted content."""
    text: str
    metadata: Dict[str, Any]
    source: str


class BaseExtractor(ABC):
    """Base class for all data extractors."""

    def __init__(self, logger=None):
        """Initialize extractor.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger

    @abstractmethod
    def extract(self) -> Iterator[ExtractedContent]:
        """Extract content from source.

        Yields:
            ExtractedContent objects
        """
        pass

    def log(self, level: str, message: str):
        """Log a message if logger is available."""
        if self.logger:
            getattr(self.logger, level)(message)
