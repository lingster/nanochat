"""Base parser interface."""

from abc import ABC, abstractmethod
from typing import Any


class BaseParser(ABC):
    """Base class for all content parsers."""

    @abstractmethod
    def parse(self, content: str, metadata: dict = None) -> Any:
        """Parse content into structured format.

        Args:
            content: Raw content to parse
            metadata: Optional metadata about the content

        Returns:
            Parsed content in appropriate format
        """
        pass
