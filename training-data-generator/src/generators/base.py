"""Base generator interface."""

from abc import ABC, abstractmethod
from typing import Any, List
from pathlib import Path


class BaseGenerator(ABC):
    """Base class for all output generators."""

    def __init__(self, output_path: str, options: Any = None, logger=None):
        """Initialize generator.

        Args:
            output_path: Path for output file(s)
            options: Generator-specific options
            logger: Optional logger instance
        """
        self.output_path = Path(output_path)
        self.options = options
        self.logger = logger

        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def add(self, item: Any):
        """Add an item to be written.

        Args:
            item: Item to add (format depends on generator type)
        """
        pass

    @abstractmethod
    def finalize(self) -> List[str]:
        """Finalize and write all data.

        Returns:
            List of output file paths
        """
        pass

    def log(self, level: str, message: str):
        """Log a message if logger is available."""
        if self.logger:
            getattr(self.logger, level)(message)
