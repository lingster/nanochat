"""Progress tracking utilities."""

from typing import Optional
from tqdm import tqdm


class ProgressTracker:
    """Wrapper around tqdm for consistent progress tracking."""

    def __init__(self, total: Optional[int] = None, desc: str = "", unit: str = "item"):
        """Initialize progress tracker.

        Args:
            total: Total number of items (None for unknown)
            desc: Description of the task
            unit: Unit name for items
        """
        self.pbar = tqdm(total=total, desc=desc, unit=unit)

    def update(self, n: int = 1):
        """Update progress by n items."""
        self.pbar.update(n)

    def set_description(self, desc: str):
        """Update the description."""
        self.pbar.set_description(desc)

    def set_postfix(self, **kwargs):
        """Set postfix information."""
        self.pbar.set_postfix(**kwargs)

    def close(self):
        """Close the progress bar."""
        self.pbar.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
