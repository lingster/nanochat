"""Content deduplication."""

import xxhash
from typing import Set


class Deduplicator:
    """Detect and remove duplicate content."""

    def __init__(self):
        """Initialize deduplicator."""
        self.seen_hashes: Set[str] = set()

    def is_duplicate(self, text: str) -> bool:
        """Check if content is duplicate.

        Args:
            text: Content to check

        Returns:
            True if duplicate
        """
        content_hash = self._hash_content(text)

        if content_hash in self.seen_hashes:
            return True

        self.seen_hashes.add(content_hash)
        return False

    def _hash_content(self, text: str) -> str:
        """Generate hash of content.

        Args:
            text: Content to hash

        Returns:
            Hash string
        """
        # Use xxhash for fast hashing
        return xxhash.xxh64(text.encode('utf-8')).hexdigest()

    def reset(self):
        """Reset deduplicator state."""
        self.seen_hashes.clear()

    def get_stats(self) -> dict:
        """Get deduplication statistics.

        Returns:
            Dictionary with stats
        """
        return {
            'unique_items': len(self.seen_hashes)
        }
