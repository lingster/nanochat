"""Caching utilities."""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Optional


class FileCache:
    """Simple file-based cache."""

    def __init__(self, cache_dir: str):
        """Initialize cache.

        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        # Hash the key to create a safe filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            # Cache corrupted, remove it
            cache_path.unlink(missing_ok=True)
            return None

    def set(self, key: str, value: Any):
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        cache_path = self._get_cache_path(key)
        with open(cache_path, 'wb') as f:
            pickle.dump(value, f)

    def has(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self._get_cache_path(key).exists()

    def clear(self):
        """Clear all cache files."""
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink()
