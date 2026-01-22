"""State management for job resumption."""

import json
import hashlib
from pathlib import Path
from typing import Set, Dict, Any
from datetime import datetime


class JobState:
    """Manages state for job resumption."""

    def __init__(self, job_name: str, cache_dir: str):
        """Initialize job state.

        Args:
            job_name: Name of the job
            cache_dir: Directory for cache files
        """
        self.job_name = job_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # State file path
        safe_name = self._sanitize_filename(job_name)
        self.state_file = self.cache_dir / f"{safe_name}_state.json"

        # State data
        self.state = self._load_state()

    def _sanitize_filename(self, name: str) -> str:
        """Create safe filename from job name."""
        # Use hash to ensure filename safety
        return hashlib.md5(name.encode()).hexdigest()

    def _load_state(self) -> Dict[str, Any]:
        """Load state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception:
                # Corrupted state file, start fresh
                return self._create_new_state()
        return self._create_new_state()

    def _create_new_state(self) -> Dict[str, Any]:
        """Create new state structure."""
        return {
            'job_name': self.job_name,
            'started_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'processed_items': {
                'urls': [],
                'files': [],
                'github_prs': []
            },
            'generator_state': {},
            'statistics': {
                'total_items_processed': 0,
                'total_items_skipped': 0,
                'total_duplicates': 0
            }
        }

    def save(self):
        """Save state to file."""
        self.state['last_updated'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def is_processed(self, item_type: str, identifier: str) -> bool:
        """Check if an item has been processed.

        Args:
            item_type: Type of item ('urls', 'files', 'github_prs')
            identifier: Unique identifier for the item

        Returns:
            True if already processed
        """
        if item_type not in self.state['processed_items']:
            return False
        return identifier in self.state['processed_items'][item_type]

    def mark_processed(self, item_type: str, identifier: str):
        """Mark an item as processed.

        Args:
            item_type: Type of item ('urls', 'files', 'github_prs')
            identifier: Unique identifier for the item
        """
        if item_type not in self.state['processed_items']:
            self.state['processed_items'][item_type] = []

        if identifier not in self.state['processed_items'][item_type]:
            self.state['processed_items'][item_type].append(identifier)
            self.state['statistics']['total_items_processed'] += 1

    def increment_skipped(self):
        """Increment skipped items counter."""
        self.state['statistics']['total_items_skipped'] += 1

    def increment_duplicates(self):
        """Increment duplicates counter."""
        self.state['statistics']['total_duplicates'] += 1

    def get_generator_state(self, key: str, default=None):
        """Get generator state value.

        Args:
            key: State key
            default: Default value if not found

        Returns:
            State value
        """
        return self.state['generator_state'].get(key, default)

    def set_generator_state(self, key: str, value):
        """Set generator state value.

        Args:
            key: State key
            value: State value
        """
        self.state['generator_state'][key] = value

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics.

        Returns:
            Statistics dictionary
        """
        stats = self.state['statistics'].copy()
        stats['processed_urls'] = len(self.state['processed_items'].get('urls', []))
        stats['processed_files'] = len(self.state['processed_items'].get('files', []))
        stats['processed_prs'] = len(self.state['processed_items'].get('github_prs', []))
        return stats

    def clear(self):
        """Clear all state (start fresh)."""
        self.state = self._create_new_state()
        self.save()

    def delete(self):
        """Delete state file."""
        if self.state_file.exists():
            self.state_file.unlink()


class StateManager:
    """Manages states for multiple jobs."""

    def __init__(self, cache_dir: str):
        """Initialize state manager.

        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, JobState] = {}

    def get_job_state(self, job_name: str, resume: bool = True) -> JobState:
        """Get or create job state.

        Args:
            job_name: Name of the job
            resume: Whether to resume from existing state

        Returns:
            JobState instance
        """
        if job_name not in self.jobs:
            self.jobs[job_name] = JobState(job_name, str(self.cache_dir))

            if not resume:
                # Clear existing state if not resuming
                self.jobs[job_name].clear()

        return self.jobs[job_name]

    def save_all(self):
        """Save all job states."""
        for job_state in self.jobs.values():
            job_state.save()

    def clear_all(self):
        """Clear all job states."""
        for job_state in self.jobs.values():
            job_state.clear()
