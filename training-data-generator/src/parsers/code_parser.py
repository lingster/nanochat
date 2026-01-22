"""Code and diff parser."""

import re
from typing import Dict, List
from .base import BaseParser


class CodeParser(BaseParser):
    """Parser for code and diffs."""

    def parse(self, content: str, metadata: dict = None) -> str:
        """Parse code content.

        Args:
            content: Code content
            metadata: Optional metadata

        Returns:
            Clean code
        """
        # Just return as-is for now
        return content.strip()


class DiffParser(BaseParser):
    """Parser for Git diffs."""

    def parse(self, patch: str, metadata: dict = None) -> Dict[str, any]:
        """Parse a Git diff patch.

        Args:
            patch: Git diff patch string
            metadata: Optional metadata

        Returns:
            Dictionary with parsed diff information
        """
        if not patch:
            return None

        lines = patch.split('\n')
        result = {
            'additions': [],
            'deletions': [],
            'context': [],
        }

        for line in lines:
            if line.startswith('+') and not line.startswith('+++'):
                result['additions'].append(line[1:])
            elif line.startswith('-') and not line.startswith('---'):
                result['deletions'].append(line[1:])
            elif line.startswith(' '):
                result['context'].append(line[1:])

        return result

    def format_diff_as_code(self, patch: str, filename: str = None) -> str:
        """Format diff patch as readable code changes.

        Args:
            patch: Git diff patch
            filename: Optional filename

        Returns:
            Formatted code changes
        """
        if not patch:
            return ""

        output = []
        if filename:
            output.append(f"File: {filename}\n")

        parsed = self.parse(patch)
        if not parsed:
            return ""

        # Show additions
        if parsed['additions']:
            output.append("Added:")
            for line in parsed['additions']:
                output.append(f"  {line}")

        # Show deletions
        if parsed['deletions']:
            if parsed['additions']:
                output.append("")
            output.append("Removed:")
            for line in parsed['deletions']:
                output.append(f"  {line}")

        return '\n'.join(output)


class CommitMessageParser(BaseParser):
    """Parser for extracting meaningful information from commit messages."""

    def parse(self, message: str, metadata: dict = None) -> Dict[str, str]:
        """Parse commit message into structured parts.

        Args:
            message: Commit message
            metadata: Optional metadata

        Returns:
            Dictionary with 'title' and 'body'
        """
        lines = message.strip().split('\n')

        title = lines[0].strip() if lines else ""
        body = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ""

        # Remove common prefixes
        prefixes = ['fix:', 'feat:', 'docs:', 'style:', 'refactor:', 'test:', 'chore:']
        for prefix in prefixes:
            if title.lower().startswith(prefix):
                title = title[len(prefix):].strip()
                break

        return {
            'title': title,
            'body': body,
            'full': message
        }
