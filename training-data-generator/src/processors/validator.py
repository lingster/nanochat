"""Content validation."""

import re
from typing import Optional


class ContentValidator:
    """Validate content quality."""

    @staticmethod
    def is_valid(text: str, min_length: int = 50, max_length: int = 100000) -> bool:
        """Check if text is valid.

        Args:
            text: Text to validate
            min_length: Minimum length
            max_length: Maximum length

        Returns:
            True if valid
        """
        if not text or not isinstance(text, str):
            return False

        # Check length
        if len(text) < min_length or len(text) > max_length:
            return False

        # Check if mostly printable
        if not ContentValidator.is_printable(text):
            return False

        return True

    @staticmethod
    def is_printable(text: str, threshold: float = 0.95) -> bool:
        """Check if text is mostly printable characters.

        Args:
            text: Text to check
            threshold: Minimum ratio of printable characters

        Returns:
            True if mostly printable
        """
        if not text:
            return False

        printable_count = sum(1 for c in text if c.isprintable() or c in '\n\t')
        ratio = printable_count / len(text)

        return ratio >= threshold

    @staticmethod
    def has_good_structure(text: str) -> bool:
        """Check if text has good structure (sentences, paragraphs).

        Args:
            text: Text to check

        Returns:
            True if well-structured
        """
        # Check for sentence-like structure (has periods, question marks, etc.)
        sentence_endings = len(re.findall(r'[.!?]\s', text))

        # Should have some sentence breaks
        if sentence_endings == 0 and len(text) > 100:
            return False

        # Check for reasonable paragraph breaks
        paragraphs = text.split('\n\n')
        if len(paragraphs) == 1 and len(text) > 1000:
            # Very long text with no paragraph breaks is suspicious
            return False

        return True

    @staticmethod
    def matches_pattern(text: str, exclude_patterns: list) -> bool:
        """Check if text matches any exclude patterns.

        Args:
            text: Text to check
            exclude_patterns: List of regex patterns to exclude

        Returns:
            True if text should be excluded
        """
        if not exclude_patterns:
            return False

        text_lower = text.lower()
        for pattern in exclude_patterns:
            if re.search(pattern, text_lower):
                return True

        return False


class ConversationValidator:
    """Validate conversation format."""

    @staticmethod
    def is_valid_conversation(conversation: dict) -> bool:
        """Validate conversation structure.

        Args:
            conversation: Conversation dictionary

        Returns:
            True if valid
        """
        if not conversation or not isinstance(conversation, dict):
            return False

        messages = conversation.get('messages', [])
        if not messages or not isinstance(messages, list):
            return False

        # Must have at least 2 messages
        if len(messages) < 2:
            return False

        # Check message structure
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                return False

            # Must have role and content
            if 'role' not in msg or 'content' not in msg:
                return False

            # Role must be user or assistant
            if msg['role'] not in ['user', 'assistant', 'system']:
                return False

            # Content must be non-empty
            if not msg['content']:
                return False

        # First message should be user (unless there's a system message)
        if messages[0]['role'] not in ['user', 'system']:
            return False

        # Check role alternation (skip system messages)
        non_system_msgs = [m for m in messages if m['role'] != 'system']
        for i in range(len(non_system_msgs) - 1):
            if non_system_msgs[i]['role'] == non_system_msgs[i + 1]['role']:
                # Allow consecutive messages of same role
                # (some datasets have multi-turn assistant responses)
                pass

        return True
