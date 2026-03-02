"""Text cleaning utilities."""

import re
from typing import Optional


class TextCleaner:
    """Clean and normalize text content."""

    @staticmethod
    def clean(text: str, options: dict = None) -> str:
        """Clean text content.

        Args:
            text: Raw text
            options: Cleaning options

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        options = options or {}

        # Normalize whitespace
        if options.get('normalize_whitespace', True):
            text = TextCleaner.normalize_whitespace(text)

        # Remove URLs
        if options.get('remove_urls', False):
            text = TextCleaner.remove_urls(text)

        # Remove emails
        if options.get('remove_emails', False):
            text = TextCleaner.remove_emails(text)

        # Remove special characters
        if options.get('remove_special_chars', False):
            text = TextCleaner.remove_special_characters(text)

        return text.strip()

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace in text.

        Args:
            text: Input text

        Returns:
            Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)

        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        # Remove trailing whitespace from lines
        text = '\n'.join(line.rstrip() for line in text.split('\n'))

        return text

    @staticmethod
    def remove_urls(text: str) -> str:
        """Remove URLs from text.

        Args:
            text: Input text

        Returns:
            Text without URLs
        """
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, '', text)

    @staticmethod
    def remove_emails(text: str) -> str:
        """Remove email addresses from text.

        Args:
            text: Input text

        Returns:
            Text without emails
        """
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)

    @staticmethod
    def remove_special_characters(text: str) -> str:
        """Remove special characters, keep alphanumeric and basic punctuation.

        Args:
            text: Input text

        Returns:
            Text with only alphanumeric and basic punctuation
        """
        # Keep letters, numbers, spaces, and basic punctuation
        return re.sub(r'[^a-zA-Z0-9\s.,!?;:()\-\'\"]', '', text)

    @staticmethod
    def remove_excessive_punctuation(text: str) -> str:
        """Remove excessive punctuation.

        Args:
            text: Input text

        Returns:
            Text with normalized punctuation
        """
        # Replace multiple punctuation with single
        text = re.sub(r'([.!?])\1+', r'\1', text)
        return text
