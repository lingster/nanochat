"""Tests for data processors."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from processors.cleaner import TextCleaner
from processors.validator import ContentValidator, ConversationValidator
from processors.deduplicator import Deduplicator


def test_normalize_whitespace():
    """Test whitespace normalization."""
    text = "Hello    world\n\n\n\nNext paragraph"
    result = TextCleaner.normalize_whitespace(text)

    assert "    " not in result
    assert "\n\n\n" not in result
    assert "Hello world" in result


def test_remove_urls():
    """Test URL removal."""
    text = "Check out https://example.com for more info"
    result = TextCleaner.remove_urls(text)

    assert "https://example.com" not in result
    assert "Check out" in result


def test_is_valid_text():
    """Test text validation."""
    # Valid text
    assert ContentValidator.is_valid("This is a valid text with enough content.", min_length=10)

    # Too short
    assert not ContentValidator.is_valid("Short", min_length=10)

    # Too long
    assert not ContentValidator.is_valid("A" * 1000, max_length=100)

    # Empty
    assert not ContentValidator.is_valid("")


def test_is_valid_conversation():
    """Test conversation validation."""
    # Valid conversation
    valid_conv = {
        'messages': [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there!'}
        ]
    }
    assert ConversationValidator.is_valid_conversation(valid_conv)

    # Invalid: missing role
    invalid_conv = {
        'messages': [
            {'content': 'Hello'}
        ]
    }
    assert not ConversationValidator.is_valid_conversation(invalid_conv)

    # Invalid: empty messages
    invalid_conv2 = {
        'messages': []
    }
    assert not ConversationValidator.is_valid_conversation(invalid_conv2)


def test_deduplicator():
    """Test deduplication."""
    dedup = Deduplicator()

    text1 = "This is some text"
    text2 = "This is different text"
    text3 = "This is some text"  # Duplicate of text1

    assert not dedup.is_duplicate(text1)
    assert not dedup.is_duplicate(text2)
    assert dedup.is_duplicate(text3)  # Should be detected as duplicate

    stats = dedup.get_stats()
    assert stats['unique_items'] == 2


def test_deduplicator_reset():
    """Test deduplicator reset."""
    dedup = Deduplicator()

    dedup.is_duplicate("text1")
    dedup.is_duplicate("text2")

    assert dedup.get_stats()['unique_items'] == 2

    dedup.reset()

    assert dedup.get_stats()['unique_items'] == 0
    assert not dedup.is_duplicate("text1")  # Should not be duplicate after reset


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
