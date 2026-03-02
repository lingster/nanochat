"""Markdown content parser."""

import re
from .base import BaseParser


class MarkdownParser(BaseParser):
    """Parser for Markdown content."""

    def parse(self, content: str, metadata: dict = None) -> str:
        """Parse Markdown content (minimal processing).

        Args:
            content: Markdown content
            metadata: Optional metadata

        Returns:
            Clean markdown text
        """
        # Just clean up excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        text = text.strip()

        return text


class MarkdownQAParser(BaseParser):
    """Parser for extracting Q&A from Markdown files."""

    def parse(self, content: str, metadata: dict = None) -> list:
        """Parse Markdown into Q&A pairs.

        Assumes format with headers as questions and content as answers.

        Args:
            content: Markdown content
            metadata: Optional metadata

        Returns:
            List of Q&A dictionaries
        """
        qa_pairs = []

        # Split by headers (##+ level headers, not top-level #)
        sections = re.split(r'\n(#{2,3})\s+(.+?)(?:\n|$)', content)

        # After split:
        # sections[0] = content before first header
        # sections[1] = header marker (## or ###)
        # sections[2] = header text (question)
        # sections[3] = content after header (answer + content before next header)
        # sections[4] = next header marker
        # etc.

        i = 1  # Start at first header marker
        while i + 2 < len(sections):
            header_marker = sections[i]
            question = sections[i + 1].strip()
            answer_section = sections[i + 2].strip() if i + 2 < len(sections) else ""

            if question and answer_section:
                qa_pairs.append({
                    'question': question,
                    'answer': answer_section
                })

            i += 3  # Move to next header marker

        return qa_pairs
