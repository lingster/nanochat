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

        # Split by headers
        sections = re.split(r'\n(#{1,3})\s+(.+)\n', content)

        i = 0
        while i < len(sections) - 2:
            if sections[i].strip():
                i += 1
                continue

            # sections[i+1] is the header marker (# or ## or ###)
            # sections[i+2] is the header text (question)
            # sections[i+3] is the content (answer)
            if i + 3 < len(sections):
                question = sections[i + 2].strip()
                answer = sections[i + 3].strip()

                if question and answer:
                    qa_pairs.append({
                        'question': question,
                        'answer': answer
                    })

            i += 3

        return qa_pairs
