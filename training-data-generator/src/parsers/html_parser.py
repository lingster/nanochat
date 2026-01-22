"""HTML content parser."""

import re
from bs4 import BeautifulSoup
import html2text

from .base import BaseParser


class HTMLParser(BaseParser):
    """Parser for HTML content."""

    def __init__(self):
        """Initialize HTML parser."""
        self.html2text = html2text.HTML2Text()
        self.html2text.ignore_links = False
        self.html2text.ignore_images = True
        self.html2text.ignore_emphasis = False

    def parse(self, content: str, metadata: dict = None) -> str:
        """Parse HTML to clean text.

        Args:
            content: HTML content
            metadata: Optional metadata

        Returns:
            Clean text content
        """
        # Parse with BeautifulSoup
        soup = BeautifulSoup(content, 'lxml')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()

        # Convert to markdown-style text
        text = self.html2text.handle(str(soup))

        # Clean up excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = text.strip()

        return text
