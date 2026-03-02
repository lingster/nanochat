"""Web scraping extractor."""

import re
from typing import Iterator, Set
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

from .base import BaseExtractor, ExtractedContent
from ..config.schema import WebSourceOptions
from ..utils.retry import retry_with_backoff, RateLimiter


class WebScraper(BaseExtractor):
    """Extractor for web content."""

    def __init__(self, urls: list, options: WebSourceOptions, logger=None, state=None):
        """Initialize web scraper.

        Args:
            urls: List of URLs to scrape
            options: Web scraping options
            logger: Optional logger
            state: Optional JobState for resume functionality
        """
        super().__init__(logger)
        self.urls = urls
        self.options = options
        self.rate_limiter = RateLimiter(options.rate_limit)
        self.visited: Set[str] = set()
        self.state = state
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TrainingDataGenerator/1.0 (Educational Purpose)'
        })

    def extract(self) -> Iterator[ExtractedContent]:
        """Extract content from web URLs.

        Yields:
            ExtractedContent objects
        """
        pages_processed = 0

        for url in self.urls:
            if pages_processed >= self.options.max_pages:
                self.log('info', f"Reached max pages limit: {self.options.max_pages}")
                break

            # Process URL and its linked pages
            for content in self._crawl_url(url, depth=0):
                yield content
                pages_processed += 1
                if pages_processed >= self.options.max_pages:
                    break

    def _crawl_url(self, url: str, depth: int) -> Iterator[ExtractedContent]:
        """Crawl a URL and optionally follow links.

        Args:
            url: URL to crawl
            depth: Current depth in the crawl tree

        Yields:
            ExtractedContent objects
        """
        # Check if already visited
        if url in self.visited:
            return

        # Check if already processed (from previous run)
        if self.state and self.state.is_processed('urls', url):
            self.log('debug', f"URL already processed, skipping: {url}")
            self.visited.add(url)  # Mark as visited to avoid reprocessing
            if self.state:
                self.state.increment_skipped()
            return

        # Check depth limit
        if depth > self.options.max_depth:
            return

        # Check URL pattern
        if self.options.url_pattern:
            if not re.match(self.options.url_pattern, url):
                self.log('debug', f"URL doesn't match pattern, skipping: {url}")
                return

        # Check exclude pattern
        if self.options.exclude_pattern:
            if re.search(self.options.exclude_pattern, url):
                self.log('debug', f"URL matches exclude pattern, skipping: {url}")
                return

        # Check same domain
        if self.options.same_domain_only and depth > 0:
            base_domain = urlparse(self.urls[0]).netloc
            url_domain = urlparse(url).netloc
            if base_domain != url_domain:
                self.log('debug', f"URL is different domain, skipping: {url}")
                return

        # Mark as visited
        self.visited.add(url)

        # Rate limiting
        self.rate_limiter.wait()

        # Fetch and parse
        try:
            content = self._fetch_page(url)
            if content:
                # Mark as processed in state
                if self.state:
                    self.state.mark_processed('urls', url)

                yield content

                # Follow links if enabled
                if self.options.follow_links and depth < self.options.max_depth:
                    links = self._extract_links(content.text, url)
                    for link in links:
                        yield from self._crawl_url(link, depth + 1)

        except Exception as e:
            self.log('error', f"Failed to process {url}: {e}")

    @retry_with_backoff(max_retries=3, exceptions=(requests.RequestException,))
    def _fetch_page(self, url: str) -> ExtractedContent:
        """Fetch and parse a web page.

        Args:
            url: URL to fetch

        Returns:
            ExtractedContent object
        """
        self.log('info', f"Fetching: {url}")

        response = self.session.get(url, timeout=30)
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.content, 'lxml')

        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()

        # Extract text
        text = soup.get_text(separator='\n', strip=True)

        # Clean up whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)

        metadata = {
            'url': url,
            'title': soup.title.string if soup.title else '',
            'content_type': response.headers.get('content-type', ''),
        }

        return ExtractedContent(
            text=text,
            metadata=metadata,
            source=url
        )

    def _extract_links(self, html: str, base_url: str) -> list:
        """Extract links from HTML.

        Args:
            html: HTML content
            base_url: Base URL for resolving relative links

        Returns:
            List of absolute URLs
        """
        soup = BeautifulSoup(html, 'lxml')
        links = []

        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            # Skip anchors and javascript
            if href.startswith('#') or href.startswith('javascript:'):
                continue

            # Convert to absolute URL
            absolute_url = urljoin(base_url, href)

            # Only include http(s) URLs
            if absolute_url.startswith(('http://', 'https://')):
                links.append(absolute_url)

        return links
