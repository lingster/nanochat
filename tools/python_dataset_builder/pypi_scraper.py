"""
PyPI Documentation Scraper

Scrapes documentation from popular Python packages and converts them into
training data format for the nanochat LLM.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup, NavigableString
from markdownify import markdownify as md

logger = logging.getLogger(__name__)


@dataclass
class ScrapedPage:
    """Represents a scraped documentation page."""
    url: str
    title: str
    content: str
    package_name: str
    section: str
    depth: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "package_name": self.package_name,
            "section": self.section,
            "depth": self.depth,
            "timestamp": self.timestamp,
        }

    @property
    def content_hash(self) -> str:
        return hashlib.md5(self.content.encode()).hexdigest()


@dataclass
class PackageConfig:
    """Configuration for a package to scrape."""
    name: str
    docs_url: str
    doc_type: str = "sphinx"


@dataclass
class ScraperConfig:
    """Configuration for the scraper."""
    max_pages_per_package: int = 500
    request_delay: float = 0.5
    request_timeout: int = 30
    max_retries: int = 3
    user_agent: str = "NanochatDatasetBuilder/1.0"
    concurrent_requests: int = 3
    max_depth: int = 5
    min_content_length: int = 100
    max_content_length: int = 50000
    exclude_patterns: list[str] = field(default_factory=list)
    include_patterns: list[str] = field(default_factory=list)


class DocumentationScraper:
    """
    Asynchronous documentation scraper for Python packages.

    Supports multiple documentation formats:
    - Sphinx (ReadTheDocs style)
    - MkDocs
    - Custom documentation sites
    """

    def __init__(self, config: ScraperConfig):
        self.config = config
        self.visited_urls: set[str] = set()
        self.pages: list[ScrapedPage] = []
        self.session: Optional[aiohttp.ClientSession] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        headers = {"User-Agent": self.config.user_agent}
        self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        self._semaphore = asyncio.Semaphore(self.config.concurrent_requests)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _normalize_url(self, url: str) -> str:
        """Normalize URL by removing fragments and trailing slashes."""
        parsed = urlparse(url)
        # Remove fragment
        normalized = parsed._replace(fragment="")
        url = normalized.geturl()
        # Remove trailing slash for consistency
        return url.rstrip("/")

    def _is_same_domain(self, url: str, base_url: str) -> bool:
        """Check if URL belongs to the same domain as base."""
        url_domain = urlparse(url).netloc
        base_domain = urlparse(base_url).netloc
        return url_domain == base_domain

    def _should_exclude(self, url: str) -> bool:
        """Check if URL matches any exclusion pattern."""
        from fnmatch import fnmatch
        for pattern in self.config.exclude_patterns:
            if fnmatch(url, pattern):
                return True
        return False

    def _should_include(self, url: str) -> bool:
        """Check if URL matches inclusion patterns (if any defined)."""
        if not self.config.include_patterns:
            return True
        from fnmatch import fnmatch
        for pattern in self.config.include_patterns:
            if fnmatch(url, pattern):
                return True
        return False

    def _extract_content_sphinx(self, soup: BeautifulSoup) -> tuple[str, str]:
        """Extract content from Sphinx-style documentation."""
        title = ""
        content = ""

        # Find title
        title_elem = soup.find("h1")
        if title_elem:
            title = title_elem.get_text(strip=True)
            # Remove the permalink symbol
            title = re.sub(r"¶$", "", title).strip()

        # Find main content area
        main_content = (
            soup.find("div", class_="body") or
            soup.find("div", class_="document") or
            soup.find("div", role="main") or
            soup.find("main") or
            soup.find("article")
        )

        if main_content:
            # Remove navigation elements
            for elem in main_content.find_all(["nav", "script", "style"]):
                elem.decompose()

            # Remove sidebar elements
            for elem in main_content.find_all(class_=re.compile(r"sidebar|toc|navigation")):
                elem.decompose()

            # Convert to markdown
            content = md(str(main_content), heading_style="ATX", code_language="python")

        return title, content

    def _extract_content_mkdocs(self, soup: BeautifulSoup) -> tuple[str, str]:
        """Extract content from MkDocs-style documentation."""
        title = ""
        content = ""

        # Find title
        title_elem = soup.find("h1")
        if title_elem:
            title = title_elem.get_text(strip=True)

        # Find main content
        main_content = (
            soup.find("article", class_="md-content__inner") or
            soup.find("div", class_="md-content") or
            soup.find("main") or
            soup.find("article")
        )

        if main_content:
            # Remove navigation and meta elements
            for elem in main_content.find_all(["nav", "script", "style", "footer"]):
                elem.decompose()

            # Convert to markdown
            content = md(str(main_content), heading_style="ATX", code_language="python")

        return title, content

    def _extract_content_generic(self, soup: BeautifulSoup) -> tuple[str, str]:
        """Extract content from generic documentation pages."""
        title = ""
        content = ""

        # Try multiple title sources
        title_elem = (
            soup.find("h1") or
            soup.find("title")
        )
        if title_elem:
            title = title_elem.get_text(strip=True)

        # Find main content using common patterns
        main_content = (
            soup.find("main") or
            soup.find("article") or
            soup.find("div", class_=re.compile(r"content|main|body")) or
            soup.find("div", id=re.compile(r"content|main|body"))
        )

        if not main_content:
            # Fallback to body
            main_content = soup.find("body")

        if main_content:
            # Remove unwanted elements
            for elem in main_content.find_all([
                "nav", "script", "style", "footer", "header",
                "aside", "form", "iframe"
            ]):
                elem.decompose()

            # Remove elements with navigation-like classes
            for elem in main_content.find_all(class_=re.compile(
                r"nav|sidebar|menu|footer|header|toc|breadcrumb"
            )):
                elem.decompose()

            # Convert to markdown
            content = md(str(main_content), heading_style="ATX", code_language="python")

        return title, content

    def _clean_content(self, content: str) -> str:
        """Clean and normalize extracted content."""
        # Remove excessive whitespace
        content = re.sub(r"\n{3,}", "\n\n", content)
        content = re.sub(r" {2,}", " ", content)

        # Remove common artifacts
        content = re.sub(r"¶", "", content)
        content = re.sub(r"\[source\]", "", content)
        content = re.sub(r"\[edit\]", "", content)

        # Clean up code blocks
        content = re.sub(r"```\s*\n\s*```", "", content)

        # Trim
        content = content.strip()

        return content

    def _extract_section(self, url: str, base_url: str) -> str:
        """Extract section name from URL path."""
        base_path = urlparse(base_url).path
        url_path = urlparse(url).path

        # Remove base path
        relative_path = url_path
        if url_path.startswith(base_path):
            relative_path = url_path[len(base_path):]

        # Clean up
        relative_path = relative_path.strip("/")

        # Get first path component as section
        parts = relative_path.split("/")
        if parts and parts[0]:
            return parts[0]

        return "root"

    def _extract_links(self, soup: BeautifulSoup, base_url: str, current_url: str) -> list[str]:
        """Extract valid documentation links from page."""
        links = []

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]

            # Skip anchors, javascript, mailto
            if href.startswith(("#", "javascript:", "mailto:")):
                continue

            # Resolve relative URLs
            full_url = urljoin(current_url, href)
            normalized = self._normalize_url(full_url)

            # Check if valid
            if not self._is_same_domain(normalized, base_url):
                continue

            if self._should_exclude(normalized):
                continue

            if not self._should_include(normalized):
                continue

            # Skip already visited
            if normalized in self.visited_urls:
                continue

            # Only HTML pages
            parsed = urlparse(normalized)
            path = parsed.path.lower()
            if path.endswith((".pdf", ".zip", ".tar", ".gz", ".whl", ".jpg", ".png", ".gif", ".svg")):
                continue

            links.append(normalized)

        return list(set(links))

    async def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch page content with retries."""
        for attempt in range(self.config.max_retries):
            try:
                async with self._semaphore:
                    async with self.session.get(url, allow_redirects=True) as response:
                        if response.status == 200:
                            return await response.text()
                        elif response.status == 404:
                            logger.debug(f"Page not found: {url}")
                            return None
                        else:
                            logger.warning(f"HTTP {response.status} for {url}")
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching {url} (attempt {attempt + 1})")
            except aiohttp.ClientError as e:
                logger.warning(f"Error fetching {url}: {e} (attempt {attempt + 1})")

            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(self.config.request_delay * (attempt + 1))

        return None

    async def _scrape_page(
        self,
        url: str,
        package: PackageConfig,
        depth: int
    ) -> list[str]:
        """Scrape a single page and return discovered links."""
        if url in self.visited_urls:
            return []

        if depth > self.config.max_depth:
            return []

        if (self.config.max_pages_per_package > 0 and
            len([p for p in self.pages if p.package_name == package.name]) >= self.config.max_pages_per_package):
            return []

        self.visited_urls.add(url)

        # Rate limiting
        await asyncio.sleep(self.config.request_delay)

        html = await self._fetch_page(url)
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")

        # Extract content based on doc type
        if package.doc_type == "sphinx":
            title, content = self._extract_content_sphinx(soup)
        elif package.doc_type == "mkdocs":
            title, content = self._extract_content_mkdocs(soup)
        else:
            title, content = self._extract_content_generic(soup)

        content = self._clean_content(content)

        # Validate content
        if len(content) < self.config.min_content_length:
            logger.debug(f"Content too short for {url}: {len(content)} chars")
            return self._extract_links(soup, package.docs_url, url)

        if len(content) > self.config.max_content_length:
            content = content[:self.config.max_content_length]
            logger.debug(f"Truncated content for {url}")

        # Create page object
        section = self._extract_section(url, package.docs_url)
        page = ScrapedPage(
            url=url,
            title=title or "Untitled",
            content=content,
            package_name=package.name,
            section=section,
            depth=depth
        )

        self.pages.append(page)
        logger.info(f"Scraped: {url} ({len(content)} chars)")

        # Extract links for further crawling
        return self._extract_links(soup, package.docs_url, url)

    async def scrape_package(self, package: PackageConfig) -> list[ScrapedPage]:
        """Scrape all documentation for a package."""
        logger.info(f"Starting scrape of {package.name} from {package.docs_url}")

        # Reset state for this package
        package_pages_before = len(self.pages)

        # BFS crawl
        queue = [(package.docs_url, 0)]  # (url, depth)
        queued_urls = {package.docs_url}

        while queue:
            url, depth = queue.pop(0)

            new_links = await self._scrape_page(url, package, depth)

            # Add new links to queue
            for link in new_links:
                if link not in queued_urls:
                    queued_urls.add(link)
                    queue.append((link, depth + 1))

        package_pages = self.pages[package_pages_before:]
        logger.info(f"Completed {package.name}: {len(package_pages)} pages scraped")

        return package_pages

    async def scrape_all(self, packages: list[PackageConfig]) -> list[ScrapedPage]:
        """Scrape documentation for all configured packages."""
        for package in packages:
            await self.scrape_package(package)
        return self.pages


def create_documentation_conversations(pages: list[ScrapedPage]) -> list[dict]:
    """
    Convert scraped documentation pages into training conversations.

    Creates various conversation formats:
    - Explanation requests
    - Code example requests
    - API reference questions
    """
    conversations = []

    templates = [
        # Explanation templates
        (
            "Explain {title} from the {package} library.",
            "{content}"
        ),
        (
            "What is {title} in {package} and how do I use it?",
            "{content}"
        ),
        (
            "Can you explain the {section} section of {package}?",
            "Here's information about {title}:\n\n{content}"
        ),
        # Documentation lookup
        (
            "Show me the documentation for {title} in {package}.",
            "{content}"
        ),
        (
            "I need help understanding {title} from {package}.",
            "Here's what you need to know about {title}:\n\n{content}"
        ),
    ]

    for page in pages:
        # Skip pages with very short content
        if len(page.content) < 200:
            continue

        # Create conversations using templates
        for user_template, assistant_template in templates:
            try:
                user_content = user_template.format(
                    title=page.title,
                    package=page.package_name,
                    section=page.section
                )
                assistant_content = assistant_template.format(
                    title=page.title,
                    package=page.package_name,
                    section=page.section,
                    content=page.content
                )

                conversation = {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content}
                    ],
                    "metadata": {
                        "source": "pypi_docs",
                        "package": page.package_name,
                        "url": page.url,
                        "section": page.section
                    }
                }
                conversations.append(conversation)

            except KeyError:
                continue

    return conversations


async def scrape_pypi_docs(
    packages: list[dict],
    scraper_config: dict,
    content_config: dict
) -> list[ScrapedPage]:
    """
    Main entry point for PyPI documentation scraping.

    Args:
        packages: List of package configurations
        scraper_config: Scraper settings
        content_config: Content filtering settings

    Returns:
        List of scraped pages
    """
    # Build config
    config = ScraperConfig(
        max_pages_per_package=scraper_config.get("max_pages_per_package", 500),
        request_delay=scraper_config.get("request_delay", 0.5),
        request_timeout=scraper_config.get("request_timeout", 30),
        max_retries=scraper_config.get("max_retries", 3),
        user_agent=scraper_config.get("user_agent", "NanochatDatasetBuilder/1.0"),
        concurrent_requests=scraper_config.get("concurrent_requests", 3),
        max_depth=scraper_config.get("max_depth", 5),
        min_content_length=content_config.get("min_content_length", 100),
        max_content_length=content_config.get("max_content_length", 50000),
        exclude_patterns=content_config.get("exclude_patterns", []),
        include_patterns=content_config.get("include_patterns", []),
    )

    # Build package configs
    package_configs = [
        PackageConfig(
            name=p["name"],
            docs_url=p["docs_url"],
            doc_type=p.get("doc_type", "sphinx")
        )
        for p in packages
    ]

    # Scrape
    async with DocumentationScraper(config) as scraper:
        pages = await scraper.scrape_all(package_configs)

    return pages


def save_scraped_pages(pages: list[ScrapedPage], output_path: Path):
    """Save scraped pages to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [page.to_dict() for page in pages]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(pages)} pages to {output_path}")


def save_conversations_jsonl(conversations: list[dict], output_path: Path):
    """Save conversations in JSONL format for training."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for conv in conversations:
            # Write in the format expected by customjson.py
            messages = conv["messages"]
            f.write(json.dumps(messages) + "\n")

    logger.info(f"Saved {len(conversations)} conversations to {output_path}")


if __name__ == "__main__":
    # Test the scraper
    import sys

    logging.basicConfig(level=logging.INFO)

    async def test():
        packages = [
            PackageConfig(
                name="requests",
                docs_url="https://requests.readthedocs.io/",
                doc_type="sphinx"
            )
        ]

        config = ScraperConfig(
            max_pages_per_package=10,
            max_depth=2
        )

        async with DocumentationScraper(config) as scraper:
            pages = await scraper.scrape_all(packages)

        print(f"\nScraped {len(pages)} pages:")
        for page in pages[:5]:
            print(f"  - {page.title}: {len(page.content)} chars")

        # Create conversations
        conversations = create_documentation_conversations(pages)
        print(f"\nCreated {len(conversations)} conversations")

    asyncio.run(test())
