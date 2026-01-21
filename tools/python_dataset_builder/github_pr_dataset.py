"""
GitHub PR Dataset Creator

Clones repositories and creates datasets from PR history, capturing:
- Repository state before the PR
- PR details (title, description, diff)
- Repository state after the PR

This creates a rich dataset for understanding code changes and their context.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from fnmatch import fnmatch

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class PRFile:
    """Represents a file changed in a PR."""
    filename: str
    status: str  # added, removed, modified, renamed
    additions: int
    deletions: int
    patch: Optional[str] = None
    content_before: Optional[str] = None
    content_after: Optional[str] = None
    previous_filename: Optional[str] = None  # For renames


@dataclass
class PullRequest:
    """Represents a GitHub Pull Request."""
    number: int
    title: str
    body: str
    author: str
    created_at: str
    merged_at: str
    base_sha: str  # Commit SHA before the PR
    merge_commit_sha: str  # Commit SHA after the PR
    head_sha: str  # The PR's head commit
    labels: list[str]
    files: list[PRFile] = field(default_factory=list)
    additions: int = 0
    deletions: int = 0
    changed_files: int = 0

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "title": self.title,
            "body": self.body,
            "author": self.author,
            "created_at": self.created_at,
            "merged_at": self.merged_at,
            "base_sha": self.base_sha,
            "merge_commit_sha": self.merge_commit_sha,
            "head_sha": self.head_sha,
            "labels": self.labels,
            "additions": self.additions,
            "deletions": self.deletions,
            "changed_files": self.changed_files,
            "files": [
                {
                    "filename": f.filename,
                    "status": f.status,
                    "additions": f.additions,
                    "deletions": f.deletions,
                    "patch": f.patch,
                    "content_before": f.content_before,
                    "content_after": f.content_after,
                    "previous_filename": f.previous_filename,
                }
                for f in self.files
            ]
        }


@dataclass
class RepoConfig:
    """Configuration for a repository to process."""
    owner: str
    repo: str
    max_prs: int = 100
    since: Optional[str] = None
    labels: list[str] = field(default_factory=list)
    file_patterns: list[str] = field(default_factory=lambda: ["*.py"])


@dataclass
class PRDatasetConfig:
    """Configuration for PR dataset creation."""
    # Filtering
    min_changed_files: int = 1
    max_changed_files: int = 20
    min_additions: int = 5
    max_additions: int = 2000
    exclude_labels: list[str] = field(default_factory=list)
    exclude_authors: list[str] = field(default_factory=list)

    # API settings
    rate_limit: int = 4000
    request_delay: float = 0.25
    request_timeout: int = 30
    max_retries: int = 3


class GitHubPRDatasetCreator:
    """
    Creates training datasets from GitHub Pull Requests.

    For each merged PR:
    1. Fetches PR metadata and file changes
    2. Retrieves file contents before and after the PR
    3. Creates structured dataset entries
    """

    GITHUB_API_BASE = "https://api.github.com"

    def __init__(self, token: Optional[str], config: PRDatasetConfig):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_remaining = 5000
        self._rate_limit_reset = 0

    async def __aenter__(self):
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "NanochatDatasetBuilder/1.0"
        }
        if self.token:
            headers["Authorization"] = f"token {self.token}"

        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _check_rate_limit(self):
        """Check and handle GitHub rate limiting."""
        if self._rate_limit_remaining < 10:
            now = time.time()
            if now < self._rate_limit_reset:
                sleep_time = self._rate_limit_reset - now + 1
                logger.warning(f"Rate limit nearly exhausted, sleeping {sleep_time:.0f}s")
                await asyncio.sleep(sleep_time)

    async def _api_request(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """Make a GitHub API request with retries and rate limiting."""
        await self._check_rate_limit()
        await asyncio.sleep(self.config.request_delay)

        url = f"{self.GITHUB_API_BASE}{endpoint}"

        for attempt in range(self.config.max_retries):
            try:
                async with self.session.get(url, params=params) as response:
                    # Update rate limit tracking
                    self._rate_limit_remaining = int(
                        response.headers.get("X-RateLimit-Remaining", 5000)
                    )
                    self._rate_limit_reset = int(
                        response.headers.get("X-RateLimit-Reset", 0)
                    )

                    if response.status == 200:
                        return await response.json()
                    elif response.status == 404:
                        logger.debug(f"Not found: {endpoint}")
                        return None
                    elif response.status == 403:
                        # Rate limited
                        reset_time = self._rate_limit_reset - time.time()
                        if reset_time > 0:
                            logger.warning(f"Rate limited, waiting {reset_time:.0f}s")
                            await asyncio.sleep(reset_time + 1)
                            continue
                    else:
                        logger.warning(f"HTTP {response.status} for {endpoint}")

            except asyncio.TimeoutError:
                logger.warning(f"Timeout for {endpoint} (attempt {attempt + 1})")
            except aiohttp.ClientError as e:
                logger.warning(f"Error for {endpoint}: {e} (attempt {attempt + 1})")

            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(2 ** attempt)

        return None

    async def _get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: str
    ) -> Optional[str]:
        """Get file content at a specific commit."""
        import base64

        endpoint = f"/repos/{owner}/{repo}/contents/{path}"
        data = await self._api_request(endpoint, {"ref": ref})

        if data and "content" in data:
            try:
                content = base64.b64decode(data["content"]).decode("utf-8")
                return content
            except (ValueError, UnicodeDecodeError):
                return None

        return None

    def _matches_file_patterns(self, filename: str, patterns: list[str]) -> bool:
        """Check if filename matches any of the given patterns."""
        if not patterns:
            return True
        return any(fnmatch(filename, p) for p in patterns)

    def _should_exclude_pr(self, pr_data: dict) -> bool:
        """Check if PR should be excluded based on filters."""
        # Check labels
        pr_labels = [label["name"] for label in pr_data.get("labels", [])]
        for exclude_label in self.config.exclude_labels:
            if exclude_label in pr_labels:
                return True

        # Check author
        author = pr_data.get("user", {}).get("login", "")
        if author in self.config.exclude_authors:
            return True

        # Check changed files count
        changed_files = pr_data.get("changed_files", 0)
        if changed_files < self.config.min_changed_files:
            return True
        if changed_files > self.config.max_changed_files:
            return True

        # Check additions
        additions = pr_data.get("additions", 0)
        if additions < self.config.min_additions:
            return True
        if additions > self.config.max_additions:
            return True

        return False

    async def fetch_merged_prs(self, repo_config: RepoConfig) -> list[dict]:
        """Fetch list of merged PRs from a repository."""
        logger.info(f"Fetching merged PRs from {repo_config.owner}/{repo_config.repo}")

        prs = []
        page = 1
        per_page = 100

        while True:
            endpoint = f"/repos/{repo_config.owner}/{repo_config.repo}/pulls"
            params = {
                "state": "closed",
                "sort": "updated",
                "direction": "desc",
                "per_page": per_page,
                "page": page
            }

            data = await self._api_request(endpoint, params)
            if not data:
                break

            for pr in data:
                # Only merged PRs
                if not pr.get("merged_at"):
                    continue

                # Check date filter
                if repo_config.since:
                    merged_at = pr.get("merged_at", "")
                    if merged_at < repo_config.since:
                        continue

                # Check label filter
                if repo_config.labels:
                    pr_labels = [l["name"] for l in pr.get("labels", [])]
                    if not any(l in pr_labels for l in repo_config.labels):
                        continue

                prs.append(pr)

                if repo_config.max_prs > 0 and len(prs) >= repo_config.max_prs:
                    break

            if len(data) < per_page:
                break

            if repo_config.max_prs > 0 and len(prs) >= repo_config.max_prs:
                break

            page += 1

        logger.info(f"Found {len(prs)} merged PRs")
        return prs[:repo_config.max_prs] if repo_config.max_prs > 0 else prs

    async def fetch_pr_details(
        self,
        owner: str,
        repo: str,
        pr_number: int
    ) -> Optional[dict]:
        """Fetch detailed PR information."""
        endpoint = f"/repos/{owner}/{repo}/pulls/{pr_number}"
        return await self._api_request(endpoint)

    async def fetch_pr_files(
        self,
        owner: str,
        repo: str,
        pr_number: int
    ) -> list[dict]:
        """Fetch files changed in a PR."""
        files = []
        page = 1

        while True:
            endpoint = f"/repos/{owner}/{repo}/pulls/{pr_number}/files"
            params = {"per_page": 100, "page": page}

            data = await self._api_request(endpoint, params)
            if not data:
                break

            files.extend(data)

            if len(data) < 100:
                break
            page += 1

        return files

    async def process_pr(
        self,
        owner: str,
        repo: str,
        pr_summary: dict,
        file_patterns: list[str]
    ) -> Optional[PullRequest]:
        """Process a single PR and extract all relevant data."""
        pr_number = pr_summary["number"]
        logger.info(f"Processing PR #{pr_number}: {pr_summary['title'][:50]}")

        # Fetch detailed PR info
        pr_details = await self.fetch_pr_details(owner, repo, pr_number)
        if not pr_details:
            return None

        # Check exclusion filters
        if self._should_exclude_pr(pr_details):
            logger.debug(f"PR #{pr_number} excluded by filters")
            return None

        # Fetch files changed
        files_data = await self.fetch_pr_files(owner, repo, pr_number)

        # Process each file
        files = []
        for f in files_data:
            filename = f["filename"]

            # Check file pattern
            if not self._matches_file_patterns(filename, file_patterns):
                continue

            # Get base SHA (before the PR)
            base_sha = pr_details.get("base", {}).get("sha")
            merge_commit_sha = pr_details.get("merge_commit_sha")

            # Fetch file contents
            content_before = None
            content_after = None

            if f["status"] != "added" and base_sha:
                prev_filename = f.get("previous_filename", filename)
                content_before = await self._get_file_content(
                    owner, repo, prev_filename, base_sha
                )

            if f["status"] != "removed" and merge_commit_sha:
                content_after = await self._get_file_content(
                    owner, repo, filename, merge_commit_sha
                )

            pr_file = PRFile(
                filename=filename,
                status=f["status"],
                additions=f["additions"],
                deletions=f["deletions"],
                patch=f.get("patch"),
                content_before=content_before,
                content_after=content_after,
                previous_filename=f.get("previous_filename")
            )
            files.append(pr_file)

        if not files:
            logger.debug(f"PR #{pr_number} has no matching files")
            return None

        # Create PR object
        pr = PullRequest(
            number=pr_number,
            title=pr_details["title"],
            body=pr_details.get("body") or "",
            author=pr_details["user"]["login"],
            created_at=pr_details["created_at"],
            merged_at=pr_details["merged_at"],
            base_sha=pr_details["base"]["sha"],
            merge_commit_sha=pr_details.get("merge_commit_sha", ""),
            head_sha=pr_details["head"]["sha"],
            labels=[l["name"] for l in pr_details.get("labels", [])],
            files=files,
            additions=pr_details["additions"],
            deletions=pr_details["deletions"],
            changed_files=pr_details["changed_files"]
        )

        return pr

    async def create_dataset(self, repo_config: RepoConfig) -> list[PullRequest]:
        """Create dataset from a repository's PRs."""
        # Fetch merged PRs
        pr_summaries = await self.fetch_merged_prs(repo_config)

        # Process each PR
        prs = []
        for pr_summary in pr_summaries:
            pr = await self.process_pr(
                repo_config.owner,
                repo_config.repo,
                pr_summary,
                repo_config.file_patterns
            )
            if pr:
                prs.append(pr)

        logger.info(f"Processed {len(prs)} PRs with matching files")
        return prs


def create_pr_conversations(prs: list[PullRequest], repo_name: str) -> list[dict]:
    """
    Convert PRs into training conversations.

    Creates various conversation formats focusing on understanding code changes.
    """
    conversations = []

    for pr in prs:
        # Skip PRs with no file content
        if not any(f.content_before or f.content_after for f in pr.files):
            continue

        # Build context showing file changes
        changes_context = []
        for f in pr.files:
            if f.status == "added":
                changes_context.append(f"### Added: `{f.filename}`\n```python\n{f.content_after or ''}\n```")
            elif f.status == "removed":
                changes_context.append(f"### Removed: `{f.filename}`\n(File was deleted)")
            elif f.status == "renamed":
                changes_context.append(
                    f"### Renamed: `{f.previous_filename}` → `{f.filename}`\n"
                    f"**Before:**\n```python\n{f.content_before or ''}\n```\n"
                    f"**After:**\n```python\n{f.content_after or ''}\n```"
                )
            else:  # modified
                changes_context.append(
                    f"### Modified: `{f.filename}`\n"
                    f"**Before:**\n```python\n{f.content_before or ''}\n```\n"
                    f"**After:**\n```python\n{f.content_after or ''}\n```"
                )

        changes_text = "\n\n".join(changes_context)

        # Truncate if too long
        max_context_length = 15000
        if len(changes_text) > max_context_length:
            changes_text = changes_text[:max_context_length] + "\n\n... (truncated)"

        # Template 1: Explain the PR
        conversations.append({
            "messages": [
                {
                    "role": "user",
                    "content": f"Explain what changes were made in this pull request:\n\n{changes_text}"
                },
                {
                    "role": "assistant",
                    "content": f"## {pr.title}\n\n{pr.body if pr.body else 'This PR makes the following changes to the codebase.'}"
                }
            ],
            "metadata": {
                "source": "github_pr",
                "repo": repo_name,
                "pr_number": pr.number,
                "type": "explain_pr"
            }
        })

        # Template 2: Code review perspective
        if pr.body:
            conversations.append({
                "messages": [
                    {
                        "role": "user",
                        "content": f"Review the following code changes and summarize the key modifications:\n\n{changes_text}"
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f"## Summary: {pr.title}\n\n"
                            f"**Author:** {pr.author}\n"
                            f"**Changes:** +{pr.additions}/-{pr.deletions} across {pr.changed_files} files\n\n"
                            f"### Description\n{pr.body}"
                        )
                    }
                ],
                "metadata": {
                    "source": "github_pr",
                    "repo": repo_name,
                    "pr_number": pr.number,
                    "type": "review_pr"
                }
            })

        # Template 3: Before/After understanding (for single file changes)
        for f in pr.files:
            if f.status == "modified" and f.content_before and f.content_after:
                # Only for reasonably sized files
                if len(f.content_before) < 5000 and len(f.content_after) < 5000:
                    conversations.append({
                        "messages": [
                            {
                                "role": "user",
                                "content": (
                                    f"What changes were made to this Python file?\n\n"
                                    f"**Before:**\n```python\n{f.content_before}\n```\n\n"
                                    f"**After:**\n```python\n{f.content_after}\n```"
                                )
                            },
                            {
                                "role": "assistant",
                                "content": (
                                    f"This file was modified as part of: **{pr.title}**\n\n"
                                    f"Changes: +{f.additions}/-{f.deletions} lines\n\n"
                                    f"{pr.body if pr.body else 'The file was updated with the changes shown.'}"
                                )
                            }
                        ],
                        "metadata": {
                            "source": "github_pr",
                            "repo": repo_name,
                            "pr_number": pr.number,
                            "file": f.filename,
                            "type": "file_diff"
                        }
                    })

    return conversations


async def create_pr_dataset(
    repositories: list[dict],
    filters: dict,
    api_config: dict,
    github_token: Optional[str] = None
) -> tuple[list[PullRequest], list[dict]]:
    """
    Main entry point for creating PR dataset.

    Args:
        repositories: List of repository configurations
        filters: PR filtering settings
        api_config: GitHub API settings
        github_token: GitHub API token (optional, can use GITHUB_TOKEN env var)

    Returns:
        Tuple of (list of PRs, list of conversations)
    """
    config = PRDatasetConfig(
        min_changed_files=filters.get("min_changed_files", 1),
        max_changed_files=filters.get("max_changed_files", 20),
        min_additions=filters.get("min_additions", 5),
        max_additions=filters.get("max_additions", 2000),
        exclude_labels=filters.get("exclude_labels", []),
        exclude_authors=filters.get("exclude_authors", []),
        rate_limit=api_config.get("rate_limit", 4000),
        request_delay=api_config.get("request_delay", 0.25),
        request_timeout=api_config.get("request_timeout", 30),
        max_retries=api_config.get("max_retries", 3),
    )

    all_prs = []
    all_conversations = []

    async with GitHubPRDatasetCreator(github_token, config) as creator:
        for repo_cfg in repositories:
            repo_config = RepoConfig(
                owner=repo_cfg["owner"],
                repo=repo_cfg["repo"],
                max_prs=repo_cfg.get("max_prs", 100),
                since=repo_cfg.get("since"),
                labels=repo_cfg.get("labels", []),
                file_patterns=repo_cfg.get("file_patterns", ["*.py"])
            )

            prs = await creator.create_dataset(repo_config)
            all_prs.extend(prs)

            repo_name = f"{repo_config.owner}/{repo_config.repo}"
            conversations = create_pr_conversations(prs, repo_name)
            all_conversations.extend(conversations)

    return all_prs, all_conversations


def save_prs_json(prs: list[PullRequest], output_path: Path):
    """Save PRs to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [pr.to_dict() for pr in prs]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(prs)} PRs to {output_path}")


def save_conversations_jsonl(conversations: list[dict], output_path: Path):
    """Save conversations in JSONL format for training."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for conv in conversations:
            messages = conv["messages"]
            f.write(json.dumps(messages) + "\n")

    logger.info(f"Saved {len(conversations)} conversations to {output_path}")


if __name__ == "__main__":
    # Test the PR dataset creator
    import sys

    logging.basicConfig(level=logging.INFO)

    async def test():
        repos = [{
            "owner": "psf",
            "repo": "requests",
            "max_prs": 5,
            "file_patterns": ["*.py"]
        }]

        filters = {
            "min_changed_files": 1,
            "max_changed_files": 10,
            "min_additions": 1,
            "exclude_authors": ["dependabot[bot]"]
        }

        api_config = {
            "request_delay": 0.5
        }

        prs, conversations = await create_pr_dataset(repos, filters, api_config)

        print(f"\nProcessed {len(prs)} PRs")
        print(f"Created {len(conversations)} conversations")

        for pr in prs[:3]:
            print(f"\n  PR #{pr.number}: {pr.title}")
            print(f"    Files: {len(pr.files)}, +{pr.additions}/-{pr.deletions}")

    asyncio.run(test())
