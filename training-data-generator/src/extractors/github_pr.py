"""GitHub Pull Request inspector extractor."""

from typing import Iterator, Optional
from datetime import datetime, timezone
from github import Github, GithubException
import time

from .base import BaseExtractor, ExtractedContent
from ..config.schema import GitHubPRSourceOptions


class GitHubPRInspector(BaseExtractor):
    """Extractor for GitHub Pull Requests."""

    def __init__(self, repositories: list, github_token: str, options: GitHubPRSourceOptions, logger=None, state=None):
        """Initialize GitHub PR inspector.

        Args:
            repositories: List of repositories (format: owner/repo)
            github_token: GitHub API token
            options: GitHub PR options
            logger: Optional logger
            state: Optional JobState for resume functionality
        """
        super().__init__(logger)
        self.repositories = repositories
        self.options = options
        self.state = state
        self.github = Github(github_token)

    def extract(self) -> Iterator[ExtractedContent]:
        """Extract content from GitHub PRs.

        Yields:
            ExtractedContent objects with PR data
        """
        for repo_name in self.repositories:
            self.log('info', f"Processing repository: {repo_name}")

            try:
                repo = self.github.get_repo(repo_name)
                pr_count = 0

                # Get pull requests
                prs = repo.get_pulls(
                    state=self.options.state,
                    sort='created',
                    direction='desc'
                )

                for pr in prs:
                    if pr_count >= self.options.max_prs_per_repo:
                        self.log('info', f"Reached max PRs for {repo_name}: {self.options.max_prs_per_repo}")
                        break

                    # Check if already processed
                    pr_id = f"{repo_name}#{pr.number}"
                    if self.state and self.state.is_processed('github_prs', pr_id):
                        self.log('debug', f"PR already processed, skipping: {pr_id}")
                        if self.state:
                            self.state.increment_skipped()
                        continue

                    # Apply filters
                    if not self._should_process_pr(pr):
                        continue

                    # Extract PR data
                    content = self._extract_pr_content(pr, repo_name)
                    if content:
                        # Mark as processed in state
                        if self.state:
                            self.state.mark_processed('github_prs', pr_id)

                        yield content
                        pr_count += 1

                    # Rate limiting pause
                    time.sleep(0.5)

            except GithubException as e:
                self.log('error', f"GitHub API error for {repo_name}: {e}")
                if e.status == 403:
                    self.log('warning', "Rate limit exceeded. Waiting...")
                    time.sleep(60)
            except Exception as e:
                self.log('error', f"Failed to process repository {repo_name}: {e}")

    def _should_process_pr(self, pr) -> bool:
        """Check if PR should be processed based on filters.

        Args:
            pr: GitHub PR object

        Returns:
            True if PR should be processed
        """
        pr_created_at = self._normalize_datetime(pr.created_at)

        # Check if merged (if required)
        if self.options.merged_only and not pr.merged:
            return False

        # Check date range
        if self.options.date_from:
            date_from = self._parse_datetime(self.options.date_from)
            if pr_created_at < date_from:
                return False

        if self.options.date_to:
            date_to = self._parse_datetime(self.options.date_to)
            if pr_created_at > date_to:
                return False

        # Check labels
        if self.options.labels:
            pr_labels = [label.name for label in pr.labels]
            if not any(label in pr_labels for label in self.options.labels):
                return False

        # Check commits count
        commits_count = pr.commits
        if commits_count < self.options.min_commits or commits_count > self.options.max_commits:
            return False

        # Check changes count
        changes_count = pr.additions + pr.deletions
        if changes_count < self.options.min_changes or changes_count > self.options.max_changes:
            return False

        # Check author exclusions
        if self.options.exclude_authors:
            if pr.user.login in self.options.exclude_authors:
                return False

        return True

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return GitHubPRInspector._normalize_datetime(parsed)

    @staticmethod
    def _normalize_datetime(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _extract_pr_content(self, pr, repo_name: str) -> Optional[ExtractedContent]:
        """Extract content from a PR.

        Args:
            pr: GitHub PR object
            repo_name: Repository name

        Returns:
            ExtractedContent object or None
        """
        try:
            self.log('info', f"Extracting PR #{pr.number}: {pr.title}")

            # Build PR description
            pr_description = f"{pr.title}\n\n"
            if pr.body:
                pr_description += pr.body

            # Extract commits and diffs
            commits_data = []
            if self.options.include_diff:
                for commit in pr.get_commits():
                    commit_data = {
                        'sha': commit.sha,
                        'message': commit.commit.message,
                        'files': []
                    }

                    # Get file changes
                    for file in commit.files:
                        # Filter by language if specified
                        if self.options.languages:
                            file_ext = file.filename.split('.')[-1]
                            if file_ext not in self.options.languages:
                                continue

                        # Check diff size
                        if file.changes > self.options.max_diff_lines:
                            continue

                        commit_data['files'].append({
                            'filename': file.filename,
                            'status': file.status,
                            'additions': file.additions,
                            'deletions': file.deletions,
                            'patch': file.patch if hasattr(file, 'patch') else None
                        })

                    commits_data.append(commit_data)

            metadata = {
                'repo': repo_name,
                'pr_number': pr.number,
                'pr_title': pr.title,
                'pr_url': pr.html_url,
                'pr_state': pr.state,
                'pr_merged': pr.merged,
                'pr_created_at': pr.created_at.isoformat(),
                'pr_author': pr.user.login,
                'commits': commits_data,
                'pr_description': pr_description,
            }

            # Store PR description as text (will be converted to conversation format by generator)
            return ExtractedContent(
                text=pr_description,
                metadata=metadata,
                source=f"{repo_name}#{pr.number}"
            )

        except Exception as e:
            self.log('error', f"Failed to extract PR #{pr.number}: {e}")
            return None
