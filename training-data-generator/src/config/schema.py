"""Configuration schema definitions."""

from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass


@dataclass
class GlobalConfig:
    """Global configuration settings."""
    output_dir: str = "./output"
    cache_dir: str = "./.cache"
    log_level: str = "INFO"
    num_workers: int = 4


@dataclass
class AuthConfig:
    """Authentication configuration."""
    github_token: Optional[str] = None


@dataclass
class WebSourceOptions:
    """Options for web scraping sources."""
    follow_links: bool = False
    max_depth: int = 1
    url_pattern: Optional[str] = None
    exclude_pattern: Optional[str] = None
    same_domain_only: bool = True
    max_pages: int = 1000
    rate_limit: float = 1.0
    parser: str = "html"


@dataclass
class FileSourceOptions:
    """Options for file scanning sources."""
    recursive: bool = True
    encoding: str = "utf-8"
    parser: str = "text"


@dataclass
class GitHubPRSourceOptions:
    """Options for GitHub PR sources."""
    state: Literal["open", "closed", "all"] = "closed"
    merged_only: bool = True
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    labels: Optional[List[str]] = None
    min_commits: int = 1
    max_commits: int = 20
    min_changes: int = 10
    max_changes: int = 500
    max_diff_lines: int = 500
    languages: Optional[List[str]] = None
    exclude_authors: Optional[List[str]] = None
    include_diff: bool = True
    max_prs_per_repo: Optional[int] = None


@dataclass
class SourceConfig:
    """Configuration for a data source."""
    type: Literal["web", "files", "github_pr", "git_local"]
    urls: Optional[List[str]] = None
    paths: Optional[List[str]] = None
    repositories: Optional[List[str]] = None
    options: Optional[Dict[str, Any]] = None


@dataclass
class ParquetOutputOptions:
    """Options for Parquet output."""
    shard_size_mb: int = 100
    compression: str = "zstd"
    row_group_size: int = 1024
    min_text_length: int = 100
    max_text_length: int = 1000000
    deduplication: bool = True


@dataclass
class ConversationOutputOptions:
    """Options for conversation JSONL output."""
    max_turns: int = 20
    validate_roles: bool = True
    min_content_length: int = 10
    deduplicate: bool = True


@dataclass
class CodeOutputOptions:
    """Options for code JSONL output."""
    languages: Optional[List[str]] = None
    max_diff_lines: int = 500
    include_context: bool = True
    parse_commit_message: bool = True


@dataclass
class JobConfig:
    """Configuration for a data collection job."""
    name: str
    type: Literal["parquet", "jsonl_conversation", "jsonl_code"]
    output_file: str
    sources: List[SourceConfig]
    options: Optional[Dict[str, Any]] = None


@dataclass
class QualityConfig:
    """Data quality configuration."""
    min_text_length: int = 50
    max_text_length: int = 100000
    allowed_languages: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None
    content_filters: Optional[List[str]] = None


@dataclass
class Config:
    """Main configuration object."""
    version: str
    global_config: GlobalConfig
    auth: Optional[AuthConfig] = None
    jobs: List[JobConfig] = None
    quality: Optional[QualityConfig] = None
