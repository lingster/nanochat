"""Configuration validation."""

import re
from typing import List
from .schema import Config, JobConfig, SourceConfig


class ConfigValidator:
    """Validates configuration objects."""

    @staticmethod
    def validate_config(config: Config) -> List[str]:
        """Validate entire configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Validate version
        if not config.version:
            errors.append("Configuration version is required")

        # Validate jobs
        if not config.jobs:
            errors.append("At least one job must be defined")
        else:
            for i, job in enumerate(config.jobs):
                job_errors = ConfigValidator.validate_job(job)
                errors.extend([f"Job {i} ({job.name}): {err}" for err in job_errors])

        return errors

    @staticmethod
    def validate_job(job: JobConfig) -> List[str]:
        """Validate a job configuration.

        Returns:
            List of validation errors
        """
        errors = []

        # Validate required fields
        if not job.name:
            errors.append("Job name is required")

        if not job.type:
            errors.append("Job type is required")
        elif job.type not in ["parquet", "jsonl_conversation", "jsonl_code"]:
            errors.append(f"Invalid job type: {job.type}")

        if not job.output_file:
            errors.append("Output file is required")

        # Validate sources
        if not job.sources:
            errors.append("At least one source is required")
        else:
            for i, source in enumerate(job.sources):
                source_errors = ConfigValidator.validate_source(source)
                errors.extend([f"Source {i}: {err}" for err in source_errors])

        return errors

    @staticmethod
    def validate_source(source: SourceConfig) -> List[str]:
        """Validate a source configuration.

        Returns:
            List of validation errors
        """
        errors = []

        # Validate type
        if not source.type:
            errors.append("Source type is required")
        elif source.type not in ["web", "files", "github_pr", "git_local"]:
            errors.append(f"Invalid source type: {source.type}")

        # Validate type-specific requirements
        if source.type == "web":
            if not source.urls:
                errors.append("Web source requires 'urls' field")
            else:
                for url in source.urls:
                    if not ConfigValidator._is_valid_url(url):
                        errors.append(f"Invalid URL: {url}")

        elif source.type == "files":
            if not source.paths:
                errors.append("File source requires 'paths' field")

        elif source.type == "github_pr":
            if not source.repositories:
                errors.append("GitHub PR source requires 'repositories' field")
            else:
                for repo in source.repositories:
                    if not ConfigValidator._is_valid_github_repo(repo):
                        errors.append(f"Invalid GitHub repository format: {repo} (expected: owner/repo)")

        elif source.type == "git_local":
            if not source.paths:
                errors.append("Git local source requires 'paths' field")

        return errors

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Check if string is a valid URL."""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,63}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IP
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return bool(url_pattern.match(url))

    @staticmethod
    def _is_valid_github_repo(repo: str) -> bool:
        """Check if string is a valid GitHub repository (owner/repo format)."""
        repo_pattern = re.compile(r'^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$')
        return bool(repo_pattern.match(repo))
