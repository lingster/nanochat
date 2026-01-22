"""Job orchestration for data generation."""

import os
from pathlib import Path
from typing import List

from .config.schema import Config, JobConfig, SourceConfig
from .config.loader import get_source_options, get_output_options
from .extractors.web_scraper import WebScraper
from .extractors.file_scanner import FileScanner
from .extractors.github_pr import GitHubPRInspector
from .parsers.html_parser import HTMLParser
from .parsers.markdown_parser import MarkdownParser, MarkdownQAParser
from .parsers.code_parser import CodeParser
from .processors.cleaner import TextCleaner
from .processors.validator import ContentValidator
from .processors.deduplicator import Deduplicator
from .processors.formatter import ConversationFormatter, PRFormatter
from .generators.parquet_gen import ParquetGenerator
from .generators.jsonl_conv_gen import JSONLConversationGenerator
from .generators.jsonl_code_gen import JSONLCodeGenerator
from .utils.progress import ProgressTracker
from .utils.state import StateManager


class JobOrchestrator:
    """Orchestrates data extraction and generation jobs."""

    def __init__(self, config: Config, logger=None, resume: bool = True):
        """Initialize orchestrator.

        Args:
            config: Configuration object
            logger: Optional logger
            resume: Whether to resume from previous state (default: True)
        """
        self.config = config
        self.logger = logger
        self.resume = resume

        # Create output directory
        output_dir = Path(config.global_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize state manager
        cache_dir = Path(config.global_config.cache_dir)
        self.state_manager = StateManager(str(cache_dir))

    def run_all_jobs(self):
        """Run all configured jobs."""
        self.log('info', f"Running {len(self.config.jobs)} job(s)")

        for i, job in enumerate(self.config.jobs):
            self.log('info', f"\n{'='*60}")
            self.log('info', f"Job {i+1}/{len(self.config.jobs)}: {job.name}")
            self.log('info', f"{'='*60}")

            try:
                self.run_job(job)
            except Exception as e:
                self.log('error', f"Job {job.name} failed: {e}")
                raise

    def run_job(self, job: JobConfig):
        """Run a single job.

        Args:
            job: Job configuration
        """
        # Get job state
        job_state = self.state_manager.get_job_state(job.name, resume=self.resume)

        # Log resume info
        if self.resume and job_state.get_statistics()['total_items_processed'] > 0:
            stats = job_state.get_statistics()
            self.log('info', f"Resuming job - previously processed: {stats['total_items_processed']} items")
            self.log('info', f"  - URLs: {stats['processed_urls']}")
            self.log('info', f"  - Files: {stats['processed_files']}")
            self.log('info', f"  - PRs: {stats['processed_prs']}")

        # Get output options
        output_options = get_output_options(job)

        # Create generator
        output_path = Path(self.config.global_config.output_dir) / job.output_file
        generator = self._create_generator(job.type, str(output_path), output_options, job_state)

        # Create processors
        cleaner = TextCleaner()
        validator = ContentValidator()
        deduplicator = Deduplicator() if getattr(output_options, 'deduplication', True) else None

        # Process each source
        total_items = 0
        duplicates = 0
        items_since_save = 0

        for source in job.sources:
            self.log('info', f"Processing source: {source.type}")

            # Create extractor
            extractor = self._create_extractor(source, job_state)

            # Extract and process content
            for extracted in extractor.extract():
                # Clean text
                clean_text = cleaner.clean(
                    extracted.text,
                    options=self.config.quality.__dict__ if self.config.quality else {}
                )

                # Validate
                min_length = getattr(output_options, 'min_text_length', 50)
                max_length = getattr(output_options, 'max_text_length', 100000)

                if not validator.is_valid(clean_text, min_length, max_length):
                    continue

                # Check for duplicates
                if deduplicator and deduplicator.is_duplicate(clean_text):
                    duplicates += 1
                    job_state.increment_duplicates()
                    continue

                # Format based on job type
                formatted = self._format_content(
                    job.type,
                    clean_text,
                    extracted.metadata,
                    source
                )

                if formatted:
                    generator.add(formatted)
                    total_items += 1
                    items_since_save += 1

                    # Periodically save state (every 10 items)
                    if items_since_save >= 10:
                        job_state.save()
                        items_since_save = 0

        # Final save
        job_state.save()

        # Finalize output
        output_files = generator.finalize()

        # Get final statistics
        stats = job_state.get_statistics()

        self.log('info', f"\nJob {job.name} completed:")
        self.log('info', f"  - Items processed this run: {total_items}")
        self.log('info', f"  - Total items processed: {stats['total_items_processed']}")
        self.log('info', f"  - Items skipped (already processed): {stats['total_items_skipped']}")
        if deduplicator:
            self.log('info', f"  - Duplicates detected: {stats['total_duplicates']}")
        self.log('info', f"  - Output files: {len(output_files)}")
        for f in output_files:
            self.log('info', f"    - {f}")

    def _create_extractor(self, source: SourceConfig, state=None):
        """Create extractor for a source.

        Args:
            source: Source configuration
            state: Optional JobState for resume functionality

        Returns:
            Extractor instance
        """
        options = get_source_options(source)

        if source.type == "web":
            return WebScraper(source.urls, options, self.logger, state)

        elif source.type == "files":
            return FileScanner(source.paths, options, self.logger, state)

        elif source.type == "github_pr":
            github_token = self.config.auth.github_token if self.config.auth else None
            if not github_token:
                raise ValueError("GitHub token required for github_pr source")
            return GitHubPRInspector(source.repositories, github_token, options, self.logger, state)

        else:
            raise ValueError(f"Unsupported source type: {source.type}")

    def _create_generator(self, job_type: str, output_path: str, options, state=None):
        """Create generator for a job type.

        Args:
            job_type: Type of job
            output_path: Output file path
            options: Generator options
            state: Optional JobState for resume functionality

        Returns:
            Generator instance
        """
        if job_type == "parquet":
            return ParquetGenerator(output_path, options, self.logger, state)

        elif job_type == "jsonl_conversation":
            return JSONLConversationGenerator(output_path, options, self.logger)

        elif job_type == "jsonl_code":
            return JSONLCodeGenerator(output_path, options, self.logger)

        else:
            raise ValueError(f"Unsupported job type: {job_type}")

    def _format_content(self, job_type: str, text: str, metadata: dict, source: SourceConfig):
        """Format content based on job type.

        Args:
            job_type: Type of job
            text: Cleaned text
            metadata: Content metadata
            source: Source configuration

        Returns:
            Formatted content for generator
        """
        if job_type == "parquet":
            # Simple text format
            return {'text': text}

        elif job_type == "jsonl_conversation":
            # Check if source has a special parser
            parser = source.options.get('parser') if source.options else None

            if parser == "markdown_qa":
                # Parse markdown into Q&A pairs
                md_parser = MarkdownQAParser()
                qa_pairs = md_parser.parse(text, metadata)

                # Create conversations from Q&A pairs
                conversations = []
                for qa in qa_pairs:
                    conversations.append(
                        ConversationFormatter.create_simple_conversation(
                            qa['question'],
                            qa['answer']
                        )
                    )
                return conversations[0] if conversations else None

            else:
                # Default: create simple Q&A from title and content
                title = metadata.get('title') or metadata.get('file_name', 'Question')
                return ConversationFormatter.create_simple_conversation(
                    title,
                    text
                )

        elif job_type == "jsonl_code":
            # Format PR data as code conversation
            if 'commits' in metadata and source.type == "github_pr":
                return PRFormatter.format_pr_as_conversation(
                    metadata.get('pr_title', ''),
                    text,  # PR description
                    metadata.get('commits', []),
                    parse_commits=True
                )
            else:
                # Fallback: simple code conversation
                return ConversationFormatter.create_code_conversation(
                    "Code example",
                    "Here's the code:",
                    text,
                    language='python'
                )

        return None

    def log(self, level: str, message: str):
        """Log a message."""
        if self.logger:
            getattr(self.logger, level)(message)
        else:
            print(f"[{level.upper()}] {message}")
