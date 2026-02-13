"""Configuration loader for YAML files."""

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml

from .schema import (
    Config, GlobalConfig, AuthConfig, JobConfig, SourceConfig,
    QualityConfig, ParquetOutputOptions, ConversationOutputOptions,
    CodeOutputOptions, WebSourceOptions, FileSourceOptions,
    GitHubPRSourceOptions
)


def expand_env_vars(text: str) -> str:
    """Expand environment variables in text.

    Supports ${VAR_NAME} format.
    """
    def replace_var(match):
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    return re.sub(r'\$\{([^}]+)\}', replace_var, text)


def expand_env_vars_in_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively expand environment variables in dictionary."""
    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = expand_env_vars(value)
        elif isinstance(value, dict):
            result[key] = expand_env_vars_in_dict(value)
        elif isinstance(value, list):
            result[key] = [
                expand_env_vars(item) if isinstance(item, str)
                else expand_env_vars_in_dict(item) if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def load_config(config_path: str, output_dir_override: str = None) -> Config:
    """Load and parse configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file
        output_dir_override: Optional CLI override for output_dir

    Returns:
        Parsed Config object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load YAML
    with open(config_file, 'r') as f:
        raw_config = yaml.safe_load(f)

    # Expand environment variables
    raw_config = expand_env_vars_in_dict(raw_config)

    # Parse configuration
    try:
        # Global config with output_dir priority:
        # 1. CLI argument (output_dir_override)
        # 2. Config file value
        # 3. Environment variable (NANOCHAT_DATA_DIR)
        # 4. Default ("./output")
        global_data = raw_config.get('global', {})

        # Determine output_dir based on priority
        if output_dir_override:
            global_data['output_dir'] = output_dir_override
        elif 'output_dir' not in global_data or not global_data['output_dir']:
            # Check environment variable
            env_data_dir = os.environ.get('NANOCHAT_DATA_DIR')
            if env_data_dir:
                global_data['output_dir'] = env_data_dir

        global_config = GlobalConfig(**global_data)

        # Create output directory if it doesn't exist
        try:
            output_path = Path(global_config.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise ValueError(f"Failed to create output directory '{global_config.output_dir}': {e}")

        # Auth config
        auth_data = raw_config.get('auth', {})
        auth = AuthConfig(**auth_data) if auth_data else None

        # Quality config
        quality_data = raw_config.get('quality', {})
        quality = QualityConfig(**quality_data) if quality_data else None

        # Jobs
        jobs = []
        for job_data in raw_config.get('jobs', []):
            # Parse sources
            sources = []
            for source_data in job_data.get('sources', []):
                sources.append(SourceConfig(
                    type=source_data['type'],
                    urls=source_data.get('urls'),
                    paths=source_data.get('paths'),
                    repositories=source_data.get('repositories'),
                    options=source_data.get('options', {})
                ))

            jobs.append(JobConfig(
                name=job_data['name'],
                type=job_data['type'],
                output_file=job_data['output_file'],
                sources=sources,
                options=job_data.get('options', {})
            ))

        return Config(
            version=raw_config['version'],
            global_config=global_config,
            auth=auth,
            jobs=jobs,
            quality=quality
        )

    except KeyError as e:
        raise ValueError(f"Missing required configuration key: {e}")
    except TypeError as e:
        raise ValueError(f"Invalid configuration value: {e}")


def get_source_options(source: SourceConfig) -> Any:
    """Get typed options object for a source based on its type."""
    if source.type == "web":
        return WebSourceOptions(**source.options) if source.options else WebSourceOptions()
    elif source.type == "files":
        return FileSourceOptions(**source.options) if source.options else FileSourceOptions()
    elif source.type == "github_pr":
        return GitHubPRSourceOptions(**source.options) if source.options else GitHubPRSourceOptions()
    else:
        return source.options or {}


def get_output_options(job: JobConfig) -> Any:
    """Get typed options object for a job based on its output type."""
    if job.type == "parquet":
        return ParquetOutputOptions(**job.options) if job.options else ParquetOutputOptions()
    elif job.type == "jsonl_conversation":
        return ConversationOutputOptions(**job.options) if job.options else ConversationOutputOptions()
    elif job.type == "jsonl_code":
        return CodeOutputOptions(**job.options) if job.options else CodeOutputOptions()
    else:
        return job.options or {}
