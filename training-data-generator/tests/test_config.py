"""Tests for configuration loading and validation."""

import pytest
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.loader import load_config, expand_env_vars
from config.validator import ConfigValidator
from config.schema import Config


def test_expand_env_vars():
    """Test environment variable expansion."""
    import os
    os.environ['TEST_VAR'] = 'test_value'

    text = "Token: ${TEST_VAR}"
    result = expand_env_vars(text)

    assert result == "Token: test_value"


def test_load_valid_config():
    """Test loading a valid configuration."""
    config_content = """
version: "1.0"

global:
  output_dir: "./output"
  log_level: "INFO"

jobs:
  - name: "test_job"
    type: "parquet"
    output_file: "test.parquet"
    sources:
      - type: "web"
        urls:
          - "https://example.com"
    """

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        config_path = f.name

    try:
        config = load_config(config_path)

        assert config.version == "1.0"
        assert config.global_config.output_dir == "./output"
        assert len(config.jobs) == 1
        assert config.jobs[0].name == "test_job"
    finally:
        Path(config_path).unlink()


def test_validate_invalid_job_type():
    """Test validation of invalid job type."""
    config_content = """
version: "1.0"

global:
  output_dir: "./output"

jobs:
  - name: "test_job"
    type: "invalid_type"
    output_file: "test.parquet"
    sources:
      - type: "web"
        urls:
          - "https://example.com"
    """

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        config_path = f.name

    try:
        config = load_config(config_path)
        errors = ConfigValidator.validate_config(config)

        assert len(errors) > 0
        assert any('Invalid job type' in error for error in errors)
    finally:
        Path(config_path).unlink()


def test_validate_missing_source_urls():
    """Test validation of missing URLs for web source."""
    config_content = """
version: "1.0"

global:
  output_dir: "./output"

jobs:
  - name: "test_job"
    type: "parquet"
    output_file: "test.parquet"
    sources:
      - type: "web"
    """

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        config_path = f.name

    try:
        config = load_config(config_path)
        errors = ConfigValidator.validate_config(config)

        assert len(errors) > 0
        assert any('urls' in error.lower() for error in errors)
    finally:
        Path(config_path).unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
