# Python Dataset Builder

A tool for building Python-focused training datasets for the nanochat LLM. This tool collects data from two main sources:

1. **PyPI Documentation** - Scrapes documentation from popular Python packages
2. **GitHub Pull Requests** - Creates before/after datasets from merged PRs

## Features

- Configurable via YAML file
- Async/concurrent scraping for performance
- Support for multiple documentation formats (Sphinx, MkDocs, custom)
- GitHub API integration with rate limiting
- Automatic deduplication
- Train/test splitting
- Multiple output formats (JSONL, Parquet)

## Installation

```bash
cd tools/python_dataset_builder
pip install -r requirements.txt
```

## Quick Start

1. **Configure your dataset** by editing `config.yaml`:
   - Add/remove PyPI packages to scrape
   - Add/remove GitHub repositories
   - Adjust filtering and output settings

2. **Set up GitHub token** (recommended for higher rate limits):
   ```bash
   export GITHUB_TOKEN=ghp_your_token_here
   ```

3. **Run the builder**:
   ```bash
   python -m tools.python_dataset_builder.build_dataset
   ```

## Usage

### Full Dataset Build

```bash
# Use default config.yaml
python -m tools.python_dataset_builder.build_dataset

# Custom config file
python -m tools.python_dataset_builder.build_dataset --config my_config.yaml

# Custom output directory
python -m tools.python_dataset_builder.build_dataset --output /path/to/output
```

### Partial Builds

```bash
# Only scrape PyPI documentation
python -m tools.python_dataset_builder.build_dataset --pypi-only

# Only create GitHub PR dataset
python -m tools.python_dataset_builder.build_dataset --github-only
```

### Options

```
--config, -c     Path to configuration YAML file (default: config.yaml)
--output, -o     Output directory (overrides config)
--pypi-only      Only run PyPI documentation scraping
--github-only    Only run GitHub PR dataset creation
--no-dedupe      Skip deduplication
--verbose, -v    Enable verbose logging
```

## Configuration

See `config.yaml` for the full configuration reference. Key sections:

### Output Settings

```yaml
output:
  base_dir: "~/.cache/nanochat/python_dataset"
  formats:
    - jsonl      # Compatible with nanochat's CustomJson task
    - parquet    # Efficient columnar storage
```

### PyPI Documentation

```yaml
pypi_docs:
  enabled: true
  packages:
    - name: requests
      docs_url: https://requests.readthedocs.io/
      doc_type: sphinx
    # Add more packages...

  scraper:
    max_pages_per_package: 500
    request_delay: 0.5
    max_depth: 5
```

### GitHub PRs

```yaml
github_prs:
  enabled: true
  # token: "ghp_xxx"  # Or use GITHUB_TOKEN env var

  repositories:
    - owner: python
      repo: cpython
      max_prs: 100
      since: "2023-01-01"
      file_patterns:
        - "*.py"

  filters:
    min_changed_files: 1
    max_changed_files: 20
    exclude_authors:
      - "dependabot[bot]"
```

## Output Format

The builder generates conversations in the format expected by nanochat's training pipeline:

### JSONL Format (conversations.jsonl)

Each line is a JSON array of messages:

```json
[{"role":"user","content":"Explain..."},{"role":"assistant","content":"..."}]
```

This format is directly compatible with the `CustomJson` task in `tasks/customjson.py`.

### Directory Structure

```
output_dir/
├── train/
│   ├── conversations.jsonl           # Training data
│   └── conversations_with_metadata.jsonl
├── test/
│   ├── conversations.jsonl           # Test data
│   └── conversations_with_metadata.jsonl
├── combined/
│   └── conversations.jsonl           # All data
├── pypi_docs/
│   ├── scraped_pages.json           # Raw scraped content
│   └── conversations.jsonl          # PyPI-only conversations
├── github_prs/
│   ├── pull_requests.json           # Raw PR data
│   └── conversations.jsonl          # PR-only conversations
└── dataset_info.json                # Statistics and metadata
```

## Using with nanochat Training

### Option 1: CustomJson Task

```python
from tasks.customjson import CustomJson

# Load your generated dataset
task = CustomJson(filepath="~/.cache/nanochat/python_dataset/train/conversations.jsonl")
train_data = task.get_train()
```

### Option 2: Add as New Task

Create a new task file in `tasks/`:

```python
# tasks/python_docs.py
from dataclasses import dataclass
from tasks.common import Task, Conversation, Message
import json

@dataclass
class PythonDocs(Task):
    filepath: str = "~/.cache/nanochat/python_dataset/train/conversations.jsonl"

    def get_train(self) -> list[Conversation]:
        return self._load_conversations("train")

    def get_test(self) -> list[Conversation]:
        return self._load_conversations("test")

    def _load_conversations(self, split: str) -> list[Conversation]:
        conversations = []
        with open(self.filepath.replace("train", split)) as f:
            for line in f:
                messages = json.loads(line)
                conv = Conversation(messages=[
                    Message(role=m["role"], content=m["content"])
                    for m in messages
                ])
                conversations.append(conv)
        return conversations
```

## Conversation Templates

### PyPI Documentation Conversations

The scraper generates conversations using these templates:

- "Explain {title} from the {package} library."
- "What is {title} in {package} and how do I use it?"
- "Show me the documentation for {title} in {package}."

### GitHub PR Conversations

PR-based conversations include:

- **PR explanation**: "Explain what changes were made in this pull request"
- **Code review**: "Review the following code changes and summarize"
- **File diff analysis**: "What changes were made to this Python file?"

## Extending

### Adding New Documentation Sources

Extend `pypi_scraper.py`:

```python
def _extract_content_my_format(self, soup: BeautifulSoup) -> tuple[str, str]:
    """Extract content from my custom documentation format."""
    # Implement extraction logic
    pass
```

### Custom Conversation Templates

Modify the `create_*_conversations` functions to add your own templates.

## Troubleshooting

### Rate Limiting

If you hit GitHub API rate limits:
1. Set `GITHUB_TOKEN` environment variable
2. Reduce `max_prs` in config
3. Increase `request_delay` in API config

### Memory Issues

For large datasets:
1. Process repositories one at a time
2. Reduce `max_pages_per_package` for PyPI scraping
3. Use streaming processing (modify code as needed)

### Missing Content

If documentation content is empty:
1. Check `doc_type` matches the site's format
2. Adjust `min_content_length` threshold
3. Review `exclude_patterns` in config

## License

Part of the nanochat project.
