# Training Data Generator

A Python tool for generating training datasets for LLM training in three formats:
1. **Parquet** - Raw text for pretraining
2. **JSONL Conversations** - Q&A pairs for finetuning
3. **JSONL Code** - GitHub PR-based code examples

## Features

- **Multiple Data Sources**
  - Web scraping with rate limiting and robots.txt compliance
  - Local file system scanning with glob patterns
  - GitHub Pull Request inspection via API

- **Three Output Formats**
  - Parquet files with ZSTD compression (for pretraining)
  - JSONL conversation format (for chat finetuning)
  - JSONL code format with structured content (for code training)

- **Data Quality Controls**
  - Deduplication (hash-based)
  - Content validation (length, structure)
  - Text cleaning and normalization
  - Configurable filters

- **GitHub PR Inspector**
  - Extract PR descriptions as "questions"
  - Extract commit diffs as "code solutions"
  - Filter by date, labels, merge status
  - Support for multiple programming languages

## Installation

```bash
cd training-data-generator
pip install -r requirements.txt
```

## Quick Start

1. Set up your configuration file (see `examples/` directory)

2. For GitHub PR extraction, set your token:
```bash
export GITHUB_TOKEN="your_github_token_here"
```

3. Run the generator:
```bash
python generate_data.py --config examples/config_github_code.yaml
```

## Configuration

Configuration is done via YAML files. See the `examples/` directory for complete examples.

### Basic Structure

```yaml
version: "1.0"

global:
  output_dir: "./output"
  log_level: "INFO"

auth:
  github_token: "${GITHUB_TOKEN}"

jobs:
  - name: "job_name"
    type: "parquet"  # or "jsonl_conversation" or "jsonl_code"
    output_file: "output.parquet"
    sources: [...]
    options: {...}
```

### Data Sources

#### Web Scraping
```yaml
- type: "web"
  urls:
    - "https://example.com/docs"
  options:
    follow_links: true
    max_depth: 2
    url_pattern: "^https://example.com/docs/"
    max_pages: 100
    rate_limit: 1.0
```

#### File Scanning
```yaml
- type: "files"
  paths:
    - "./docs/**/*.md"
    - "./src/**/*.py"
  options:
    recursive: true
    encoding: "utf-8"
```

#### GitHub PRs
```yaml
- type: "github_pr"
  repositories:
    - "owner/repo"
  options:
    state: "closed"
    merged_only: true
    date_from: "2024-01-01"
    max_prs_per_repo: 100
    languages: ["py", "js"]
    exclude_authors: ["dependabot"]
```

## Output Formats

### 1. Parquet (Pretraining)

**Schema**: `{text: string}`

```python
# Reading Parquet files
import pyarrow.parquet as pq

table = pq.read_table("output_shard_00000.parquet")
for row in table.to_pylist():
    print(row['text'])
```

### 2. JSONL Conversation (Finetuning)

**Format**: One conversation per line

```json
[
  {"role": "user", "content": "What is Python?"},
  {"role": "assistant", "content": "Python is a programming language..."}
]
```

### 3. JSONL Code (Code Training)

**Format**: Structured assistant content

```json
[
  {
    "role": "user",
    "content": "Fix the authentication bug in the login function"
  },
  {
    "role": "assistant",
    "content": [
      {"type": "text", "text": "The issue was in password validation..."},
      {"type": "python", "text": "def login(username, password):\n    ..."}
    ]
  }
]
```

## Examples

### Example 1: Generate Pretraining Data from Web

```bash
python generate_data.py --config examples/config_parquet.yaml
```

This will scrape Wikipedia articles and local files, generating Parquet shards.

### Example 2: Generate Q&A Conversations

```bash
python generate_data.py --config examples/config_conversations.yaml
```

This will parse Markdown FAQ files into conversation format.

### Example 3: Extract Code from GitHub PRs

```bash
export GITHUB_TOKEN="ghp_your_token_here"
python generate_data.py --config examples/config_github_code.yaml
```

This will extract PR descriptions and code changes from popular Python repositories.

## GitHub PR Inspector Details

The GitHub PR inspector is particularly powerful for generating code training data:

### How It Works

1. **Extracts PR metadata**: title, description, commits, diffs
2. **Formats as conversation**:
   - User message = PR title + description (the problem/request)
   - Assistant message = Commit messages + code changes (the solution)

3. **Structured code format**:
   ```json
   {
     "role": "assistant",
     "content": [
       {"type": "text", "text": "Fixed the bug by..."},
       {"type": "python", "text": "# code changes here"}
     ]
   }
   ```

### Filtering Options

- **State**: open, closed, or all PRs
- **Merged only**: only include merged PRs
- **Date range**: filter by creation date
- **Labels**: filter by PR labels
- **Commit count**: min/max commits per PR
- **Change count**: min/max lines changed
- **Languages**: filter by file extensions
- **Exclude authors**: skip bot accounts

## Architecture

```
training-data-generator/
├── generate_data.py          # CLI entry point
├── requirements.txt
├── README.md
├── examples/                 # Example configs
├── src/
│   ├── config/              # YAML parsing & validation
│   ├── extractors/          # Web, File, GitHub extractors
│   ├── parsers/             # HTML, Markdown, Code parsers
│   ├── processors/          # Cleaning, validation, dedup
│   ├── generators/          # Parquet, JSONL generators
│   ├── utils/               # Logging, retry, progress
│   └── orchestrator.py      # Job orchestration
└── tests/                   # Unit tests
```

## Data Flow

```
Source (Web/Files/GitHub)
    ↓
Extractor (scrape/read/fetch)
    ↓
Parser (HTML/Markdown/Code)
    ↓
Processor (clean/validate/dedup)
    ↓
Formatter (convert to target format)
    ↓
Generator (write Parquet/JSONL)
    ↓
Output Files
```

## Advanced Usage

### Custom Parsers

You can specify custom parsers for sources:

```yaml
- type: "files"
  paths: ["./docs/**/*.md"]
  options:
    parser: "markdown_qa"  # Extract Q&A from markdown
```

### Environment Variables

Use environment variables in config:

```yaml
auth:
  github_token: "${GITHUB_TOKEN}"
```

### Multiple Jobs

Run multiple jobs in sequence:

```yaml
jobs:
  - name: "pretraining"
    type: "parquet"
    ...

  - name: "finetuning"
    type: "jsonl_conversation"
    ...
```

## Performance Tips

1. **Web Scraping**: Adjust `rate_limit` to avoid getting blocked
2. **Large Repositories**: Use `max_prs_per_repo` to limit PR count
3. **File Scanning**: Use specific glob patterns instead of `**/*`
4. **Deduplication**: Can use significant memory for large datasets

## Troubleshooting

### GitHub Rate Limiting

If you hit GitHub API rate limits:
- Wait for rate limit reset (check response headers)
- Use multiple GitHub tokens (rotate them)
- Reduce `max_prs_per_repo`

### Memory Issues

For large datasets:
- Reduce `shard_size_mb` for Parquet output
- Process files in smaller batches
- Disable deduplication if not needed

### Invalid Output

Check:
- Configuration validation errors
- Log output for warnings
- Generated files with validation tools

## Contributing

Contributions welcome! Areas for improvement:
- Additional parsers (PDF, DOCX)
- More sophisticated content extraction
- Parallel processing support
- Cloud storage integration

## License

MIT License

## Related Projects

- [nanochat](https://github.com/karpathy/nanochat) - The LLM training framework this tool generates data for
- [FineWeb-Edu](https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle) - Example pretraining dataset

## Support

For issues and questions:
- Check the examples in `examples/` directory
- Review logs with `--log-level DEBUG`
- Open an issue on GitHub
