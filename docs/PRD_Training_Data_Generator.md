# Product Requirements Document: Training Data Generator

**Version:** 1.0
**Date:** 2026-01-22
**Status:** Draft
**Author:** Claude

---

## 1. Executive Summary

A Python-based web scraping and data extraction tool designed to generate training datasets for LLM training pipelines in three distinct formats: Parquet (pretraining), JSONL conversations (finetuning), and JSONL code interactions (programming finetuning). The tool will be fully configurable via YAML and support multiple data sources including web URLs, local file systems, and GitHub Pull Requests.

---

## 2. Goals & Objectives

### Primary Goals
1. **Automate training data collection** from diverse sources (web, files, GitHub)
2. **Generate standardized datasets** compatible with nanochat training pipeline
3. **Provide flexible configuration** via YAML for different data collection scenarios
4. **Ensure data quality** through validation and preprocessing

### Success Metrics
- Generate 100+ MB of pretraining data (Parquet) from web sources
- Produce 1,000+ conversation pairs for finetuning
- Extract 500+ PR-based code examples from GitHub
- Configuration time < 5 minutes per new data source
- Data validation pass rate > 95%

---

## 3. User Stories

### Data Scientist/ML Engineer
- **As a** data scientist, **I want to** configure data sources via YAML **so that** I can quickly iterate on different data collection strategies
- **As a** ML engineer, **I want to** generate Parquet files from web articles **so that** I can pretrain language models
- **As a** researcher, **I want to** extract conversations from documentation **so that** I can finetune models on domain-specific knowledge

### Software Developer
- **As a** developer, **I want to** extract code examples from GitHub PRs **so that** I can train code-generation models
- **As a** developer, **I want to** recursively process local codebases **so that** I can create training data from existing projects

---

## 4. Functional Requirements

### 4.1 Core Features

#### FR-1: YAML Configuration System
- **Priority:** P0 (Critical)
- **Description:** Support YAML-based configuration for all data sources and output formats
- **Acceptance Criteria:**
  - Parse YAML configuration files with validation
  - Support multiple jobs in a single config file
  - Provide clear error messages for invalid configurations
  - Allow environment variable substitution (e.g., `${GITHUB_TOKEN}`)

#### FR-2: Web Scraping Module
- **Priority:** P0 (Critical)
- **Description:** Scrape and extract text content from web URLs
- **Acceptance Criteria:**
  - Support HTTP/HTTPS URLs
  - Handle common web formats (HTML, Markdown, plain text)
  - Extract clean text content (strip HTML tags, scripts, styles)
  - Respect robots.txt and rate limiting
  - Support custom headers and authentication
  - Handle pagination (optional)
  - Retry failed requests with exponential backoff

#### FR-3: File System Scanner
- **Priority:** P0 (Critical)
- **Description:** Recursively scan local directories for text data
- **Acceptance Criteria:**
  - Support recursive directory traversal
  - Filter by file extensions (e.g., .txt, .md, .py, .java)
  - Support glob patterns for file matching
  - Handle large files efficiently (streaming)
  - Skip binary files automatically
  - Respect .gitignore patterns (optional)

#### FR-4: GitHub PR Inspector
- **Priority:** P0 (Critical)
- **Description:** Extract code examples from GitHub Pull Requests
- **Acceptance Criteria:**
  - Authenticate via GitHub API token
  - Fetch PR data (title, description, commits, diffs)
  - Parse PR description as "user question"
  - Extract commit diffs as "code solution"
  - Support filtering by repository, date range, labels
  - Handle pagination for repositories with many PRs
  - Extract both the problem context and code changes

#### FR-5: Parquet Output Generator
- **Priority:** P0 (Critical)
- **Description:** Generate Parquet files for pretraining data
- **Acceptance Criteria:**
  - Output schema: `{text: string}`
  - Use ZSTD compression
  - Shard files at configurable size (default: 100MB)
  - Support row group size configuration (default: 1024)
  - Validate data integrity after write

#### FR-6: JSONL Conversation Generator
- **Priority:** P0 (Critical)
- **Description:** Generate JSONL files for conversational finetuning
- **Acceptance Criteria:**
  - Output format: `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`
  - Validate role alternation (user → assistant → user → ...)
  - Support multi-turn conversations
  - Handle empty content validation
  - Support both string and structured content types

#### FR-7: JSONL Code Format Generator
- **Priority:** P0 (Critical)
- **Description:** Generate JSONL files with code execution format
- **Acceptance Criteria:**
  - Support structured assistant content:
    ```json
    {"type": "text", "text": "explanation"},
    {"type": "python", "text": "code"},
    {"type": "python_output", "text": "result"}
    ```
  - Extract PR title/description as user question
  - Extract commit message as explanation
  - Extract code diff as solution
  - Validate code syntax (optional)
  - Support multiple programming languages

#### FR-8: Data Validation & Quality Control
- **Priority:** P1 (High)
- **Description:** Validate and clean extracted data
- **Acceptance Criteria:**
  - Remove duplicate content
  - Validate minimum/maximum text length
  - Check for encoding issues
  - Remove low-quality content (e.g., too many special characters)
  - Generate data quality report
  - Support custom validation rules

#### FR-9: Progress Tracking & Logging
- **Priority:** P1 (High)
- **Description:** Provide visibility into data collection progress
- **Acceptance Criteria:**
  - Display progress bar for long-running operations
  - Log all operations (INFO, WARNING, ERROR levels)
  - Report statistics (items processed, errors, output sizes)
  - Support resume from interruption (checkpointing)
  - Generate summary report at completion

### 4.2 Non-Functional Requirements

#### NFR-1: Performance
- Process 1,000 web pages in < 30 minutes (with rate limiting)
- Handle 100,000 files in local directory scan
- Generate 1GB Parquet file in < 5 minutes

#### NFR-2: Reliability
- Handle network failures gracefully (retry logic)
- Support interrupted job resumption
- Validate all outputs before completion

#### NFR-3: Maintainability
- Modular architecture (pluggable extractors/generators)
- Comprehensive error handling
- Unit test coverage > 80%

#### NFR-4: Usability
- Single command execution: `python generate_data.py --config config.yaml`
- Clear documentation with examples
- Helpful error messages

---

## 5. Configuration Schema

### 5.1 YAML Structure

```yaml
# config.yaml
version: "1.0"

# Global settings
global:
  output_dir: "./output"
  cache_dir: "./.cache"
  log_level: "INFO"
  num_workers: 4  # Parallel processing

# Authentication (optional)
auth:
  github_token: "${GITHUB_TOKEN}"  # Environment variable

# Jobs definition
jobs:
  # Job 1: Pretraining data from web
  - name: "wikipedia_pretraining"
    type: "parquet"
    output_file: "wikipedia_shard_{index:05d}.parquet"

    sources:
      # Web URLs
      - type: "web"
        urls:
          - "https://en.wikipedia.org/wiki/Machine_learning"
          - "https://en.wikipedia.org/wiki/Deep_learning"
        options:
          follow_links: true
          max_depth: 2
          url_pattern: "^https://en.wikipedia.org/wiki/"
          rate_limit: 1.0  # seconds between requests

      # Local files
      - type: "files"
        paths:
          - "./data/articles/**/*.txt"
          - "./data/books/**/*.md"
        options:
          recursive: true
          encoding: "utf-8"

    # Output options
    options:
      shard_size_mb: 100
      compression: "zstd"
      row_group_size: 1024
      min_text_length: 100
      max_text_length: 1000000
      deduplication: true

  # Job 2: Conversational finetuning data
  - name: "qa_conversations"
    type: "jsonl_conversation"
    output_file: "qa_conversations.jsonl"

    sources:
      # Extract Q&A from web pages
      - type: "web"
        urls:
          - "https://stackoverflow.com/questions/tagged/python"
        options:
          parser: "stackoverflow_qa"  # Custom parser
          max_pages: 100

      # Local markdown files as conversations
      - type: "files"
        paths:
          - "./docs/faq/**/*.md"
        options:
          parser: "markdown_qa"  # Parse markdown headers as Q&A

    # Conversation generation options
    options:
      max_turns: 10
      validate_roles: true
      min_content_length: 10
      deduplicate: true

  # Job 3: Code training data from GitHub PRs
  - name: "github_code_examples"
    type: "jsonl_code"
    output_file: "github_code_examples.jsonl"

    sources:
      # GitHub repositories
      - type: "github_pr"
        repositories:
          - "anthropics/anthropic-sdk-python"
          - "karpathy/nanochat"
        options:
          state: "closed"  # open, closed, all
          merged_only: true
          date_from: "2024-01-01"
          date_to: "2026-01-22"
          labels: ["bug", "enhancement"]  # Optional filter
          max_prs_per_repo: 100
          include_diff: true

      # Local git repository
      - type: "git_local"
        paths:
          - "./my_project/.git"
        options:
          branch: "main"
          commit_range: "HEAD~100..HEAD"

    # Code format options
    options:
      languages: ["python", "javascript", "rust"]
      max_diff_lines: 500
      include_context: true  # Include file paths
      parse_commit_message: true  # Extract explanation from commit msg

# Data quality rules (optional)
quality:
  min_text_length: 50
  max_text_length: 100000
  allowed_languages: ["en"]  # Language detection
  exclude_patterns:
    - ".*cookie.*"
    - ".*privacy policy.*"
  content_filters:
    - "remove_urls"
    - "remove_emails"
    - "normalize_whitespace"
```

---

## 6. Technical Architecture

### 6.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Entry Point                         │
│                  (generate_data.py --config)                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Configuration Loader                        │
│                 (YAML parsing, validation)                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Job Orchestrator                         │
│              (Job scheduling, parallel execution)               │
└────┬───────────────────┬────────────────────┬───────────────────┘
     │                   │                    │
     ▼                   ▼                    ▼
┌──────────┐      ┌──────────┐        ┌──────────────┐
│   Web    │      │   File   │        │   GitHub     │
│ Scraper  │      │  Scanner │        │ PR Inspector │
└────┬─────┘      └────┬─────┘        └──────┬───────┘
     │                 │                     │
     └─────────────────┴─────────────────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │   Content Extractors   │
          │  - HTML Parser         │
          │  - Markdown Parser     │
          │  - Code Parser         │
          │  - Custom Parsers      │
          └────────────┬───────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │   Data Processors      │
          │  - Cleaning            │
          │  - Validation          │
          │  - Deduplication       │
          │  - Formatting          │
          └────────────┬───────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │   Output Generators    │
          │  - Parquet Writer      │
          │  - JSONL Writer        │
          │  - Format Validators   │
          └────────────────────────┘
```

### 6.2 Module Structure

```
training-data-generator/
├── generate_data.py              # CLI entry point
├── config.yaml                   # Example configuration
├── requirements.txt              # Dependencies
├── README.md                     # Documentation
│
├── src/
│   ├── __init__.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── loader.py            # YAML config loading
│   │   ├── validator.py         # Config validation
│   │   └── schema.py            # Config schema definitions
│   │
│   ├── extractors/
│   │   ├── __init__.py
│   │   ├── base.py              # Base extractor interface
│   │   ├── web_scraper.py       # Web URL scraping
│   │   ├── file_scanner.py      # Local file scanning
│   │   └── github_pr.py         # GitHub PR extraction
│   │
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── html_parser.py       # HTML content extraction
│   │   ├── markdown_parser.py   # Markdown parsing
│   │   ├── code_parser.py       # Code/diff parsing
│   │   └── qa_parser.py         # Q&A extraction (StackOverflow, etc.)
│   │
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── cleaner.py           # Text cleaning
│   │   ├── validator.py         # Content validation
│   │   ├── deduplicator.py      # Duplicate detection
│   │   └── formatter.py         # Format conversion
│   │
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── base.py              # Base generator interface
│   │   ├── parquet_gen.py       # Parquet file generation
│   │   ├── jsonl_conv_gen.py    # JSONL conversation generation
│   │   └── jsonl_code_gen.py    # JSONL code format generation
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py            # Logging configuration
│   │   ├── progress.py          # Progress tracking
│   │   ├── retry.py             # Retry logic
│   │   └── cache.py             # Caching utilities
│   │
│   └── orchestrator.py          # Job orchestration
│
└── tests/
    ├── __init__.py
    ├── test_extractors.py
    ├── test_parsers.py
    ├── test_generators.py
    └── fixtures/
        ├── sample_config.yaml
        ├── sample_html.html
        └── sample_pr_data.json
```

---

## 7. Detailed Feature Specifications

### 7.1 GitHub PR Inspector

#### Data Extraction Flow
```
GitHub API Call
    ↓
Fetch PR List (filtered by repo, date, labels, state)
    ↓
For each PR:
    ├── Extract PR metadata (title, description, author, date)
    ├── Fetch all commits in PR
    ├── For each commit:
    │   ├── Parse commit message
    │   ├── Fetch file diffs
    │   └── Extract added/modified code
    ↓
Generate JSONL entry:
{
  "messages": [
    {
      "role": "user",
      "content": "<PR_TITLE>\n\n<PR_DESCRIPTION>"
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "<COMMIT_MESSAGE_EXPLANATION>"},
        {"type": "python", "text": "<CODE_CHANGES>"}
      ]
    }
  ]
}
```

#### GitHub API Integration
- Use `PyGithub` library for API access
- Rate limiting: 5,000 requests/hour (authenticated)
- Implement token bucket algorithm for rate limiting
- Support pagination (100 PRs per request)
- Cache API responses to avoid redundant calls

#### PR Filtering Options
```yaml
options:
  state: "closed"           # open, closed, all
  merged_only: true         # Only include merged PRs
  date_from: "2024-01-01"  # ISO date
  date_to: "2026-01-22"    # ISO date
  labels: ["bug", "feature"]  # PR labels
  min_commits: 1            # Minimum commits in PR
  max_commits: 20           # Maximum commits (avoid huge PRs)
  min_changes: 10           # Minimum line changes
  max_changes: 500          # Maximum line changes
  languages: ["python"]     # Filter by file extensions
  exclude_authors: ["dependabot"]  # Skip bot PRs
```

#### Code Diff Parsing
- Extract only meaningful changes (ignore whitespace, comments)
- Preserve context lines (configurable, default: 3)
- Separate additions and deletions
- Group related changes by file
- Support unified diff format

### 7.2 Web Scraper

#### Content Extraction
- Use `BeautifulSoup4` for HTML parsing
- Use `readability-lxml` for article extraction
- Extract main content area (remove nav, footer, ads)
- Preserve paragraph structure
- Handle different encodings (UTF-8, Latin-1, etc.)

#### Link Following
```yaml
options:
  follow_links: true
  max_depth: 2              # BFS depth
  url_pattern: "^https://example.com/docs/"  # Regex filter
  exclude_pattern: "(login|signup|cart)"
  same_domain_only: true
  max_pages: 1000
  rate_limit: 1.0          # Seconds between requests
```

#### Robots.txt Compliance
- Parse and respect robots.txt
- Honor crawl-delay directive
- Skip disallowed paths
- Option to override (for owned sites)

### 7.3 File Scanner

#### File Type Detection
- Use `python-magic` for MIME type detection
- Support text formats: .txt, .md, .rst, .tex
- Support code formats: .py, .js, .java, .cpp, .rs, .go
- Support document formats: .html, .xml, .json
- Skip binary files automatically

#### Glob Pattern Matching
```yaml
paths:
  - "./docs/**/*.md"           # All markdown in docs
  - "./src/**/*.{py,js}"       # Python and JS files
  - "!./src/tests/**"          # Exclude tests
```

#### Large File Handling
- Stream files > 10MB (don't load into memory)
- Process files in chunks (default: 1MB chunks)
- Skip files > configurable limit (default: 100MB)

---

## 8. Output Format Specifications

### 8.1 Parquet Format (Pretraining)

**Schema:**
```python
{
    "text": pa.string()  # Raw text content
}
```

**File naming:** `{job_name}_shard_{index:05d}.parquet`

**Example:**
```
wikipedia_pretraining_shard_00000.parquet
wikipedia_pretraining_shard_00001.parquet
```

**Compression:** ZSTD (level 3)

**Row Group Size:** 1024 documents

**Validation:**
- Each row must have non-empty text field
- Text length: 100 ≤ len ≤ 1,000,000 characters
- No duplicate documents (hash-based)

### 8.2 JSONL Conversation Format (Finetuning)

**Format:** One conversation per line

```json
[
  {"role": "user", "content": "What is machine learning?"},
  {"role": "assistant", "content": "Machine learning is a subset of AI..."}
]
```

**Multi-turn example:**
```json
[
  {"role": "user", "content": "What is Python?"},
  {"role": "assistant", "content": "Python is a programming language..."},
  {"role": "user", "content": "What can I use it for?"},
  {"role": "assistant", "content": "You can use Python for web development..."}
]
```

**Validation:**
- Must start with user message
- Roles must alternate (user → assistant → user → ...)
- Content must be non-empty strings
- Maximum 20 turns per conversation

### 8.3 JSONL Code Format (Code Finetuning)

**Format:** Structured assistant content

```json
[
  {
    "role": "user",
    "content": "Fix the bug where the function returns None instead of 0"
  },
  {
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "The issue is in the return statement. We need to return 0 when the list is empty."
      },
      {
        "type": "python",
        "text": "def sum_list(items):\n    if not items:\n        return 0  # Fixed: was 'return None'\n    return sum(items)"
      }
    ]
  }
]
```

**Supported content types:**
- `text`: Explanation or description
- `python`: Python code
- `javascript`: JavaScript code
- `rust`: Rust code
- `output`: Execution output (optional)

**Validation:**
- User content must be string
- Assistant content can be string or array of parts
- Each part must have `type` and `text` fields
- Code syntax validation (optional, per language)

---

## 9. Error Handling & Edge Cases

### 9.1 Network Errors
- **HTTP 429 (Rate Limited)**: Exponential backoff, respect Retry-After header
- **HTTP 404**: Log warning, skip URL, continue
- **HTTP 500-599**: Retry up to 3 times with backoff
- **Connection timeout**: Retry with increased timeout (10s → 30s → 60s)
- **DNS errors**: Skip URL, log error

### 9.2 File System Errors
- **Permission denied**: Log error, skip file
- **File not found**: Log warning, skip
- **Encoding errors**: Try fallback encodings (UTF-8 → Latin-1 → ASCII)
- **Corrupted files**: Log error, skip
- **Disk space**: Check before write, fail gracefully

### 9.3 GitHub API Errors
- **Rate limit exceeded**: Wait until reset time, show countdown
- **Authentication failed**: Clear error message, check token
- **Repository not found**: Skip, log error
- **API changes**: Graceful degradation, log warnings

### 9.4 Data Quality Issues
- **Empty content**: Skip, log warning
- **Too short (<50 chars)**: Skip or log based on config
- **Too long (>1MB)**: Truncate or skip based on config
- **Invalid encoding**: Attempt repair, skip if unfixable
- **Duplicate detection**: Hash-based deduplication, log duplicates

---

## 10. Success Criteria

### 10.1 Functional Success
- [ ] Generate valid Parquet files readable by PyArrow
- [ ] Generate valid JSONL files parsable by nanochat tasks
- [ ] Extract 100+ GitHub PRs with valid code examples
- [ ] Process 1,000+ web pages without crashes
- [ ] Handle all specified file formats correctly

### 10.2 Quality Success
- [ ] 95%+ data validation pass rate
- [ ] < 5% duplicate content in output
- [ ] Extracted text quality: readable, properly formatted
- [ ] Code examples: syntactically valid, complete context

### 10.3 Performance Success
- [ ] Process 100 web pages in < 5 minutes (1 req/sec rate limit)
- [ ] Scan 10,000 files in < 2 minutes
- [ ] Generate 100MB Parquet file in < 3 minutes
- [ ] Memory usage < 1GB for typical workloads

### 10.4 Usability Success
- [ ] Configure new job in < 5 minutes
- [ ] Clear error messages for all failures
- [ ] Progress visibility for long-running jobs
- [ ] Resume capability for interrupted jobs

---

## 11. Out of Scope (Future Enhancements)

### Phase 2 Features
- **Distributed scraping**: Multi-machine coordination
- **Real-time streaming**: Continuous data collection
- **Advanced parsers**: PDF, DOCX, LaTeX extraction
- **Data augmentation**: Paraphrasing, back-translation
- **Quality scoring**: ML-based quality prediction
- **Interactive mode**: Human-in-the-loop curation
- **Cloud storage**: S3/GCS direct upload
- **Scheduling**: Cron-based periodic runs

### Not Planned
- GUI interface (CLI only)
- Real-time data streaming from APIs (batch only)
- Video/audio transcription (text only)
- OCR for images (text sources only)

---

## 12. Dependencies & Technology Stack

### Core Libraries
```txt
# Web scraping
requests>=2.31.0
beautifulsoup4>=4.12.0
readability-lxml>=0.8.1
urllib3>=2.0.0

# GitHub API
PyGithub>=2.1.1
gitpython>=3.1.40

# Data processing
pyarrow>=14.0.0          # Parquet I/O
pandas>=2.1.0            # Data manipulation
pyyaml>=6.0.1            # Config parsing

# Utilities
tqdm>=4.66.0             # Progress bars
python-magic>=0.4.27     # File type detection
chardet>=5.2.0           # Encoding detection
regex>=2023.10.0         # Advanced regex
xxhash>=3.4.1            # Fast hashing for dedup

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
responses>=0.24.0        # Mock HTTP responses
```

### Python Version
- **Minimum:** Python 3.9
- **Recommended:** Python 3.11+

---

## 13. Testing Strategy

### Unit Tests
- Config parsing and validation
- Each extractor independently
- Each parser independently
- Data validators
- Output generators

### Integration Tests
- End-to-end job execution
- Multi-source jobs
- Error recovery scenarios
- Output validation

### Test Data
- Mock web pages (HTML fixtures)
- Sample GitHub PR data (JSON fixtures)
- Test file system structure
- Edge cases (empty, malformed, huge files)

### Coverage Target
- Line coverage: > 80%
- Branch coverage: > 70%

---

## 14. Documentation Requirements

### User Documentation
- **README.md**: Quick start, installation, basic usage
- **Configuration Guide**: Complete YAML reference
- **Examples**: 5+ example configurations for common use cases
- **Troubleshooting**: Common errors and solutions

### Developer Documentation
- **Architecture Overview**: Component diagram, data flow
- **API Reference**: All public classes and functions
- **Contributing Guide**: How to add new extractors/parsers
- **Testing Guide**: How to run tests, add test cases

---

## 15. Timeline & Milestones

### Phase 1: Core Infrastructure (Week 1-2)
- Config loader and validator
- Base extractor/generator interfaces
- Logging and progress tracking
- Basic CLI

### Phase 2: Extractors (Week 3-4)
- Web scraper with rate limiting
- File scanner with glob support
- GitHub PR inspector
- Content parsers (HTML, Markdown, code)

### Phase 3: Generators (Week 5)
- Parquet generator with sharding
- JSONL conversation generator
- JSONL code format generator
- Data validation pipeline

### Phase 4: Testing & Polish (Week 6)
- Unit and integration tests
- Documentation
- Example configurations
- Bug fixes and optimization

### Phase 5: Beta Testing (Week 7)
- Real-world data collection
- Performance tuning
- User feedback integration
- Release preparation

---

## 16. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| GitHub API rate limits | High | Medium | Implement caching, support multiple tokens |
| Website blocking scrapers | Medium | Medium | Respect robots.txt, add rate limiting, user-agent rotation |
| Large file memory issues | Medium | High | Streaming processing, chunk-based reading |
| Invalid output formats | Low | High | Comprehensive validation, extensive testing |
| PR data quality (spam, irrelevant) | Medium | Medium | Filtering options, quality validation rules |
| Configuration complexity | Medium | Low | Examples, clear documentation, validation errors |

---

## 17. Appendix

### A. Example Use Cases

#### Use Case 1: Python Documentation Dataset
```yaml
jobs:
  - name: "python_docs"
    type: "parquet"
    sources:
      - type: "web"
        urls: ["https://docs.python.org/3/"]
        options:
          follow_links: true
          url_pattern: "^https://docs.python.org/3/"
          max_depth: 3
```

#### Use Case 2: StackOverflow Q&A
```yaml
jobs:
  - name: "stackoverflow_qa"
    type: "jsonl_conversation"
    sources:
      - type: "web"
        urls: ["https://stackoverflow.com/questions/tagged/python"]
        options:
          parser: "stackoverflow_qa"
          max_pages: 500
```

#### Use Case 3: Open Source Code Examples
```yaml
jobs:
  - name: "oss_code_examples"
    type: "jsonl_code"
    sources:
      - type: "github_pr"
        repositories:
          - "pallets/flask"
          - "django/django"
          - "fastapi/fastapi"
        options:
          merged_only: true
          date_from: "2024-01-01"
          max_prs_per_repo: 200
```

### B. Performance Benchmarks (Target)

| Operation | Volume | Time | Rate |
|-----------|--------|------|------|
| Web scraping | 1,000 pages | 20 min | 50 pages/min |
| File scanning | 10,000 files | 2 min | 5,000 files/min |
| GitHub PR extraction | 100 PRs | 5 min | 20 PRs/min |
| Parquet generation | 1 GB | 5 min | 200 MB/min |
| JSONL generation | 10,000 convs | 1 min | 10,000 convs/min |

### C. Configuration Validation Rules

1. **Required fields:** version, jobs
2. **Job requirements:** name, type, output_file, sources
3. **Source requirements:** type, appropriate fields per type
4. **Type validation:** Enum values for type, state, etc.
5. **Range validation:** Numeric values (depths, limits, etc.)
6. **Pattern validation:** URL patterns, file globs
7. **Dependency validation:** GitHub token required for github_pr sources

---

## 18. Approval & Sign-off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Manager | | | |
| Tech Lead | | | |
| Engineering | | | |
| QA Lead | | | |

---

**Document History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-22 | Claude | Initial draft |

