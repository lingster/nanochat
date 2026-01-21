# Dataset Curation Guide for nanochat

This guide explains how to curate and format custom datasets for training the nanochat LLM. It covers both pretraining data (raw text) and finetuning data (conversational format).

## Table of Contents

1. [Overview](#overview)
2. [Pretraining Data Format](#pretraining-data-format)
3. [Finetuning Data Format](#finetuning-data-format)
4. [Special Tokens](#special-tokens)
5. [Creating Custom Tasks](#creating-custom-tasks)
6. [Data Quality Guidelines](#data-quality-guidelines)
7. [Examples](#examples)

---

## Overview

nanochat uses different data formats for different training stages:

| Stage | Format | Purpose |
|-------|--------|---------|
| **Pretraining** | Parquet with text column | Build language understanding |
| **Midtraining** | Conversation JSON | Learn conversation structure |
| **SFT** | Conversation JSON | Learn specific tasks |
| **Custom JSONL** | JSONL with message arrays | Identity/synthetic data |

---

## Pretraining Data Format

Pretraining uses Apache Parquet files with zstd compression containing a single `text` column.

### Structure

```
my_dataset/
├── shard_00000.parquet
├── shard_00001.parquet
├── shard_00002.parquet
└── ...
```

### Parquet Schema

```python
schema = pa.schema([
    ('text', pa.string())
])
```

### Creating Pretraining Data

```python
import pyarrow as pa
import pyarrow.parquet as pq

def create_pretraining_shard(documents: list[str], output_path: str):
    """
    Create a pretraining data shard from a list of documents.

    Args:
        documents: List of text documents (each document is a string)
        output_path: Path to save the parquet file
    """
    table = pa.table({'text': documents})

    pq.write_table(
        table,
        output_path,
        compression='zstd',
        compression_level=3,
        row_group_size=1024  # Recommended: 1024 documents per row group
    )

# Example usage
documents = [
    "This is the first document about Python programming...",
    "Another document covering machine learning concepts...",
    "Documentation for the requests library..."
]

create_pretraining_shard(documents, "shard_00000.parquet")
```

### Guidelines for Pretraining Data

1. **Document Size**: Cap documents at ~10,000 characters (prevents tokenizer skew)
2. **Shard Size**: Aim for ~250M characters per shard (~100MB compressed)
3. **Diversity**: Include varied topics, styles, and domains
4. **Quality**: Filter out low-quality content, duplicates, and garbage text
5. **Shuffling**: Shuffle documents before sharding for better training dynamics

### Recommended Shard Creation Script

```python
import os
import random
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

def create_pretraining_dataset(
    documents: list[str],
    output_dir: str,
    chars_per_shard: int = 250_000_000,
    doc_cap: int = 10_000,
    seed: int = 42
):
    """
    Create a complete pretraining dataset from documents.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Shuffle documents
    random.seed(seed)
    random.shuffle(documents)

    # Cap document lengths
    documents = [doc[:doc_cap] for doc in documents]

    # Shard documents
    current_shard = []
    current_chars = 0
    shard_idx = 0

    for doc in documents:
        doc_chars = len(doc)

        if current_chars + doc_chars > chars_per_shard and current_shard:
            # Write current shard
            output_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.parquet")
            table = pa.table({'text': current_shard})
            pq.write_table(table, output_path, compression='zstd',
                          compression_level=3, row_group_size=1024)

            print(f"Created {output_path} with {len(current_shard)} documents")

            current_shard = []
            current_chars = 0
            shard_idx += 1

        current_shard.append(doc)
        current_chars += doc_chars

    # Write final shard
    if current_shard:
        output_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.parquet")
        table = pa.table({'text': current_shard})
        pq.write_table(table, output_path, compression='zstd',
                      compression_level=3, row_group_size=1024)
        print(f"Created {output_path} with {len(current_shard)} documents")
```

---

## Finetuning Data Format

Finetuning uses a conversation-based format with structured messages.

### Basic Conversation Structure

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: str | list[dict]

@dataclass
class Conversation:
    messages: list[Message]
```

### Simple Conversation (JSON)

```json
{
  "messages": [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a high-level programming language..."}
  ]
}
```

### Multi-turn Conversation

```json
{
  "messages": [
    {"role": "user", "content": "Explain recursion"},
    {"role": "assistant", "content": "Recursion is when a function calls itself..."},
    {"role": "user", "content": "Can you show an example?"},
    {"role": "assistant", "content": "Here's a factorial function:\n\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"}
  ]
}
```

### Complex Content with Tool Calls

For tasks involving code execution or tool use:

```json
{
  "messages": [
    {"role": "user", "content": "Calculate 15% of 200"},
    {"role": "assistant", "content": [
      {"type": "text", "text": "I'll calculate that for you."},
      {"type": "python", "text": "200 * 0.15"},
      {"type": "python_output", "text": "30.0"},
      {"type": "text", "text": "15% of 200 is 30."}
    ]}
  ]
}
```

### Content Part Types

| Type | Description | Mask |
|------|-------------|------|
| `text` | Plain text response | 1 (train) |
| `python` | Python code to execute | 1 (train) |
| `python_output` | Execution result | 0 (context only) |

---

## Special Tokens

The nanochat tokenizer includes special tokens for conversation structure:

| Token | ID | Purpose |
|-------|-----|---------|
| `<\|bos\|>` | 0 | Beginning of sequence/document |
| `<\|user_start\|>` | 1 | Start of user message |
| `<\|user_end\|>` | 2 | End of user message |
| `<\|assistant_start\|>` | 3 | Start of assistant message |
| `<\|assistant_end\|>` | 4 | End of assistant message |
| `<\|python_start\|>` | 5 | Start of Python code block |
| `<\|python_end\|>` | 6 | End of Python code block |
| `<\|output_start\|>` | 7 | Start of tool output |
| `<\|output_end\|>` | 8 | End of tool output |

### Rendered Conversation Example

A conversation like:
```json
{"messages": [
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]}
```

Gets rendered as:
```
<|bos|><|user_start|>Hello<|user_end|><|assistant_start|>Hi there!<|assistant_end|>
```

---

## Creating Custom Tasks

To add a new finetuning task, create a Python file in `tasks/`:

### Task Template

```python
# tasks/my_custom_task.py

from dataclasses import dataclass
from datasets import load_dataset  # or use your own data source
from tasks.common import Task, Conversation, Message

@dataclass
class MyCustomTask(Task):
    """Description of your custom task."""

    def get_train(self) -> list[Conversation]:
        """Return training conversations."""
        conversations = []

        # Load your data
        for example in self._load_data("train"):
            conv = Conversation(messages=[
                Message(role="user", content=example["question"]),
                Message(role="assistant", content=example["answer"])
            ])
            conversations.append(conv)

        return conversations

    def get_test(self) -> list[Conversation]:
        """Return test conversations."""
        conversations = []

        for example in self._load_data("test"):
            conv = Conversation(messages=[
                Message(role="user", content=example["question"]),
                Message(role="assistant", content=example["answer"])
            ])
            conversations.append(conv)

        return conversations

    def _load_data(self, split: str):
        """Load data from your source."""
        # Option 1: HuggingFace dataset
        return load_dataset("my-dataset", split=split)

        # Option 2: Local JSONL file
        # with open(f"data/{split}.jsonl") as f:
        #     return [json.loads(line) for line in f]
```

### Custom JSONL Format

For quick custom data, use the JSONL format compatible with `tasks/customjson.py`:

**File: `~/.cache/nanochat/my_data.jsonl`**
```jsonl
[{"role":"user","content":"Question 1"},{"role":"assistant","content":"Answer 1"}]
[{"role":"user","content":"Question 2"},{"role":"assistant","content":"Answer 2"}]
```

Each line is a JSON array of messages representing one conversation.

### Loading Custom JSONL

```python
from tasks.customjson import CustomJson

# In your training script
task = CustomJson(filepath="/path/to/my_data.jsonl")
train_data = task.get_train()
```

---

## Data Quality Guidelines

### Content Quality

1. **Accuracy**: Ensure factual correctness
2. **Completeness**: Full responses, not truncated
3. **Formatting**: Proper code formatting, markdown where appropriate
4. **Language**: Clear, professional language

### Structural Quality

1. **Role Alternation**: Messages must alternate user → assistant → user → ...
2. **Minimum Messages**: At least 2 messages per conversation
3. **Content Presence**: No empty messages
4. **Valid JSON**: Proper escaping of special characters

### Validation Script

```python
import json
from typing import List, Dict

def validate_conversation(conv: Dict) -> List[str]:
    """Validate a conversation and return list of errors."""
    errors = []

    if "messages" not in conv:
        return ["Missing 'messages' key"]

    messages = conv["messages"]

    if len(messages) < 2:
        errors.append("Conversation must have at least 2 messages")

    expected_role = "user"
    for i, msg in enumerate(messages):
        if "role" not in msg:
            errors.append(f"Message {i}: missing 'role'")
            continue
        if "content" not in msg:
            errors.append(f"Message {i}: missing 'content'")
            continue

        if msg["role"] != expected_role:
            errors.append(f"Message {i}: expected role '{expected_role}', got '{msg['role']}'")

        if not msg["content"]:
            errors.append(f"Message {i}: empty content")

        expected_role = "assistant" if expected_role == "user" else "user"

    if messages and messages[-1]["role"] != "assistant":
        errors.append("Last message must be from assistant")

    return errors

def validate_jsonl_file(filepath: str):
    """Validate a JSONL file of conversations."""
    with open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            try:
                messages = json.loads(line)
                conv = {"messages": [{"role": m["role"], "content": m["content"]}
                                    for m in messages]}
                errors = validate_conversation(conv)
                if errors:
                    print(f"Line {line_num}: {errors}")
            except json.JSONDecodeError as e:
                print(f"Line {line_num}: Invalid JSON - {e}")
```

---

## Examples

### Example 1: Code Documentation Dataset

```python
def create_code_doc_conversation(code: str, docstring: str) -> dict:
    """Create a conversation for code documentation."""
    return {
        "messages": [
            {
                "role": "user",
                "content": f"Write a docstring for this Python function:\n\n```python\n{code}\n```"
            },
            {
                "role": "assistant",
                "content": f"Here's a docstring for that function:\n\n```python\n{docstring}\n```"
            }
        ]
    }
```

### Example 2: Code Review Dataset

```python
def create_code_review_conversation(code: str, issues: list[str], fixed_code: str) -> dict:
    """Create a code review conversation."""
    issues_text = "\n".join(f"- {issue}" for issue in issues)

    return {
        "messages": [
            {
                "role": "user",
                "content": f"Review this code and suggest improvements:\n\n```python\n{code}\n```"
            },
            {
                "role": "assistant",
                "content": f"I found the following issues:\n\n{issues_text}\n\nHere's the improved code:\n\n```python\n{fixed_code}\n```"
            }
        ]
    }
```

### Example 3: PR Description Dataset

```python
def create_pr_dataset_entry(
    files_before: dict[str, str],
    files_after: dict[str, str],
    pr_title: str,
    pr_description: str
) -> dict:
    """Create a PR understanding dataset entry."""

    # Format file changes
    changes = []
    all_files = set(files_before.keys()) | set(files_after.keys())

    for filepath in sorted(all_files):
        before = files_before.get(filepath, "")
        after = files_after.get(filepath, "")

        if not before:
            changes.append(f"### Added: {filepath}\n```\n{after}\n```")
        elif not after:
            changes.append(f"### Deleted: {filepath}")
        elif before != after:
            changes.append(f"### Modified: {filepath}\n**Before:**\n```\n{before}\n```\n**After:**\n```\n{after}\n```")

    changes_text = "\n\n".join(changes)

    return {
        "messages": [
            {
                "role": "user",
                "content": f"Explain what this PR does:\n\n{changes_text}"
            },
            {
                "role": "assistant",
                "content": f"## {pr_title}\n\n{pr_description}"
            }
        ]
    }
```

---

## Integration with Training

Once your dataset is prepared, integrate it with training:

### For Pretraining

1. Place parquet shards in `~/.cache/nanochat/base_data/`
2. Update `nanochat/dataset.py` to point to your data
3. Run tokenizer training: `python scripts/tok_train.py`
4. Run pretraining: `python scripts/base_train.py`

### For Finetuning

1. Create a task class in `tasks/`
2. Add your task to the task mixture in the training script
3. Run SFT: `python scripts/chat_sft.py`

### Quick Custom Data Path

```python
# scripts/chat_sft.py - add your task to the mixture

from tasks.customjson import CustomJson
from tasks.common import TaskMixture

# Load your custom data
custom_task = CustomJson(filepath="/path/to/your/data.jsonl")

# Mix with other tasks
mixture = TaskMixture([
    custom_task,
    # ... other tasks
])
```

---

## Troubleshooting

### Common Issues

1. **"Role alternation error"**: Messages don't alternate user/assistant
2. **"Empty content"**: A message has empty or whitespace-only content
3. **"Invalid JSON"**: Check for unescaped quotes or special characters
4. **"Memory error during training"**: Reduce document/conversation length

### Debugging Tips

```python
# Test tokenization of your data
from nanochat.tokenizer import Tokenizer

tok = Tokenizer()
conv = {"messages": [
    {"role": "user", "content": "Test"},
    {"role": "assistant", "content": "Response"}
]}

ids, mask = tok.render_conversation(conv["messages"])
print(f"Token count: {len(ids)}")
print(f"Trainable tokens: {sum(mask)}")
print(f"Decoded: {tok.decode(ids)}")
```

---

## Next Steps

- See `dev/gen_synthetic_data.py` for generating synthetic conversations
- See `tasks/` directory for more task implementation examples
- See `runs/speedrun.sh` for complete training pipeline
