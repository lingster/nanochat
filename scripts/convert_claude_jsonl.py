#!/usr/bin/env python
"""Convert Claude CLI session logs (JSONL) into nanochat conversation records."""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import typer
except ModuleNotFoundError as exc:  # pragma: no cover - typed at import time
    raise SystemExit(
        "Typer is required for this utility. Install it with 'uv pip install typer' "
        "or 'pip install typer'."
    ) from exc

app = typer.Typer(help="Convert Claude CLI JSONL transcripts into nanochat-friendly conversations.")

Message = Dict[str, str]
Conversation = Dict[str, object]


def iter_jsonl_lines(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                typer.echo(f"[WARN] {path}: failed to parse line ({exc}); skipping", err=True)
                continue


def extract_text(message: Dict[str, object]) -> Optional[str]:
    content = message.get("content")
    if content is None:
        return None
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        text = "\n\n".join(chunks)
    else:
        return None
    text = text.strip()
    return text or None


def harvest_conversations(path: Path, skip_sidechains: bool = True, min_messages: int = 2) -> List[Conversation]:
    conversations: List[Conversation] = []
    current: List[Message] = []
    metadata: Dict[str, object] = {}

    for obj in iter_jsonl_lines(path):
        if skip_sidechains and obj.get("isSidechain"):
            continue

        message = obj.get("message")
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            continue

        text = extract_text(message)
        if text is None:
            continue

        parent_uuid = obj.get("parentUuid")
        if parent_uuid is None and current:
            if len(current) >= min_messages:
                conversations.append({
                    "messages": current,
                    "metadata": metadata.copy(),
                })
            current = []

        if not metadata:
            metadata = {
                "source_path": str(path),
                "session_id": obj.get("sessionId"),
                "git_branch": obj.get("gitBranch"),
            }

        current.append({"role": role, "content": text})

    if current and len(current) >= min_messages:
        conversations.append({
            "messages": current,
            "metadata": metadata.copy(),
        })

    return conversations


def collect_files(input_path: Path, recursive: bool) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    pattern = "**/*.jsonl" if recursive else "*.jsonl"
    return sorted(p for p in input_path.glob(pattern) if p.is_file())


@app.command()
def convert(
    input_path: Path = typer.Argument(..., help="JSONL file or directory to process."),
    output_path: Path = typer.Argument(..., help="Destination JSONL file for converted conversations."),
    skip_sidechains: bool = typer.Option(True, help="Ignore messages marked as sidechains."),
    recursive: bool = typer.Option(True, help="Recurse into subdirectories when input_path is a directory."),
    min_messages: int = typer.Option(2, help="Minimum number of messages required to keep a conversation."),
) -> None:
    """Convert Claude session logs into nanochat-style conversation JSONL."""
    files = collect_files(input_path.expanduser().resolve(), recursive)
    if not files:
        raise typer.BadParameter(f"No JSONL files found under {input_path}")

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_conversations = 0
    with output_path.open("w", encoding="utf-8") as writer:
        for path in files:
            conversations = harvest_conversations(path, skip_sidechains=skip_sidechains, min_messages=min_messages)
            for convo in conversations:
                writer.write(json.dumps(convo, ensure_ascii=False) + "\n")
            total_conversations += len(conversations)
            typer.echo(f"Processed {path} -> {len(conversations)} conversations")

    typer.echo(f"Done. Conversations written: {total_conversations}. Output: {output_path}")


if __name__ == "__main__":
    app()
