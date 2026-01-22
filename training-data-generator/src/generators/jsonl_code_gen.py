"""JSONL code format generator."""

import json
from typing import List

from .base import BaseGenerator
from ..config.schema import CodeOutputOptions
from ..processors.validator import ConversationValidator


class JSONLCodeGenerator(BaseGenerator):
    """Generator for JSONL code format files."""

    def __init__(self, output_path: str, options: CodeOutputOptions = None, logger=None):
        """Initialize JSONL code generator.

        Args:
            output_path: Path for output file
            options: Code generation options
            logger: Optional logger
        """
        super().__init__(output_path, options, logger)
        self.options = options or CodeOutputOptions()

        # Buffer for conversations
        self.conversations = []
        self.validator = ConversationValidator()

    def add(self, item: dict):
        """Add a code conversation to be written.

        Args:
            item: Conversation dictionary with 'messages' key
                  Assistant content can be string or list of parts
        """
        # Validate conversation structure
        if not self.validator.is_valid_conversation(item):
            self.log('warning', 'Invalid conversation structure, skipping')
            return

        messages = item.get('messages', [])

        # Filter by language if specified
        if self.options.languages:
            has_target_language = False
            for msg in messages:
                if msg['role'] == 'assistant':
                    content = msg['content']
                    if isinstance(content, list):
                        for part in content:
                            if part.get('type') in self.options.languages:
                                has_target_language = True
                                break

            if not has_target_language:
                self.log('debug', 'No target language found, skipping')
                return

        # Add to buffer
        self.conversations.append(messages)

    def finalize(self) -> List[str]:
        """Write all conversations to JSONL file.

        Returns:
            List with single output file path
        """
        if not self.conversations:
            self.log('warning', 'No code conversations to write')
            return []

        self.log('info', f"Writing {len(self.conversations)} code conversations to {self.output_path}")

        # Write JSONL file
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for conversation in self.conversations:
                json.dump(conversation, f, ensure_ascii=False)
                f.write('\n')

        return [str(self.output_path)]
