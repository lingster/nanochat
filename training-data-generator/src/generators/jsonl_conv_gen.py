"""JSONL conversation format generator."""

import json
from typing import List

from .base import BaseGenerator
from ..config.schema import ConversationOutputOptions
from ..processors.validator import ConversationValidator


class JSONLConversationGenerator(BaseGenerator):
    """Generator for JSONL conversation files."""

    def __init__(self, output_path: str, options: ConversationOutputOptions = None, logger=None):
        """Initialize JSONL conversation generator.

        Args:
            output_path: Path for output file
            options: Conversation generation options
            logger: Optional logger
        """
        super().__init__(output_path, options, logger)
        self.options = options or ConversationOutputOptions()

        # Buffer for conversations
        self.conversations = []
        self.validator = ConversationValidator()

    def add(self, item: dict):
        """Add a conversation to be written.

        Args:
            item: Conversation dictionary with 'messages' key
        """
        # Validate conversation structure
        if self.options.validate_roles:
            if not self.validator.is_valid_conversation(item):
                self.log('warning', 'Invalid conversation structure, skipping')
                return

        # Check turn count
        messages = item.get('messages', [])
        if len(messages) > self.options.max_turns * 2:  # *2 because each turn is user+assistant
            self.log('debug', f"Too many turns ({len(messages)}), skipping")
            return

        # Check content length
        for msg in messages:
            content = msg.get('content', '')
            if isinstance(content, str):
                if len(content) < self.options.min_content_length:
                    self.log('debug', 'Message content too short, skipping conversation')
                    return

        # Add to buffer
        self.conversations.append(item['messages'])

    def finalize(self) -> List[str]:
        """Write all conversations to JSONL file.

        Returns:
            List with single output file path
        """
        if not self.conversations:
            self.log('warning', 'No conversations to write')
            return []

        self.log('info', f"Writing {len(self.conversations)} conversations to {self.output_path}")

        # Write JSONL file
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for conversation in self.conversations:
                json.dump(conversation, f, ensure_ascii=False)
                f.write('\n')

        return [str(self.output_path)]
