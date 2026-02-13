"""JSONL conversation format generator."""

import json
from typing import List, Optional

from .base import BaseGenerator
from ..config.schema import ConversationOutputOptions
from ..processors.validator import ConversationValidator


class JSONLConversationGenerator(BaseGenerator):
    """Generator for JSONL conversation files with incremental writing."""

    DEFAULT_BUFFER_SIZE = 100

    def __init__(self, output_path: str, options: ConversationOutputOptions = None, logger=None, buffer_size: int = None):
        """Initialize JSONL conversation generator.

        Args:
            output_path: Path for output file
            options: Conversation generation options
            logger: Optional logger
            buffer_size: Number of conversations to buffer before flushing (default: 100)
        """
        super().__init__(output_path, options, logger)
        self.options = options or ConversationOutputOptions()
        self.buffer_size = buffer_size or self.DEFAULT_BUFFER_SIZE

        # Buffer for conversations
        self.conversations = []
        self.validator = ConversationValidator()

        # File handle for incremental writing
        self._file_handle: Optional[object] = None
        self._total_written = 0

        # Open file for writing
        self._open_file()

    def _open_file(self):
        """Open output file for incremental writing."""
        try:
            self._file_handle = open(self.output_path, 'w', encoding='utf-8')
            self.log('info', f"Opened file for incremental writing: {self.output_path}")
        except IOError as e:
            self.log('error', f"Failed to open file {self.output_path}: {e}")
            raise

    def _flush(self):
        """Flush buffered conversations to disk."""
        if not self.conversations:
            return

        if not self._file_handle:
            self.log('error', 'File handle is not open, cannot flush')
            return

        try:
            for conversation in self.conversations:
                json.dump(conversation, self._file_handle, ensure_ascii=False)
                self._file_handle.write('\n')

            # Ensure data is written to disk
            self._file_handle.flush()

            # Update stats
            num_flushed = len(self.conversations)
            self._total_written += num_flushed
            self.log('debug', f"Flushed {num_flushed} conversations (total: {self._total_written})")

            # Clear buffer
            self.conversations = []

        except IOError as e:
            self.log('error', f"Failed to flush conversations: {e}")
            raise

    def add(self, item: dict):
        """Add a conversation to be written.

        Conversations are buffered and flushed to disk incrementally.

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

        # Flush if buffer is full
        if len(self.conversations) >= self.buffer_size:
            self._flush()

    def finalize(self) -> List[str]:
        """Flush remaining conversations and close file.

        Returns:
            List with single output file path
        """
        try:
            # Flush any remaining conversations
            if self.conversations:
                self.log('info', f"Flushing final {len(self.conversations)} conversations")
                self._flush()

            # Close file handle
            if self._file_handle:
                self._file_handle.close()
                self._file_handle = None

            if self._total_written == 0:
                self.log('warning', 'No conversations were written')
                return []

            self.log('info', f"Finalized {self._total_written} total conversations to {self.output_path}")
            return [str(self.output_path)]

        except Exception as e:
            self.log('error', f"Error during finalization: {e}")
            # Ensure file is closed even on error
            if self._file_handle:
                try:
                    self._file_handle.close()
                except:
                    pass
                self._file_handle = None
            raise

    def __del__(self):
        """Cleanup: ensure file handle is closed."""
        if self._file_handle:
            try:
                self._file_handle.close()
            except:
                pass
