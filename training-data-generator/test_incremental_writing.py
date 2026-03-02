#!/usr/bin/env python3
"""Test script for incremental writing in JSONL generators.

This script verifies that:
1. Files start appearing within seconds of job start
2. Files grow incrementally as items are added
3. Final output is correct
"""

import json
import logging
import os
import time
from pathlib import Path
from src.generators.jsonl_code_gen import JSONLCodeGenerator
from src.generators.jsonl_conv_gen import JSONLConversationGenerator
from src.config.schema import CodeOutputOptions, ConversationOutputOptions


def setup_logging():
    """Set up logging."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def test_code_generator_incremental_writing(logger):
    """Test JSONLCodeGenerator incremental writing."""
    logger.info("=" * 60)
    logger.info("Testing JSONLCodeGenerator incremental writing")
    logger.info("=" * 60)

    output_path = "/tmp/test_code_incremental.jsonl"

    # Clean up previous test file
    if os.path.exists(output_path):
        os.remove(output_path)

    # Create generator with small buffer size for testing
    generator = JSONLCodeGenerator(
        output_path=output_path,
        options=CodeOutputOptions(),
        logger=logger,
        buffer_size=5  # Small buffer for quick flushing
    )

    # Check that file was created immediately
    assert os.path.exists(output_path), "Output file should be created immediately"
    initial_size = os.path.getsize(output_path)
    logger.info(f"Initial file size: {initial_size} bytes")

    # Add test conversations
    num_conversations = 23  # Not a multiple of buffer_size to test finalization
    for i in range(num_conversations):
        conversation = {
            'messages': [
                {'role': 'user', 'content': f'Question {i}'},
                {'role': 'assistant', 'content': [
                    {'type': 'python', 'value': f'def test_{i}():\n    return {i}'}
                ]}
            ]
        }
        generator.add(conversation)

        # Check file size grows after buffer flushes
        if (i + 1) % 5 == 0:
            time.sleep(0.1)  # Give time for flush
            current_size = os.path.getsize(output_path)
            logger.info(f"After {i+1} items: {current_size} bytes")
            assert current_size > initial_size, f"File should grow after flush at item {i+1}"
            initial_size = current_size

    # Finalize
    output_files = generator.finalize()
    assert len(output_files) == 1
    assert output_files[0] == output_path

    # Verify file contents
    with open(output_path, 'r') as f:
        lines = f.readlines()

    assert len(lines) == num_conversations, f"Expected {num_conversations} lines, got {len(lines)}"

    # Verify each line is valid JSON
    for i, line in enumerate(lines):
        try:
            data = json.loads(line)
            assert isinstance(data, list), f"Line {i} should be a list"
            assert len(data) == 2, f"Line {i} should have 2 messages"
        except json.JSONDecodeError as e:
            raise AssertionError(f"Line {i} is not valid JSON: {e}")

    logger.info(f"Successfully wrote and verified {num_conversations} conversations")
    logger.info(f"Final file size: {os.path.getsize(output_path)} bytes")
    logger.info("JSONLCodeGenerator test PASSED")


def test_conversation_generator_incremental_writing(logger):
    """Test JSONLConversationGenerator incremental writing."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing JSONLConversationGenerator incremental writing")
    logger.info("=" * 60)

    output_path = "/tmp/test_conv_incremental.jsonl"

    # Clean up previous test file
    if os.path.exists(output_path):
        os.remove(output_path)

    # Create generator with small buffer size for testing
    generator = JSONLConversationGenerator(
        output_path=output_path,
        options=ConversationOutputOptions(validate_roles=False),
        logger=logger,
        buffer_size=5  # Small buffer for quick flushing
    )

    # Check that file was created immediately
    assert os.path.exists(output_path), "Output file should be created immediately"
    initial_size = os.path.getsize(output_path)
    logger.info(f"Initial file size: {initial_size} bytes")

    # Add test conversations
    num_conversations = 17  # Not a multiple of buffer_size to test finalization
    for i in range(num_conversations):
        conversation = {
            'messages': [
                {'role': 'user', 'content': f'This is question number {i} with enough content'},
                {'role': 'assistant', 'content': f'This is answer number {i} with enough content'}
            ]
        }
        generator.add(conversation)

        # Check file size grows after buffer flushes
        if (i + 1) % 5 == 0:
            time.sleep(0.1)  # Give time for flush
            current_size = os.path.getsize(output_path)
            logger.info(f"After {i+1} items: {current_size} bytes")
            assert current_size > initial_size, f"File should grow after flush at item {i+1}"
            initial_size = current_size

    # Finalize
    output_files = generator.finalize()
    assert len(output_files) == 1
    assert output_files[0] == output_path

    # Verify file contents
    with open(output_path, 'r') as f:
        lines = f.readlines()

    assert len(lines) == num_conversations, f"Expected {num_conversations} lines, got {len(lines)}"

    # Verify each line is valid JSON
    for i, line in enumerate(lines):
        try:
            data = json.loads(line)
            assert isinstance(data, list), f"Line {i} should be a list"
            assert len(data) == 2, f"Line {i} should have 2 messages"
        except json.JSONDecodeError as e:
            raise AssertionError(f"Line {i} is not valid JSON: {e}")

    logger.info(f"Successfully wrote and verified {num_conversations} conversations")
    logger.info(f"Final file size: {os.path.getsize(output_path)} bytes")
    logger.info("JSONLConversationGenerator test PASSED")


def main():
    """Run tests."""
    logger = setup_logging()

    try:
        test_code_generator_incremental_writing(logger)
        test_conversation_generator_incremental_writing(logger)

        logger.info("\n" + "=" * 60)
        logger.info("ALL TESTS PASSED")
        logger.info("=" * 60)

    except AssertionError as e:
        logger.error(f"TEST FAILED: {e}")
        return 1
    except Exception as e:
        logger.error(f"TEST ERROR: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
