#!/usr/bin/env python3
"""Training Data Generator CLI.

Usage:
    python generate_data.py --config config.yaml
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.loader import load_config
from config.validator import ConfigValidator
from orchestrator import JobOrchestrator
from utils.logger import setup_logger


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate training data from web, files, and GitHub PRs"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignoring previous state (default: resume from last run)"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all cached state before starting"
    )

    args = parser.parse_args()

    # Set up logger
    logger = setup_logger(
        "training-data-generator",
        log_level=args.log_level
    )

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Validate configuration
        logger.info("Validating configuration...")
        errors = ConfigValidator.validate_config(config)
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            sys.exit(1)

        logger.info("Configuration valid")

        # Override log level if specified in config
        if config.global_config.log_level:
            logger.setLevel(config.global_config.log_level)

        # Handle clear cache flag
        if args.clear_cache:
            logger.info("Clearing cached state...")
            from utils.state import StateManager
            cache_dir = Path(config.global_config.cache_dir)
            state_manager = StateManager(str(cache_dir))
            state_manager.clear_all()
            logger.info("Cache cleared")

        # Determine resume mode
        resume = not args.no_resume
        if resume:
            logger.info("Resume mode: ON (will skip already processed items)")
        else:
            logger.info("Resume mode: OFF (starting fresh)")

        # Create orchestrator and run jobs
        orchestrator = JobOrchestrator(config, logger, resume=resume)
        orchestrator.run_all_jobs()

        logger.info("\n" + "="*60)
        logger.info("All jobs completed successfully!")
        logger.info("="*60)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
