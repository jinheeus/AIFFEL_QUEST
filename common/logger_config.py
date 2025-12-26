import logging
import sys


def setup_logger(name: str = "AURA_LOG") -> logging.Logger:
    """
    Sets up a standardized logger with a specific format.
    """
    logger = logging.getLogger(name)

    # helper to avoid adding multiple handlers if logger is reused
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        # Create formatter
        # Format: Time | Level | Component | Message
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

        # Add handler
        logger.addHandler(handler)

        # Prevent propagation to root logger if configured elsewhere
        logger.propagate = False

    return logger
