"""
Logging setup for Web Intelligence.

All modules use ``logging.getLogger("web_intelligence")`` so users can
control verbosity with a single call:

    import logging
    logging.getLogger("web_intelligence").setLevel(logging.WARNING)
"""

import logging

# Library root logger — NullHandler by default (best practice for libraries).
# Users opt-in to output by adding their own handler or calling setup_logging().
logger = logging.getLogger("web_intelligence")
logger.addHandler(logging.NullHandler())


def setup_logging(level: int = logging.INFO, fmt: str | None = None):
    """
    Convenience helper to enable console logging for the library.

    Call this once at application startup if you want to see log output:

        from web_intelligence._logging import setup_logging
        setup_logging()                       # INFO level
        setup_logging(logging.DEBUG)          # verbose

    Args:
        level: Logging level (default ``logging.INFO``).
        fmt:   Custom format string.  Falls back to a sensible default.
    """
    if fmt is None:
        fmt = "[%(levelname)s] web_intelligence.%(name)s: %(message)s"

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger("web_intelligence")
    # Avoid duplicate handlers if called multiple times.
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)
    root.setLevel(level)
