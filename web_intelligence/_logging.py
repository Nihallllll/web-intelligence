import logging

logger = logging.getLogger("web_intelligence")
logger.addHandler(logging.NullHandler())


def setup_logging(level: int = logging.INFO, fmt: str | None = None):
    if fmt is None:
        fmt = "[%(levelname)s] web_intelligence.%(name)s: %(message)s"

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger("web_intelligence")
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)
    root.setLevel(level)
