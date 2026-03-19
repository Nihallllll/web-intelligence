__version__ = "0.4.0"

from .optimized_pipeline import FastPipeline
from .config import Config, default_config
from .context_formatter import (
    RetrievedContext,
    format_context_plain,
    format_context_numbered,
    format_context_structured,
)
from .exceptions import (
    WebIntelligenceError,
    CrawlError,
    ExtractionError,
    EmbeddingError,
    NoEmbedderError,
    VectorStoreError,
    SearchProviderError,
    ConfigError,
)
from ._logging import setup_logging

__all__ = [
    "FastPipeline",
    "Config",
    "default_config",
    "RetrievedContext",
    "format_context_plain",
    "format_context_numbered",
    "format_context_structured",
    "WebIntelligenceError",
    "CrawlError",
    "ExtractionError",
    "EmbeddingError",
    "NoEmbedderError",
    "VectorStoreError",
    "SearchProviderError",
    "ConfigError",
    "setup_logging",
]
