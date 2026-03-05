"""
Custom exception types for Web Intelligence.

Provides specific error classes so users can catch and handle
different failure modes (crawl errors, extraction errors, etc.).
"""


class WebIntelligenceError(Exception):
    """Base exception for all Web Intelligence errors."""
    pass


class CrawlError(WebIntelligenceError):
    """Raised when a URL cannot be fetched."""

    def __init__(self, url: str, message: str, status_code: int = 0):
        self.url = url
        self.status_code = status_code
        super().__init__(f"Failed to crawl {url}: {message}")


class ExtractionError(WebIntelligenceError):
    """Raised when HTML content cannot be extracted into usable text."""

    def __init__(self, url: str, message: str = "No content extracted"):
        self.url = url
        super().__init__(f"Extraction failed for {url}: {message}")


class EmbeddingError(WebIntelligenceError):
    """Raised when text cannot be embedded into a vector."""

    def __init__(self, message: str):
        super().__init__(f"Embedding error: {message}")


class NoEmbedderError(WebIntelligenceError):
    """Raised when no embedding backend is available."""

    def __init__(self):
        super().__init__(
            "No embedding backend found. Install one:\n"
            "  pip install web-intelligence[fastembed]     # lightweight, ~200MB\n"
            "  pip install web-intelligence[gpu]           # sentence-transformers + torch, ~2GB\n"
            "  pip install fastembed                       # standalone\n"
            "  pip install sentence-transformers torch     # standalone\n"
            "\n"
            "Or pass your own embedder:\n"
            "  from web_intelligence.embedders import OpenAIEmbedder\n"
            "  pipeline = FastPipeline(embedder=OpenAIEmbedder(api_key='...'))"
        )


class VectorStoreError(WebIntelligenceError):
    """Raised when vector store operations fail."""

    def __init__(self, message: str):
        super().__init__(f"Vector store error: {message}")


class SearchProviderError(WebIntelligenceError):
    """Raised when a web search provider fails."""

    def __init__(self, provider: str, message: str):
        self.provider = provider
        super().__init__(f"Search provider '{provider}' error: {message}")


class ConfigError(WebIntelligenceError):
    """Raised for invalid configuration."""

    def __init__(self, message: str):
        super().__init__(f"Configuration error: {message}")
