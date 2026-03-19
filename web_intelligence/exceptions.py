class WebIntelligenceError(Exception):
    pass


class CrawlError(WebIntelligenceError):
    def __init__(self, url: str, message: str, status_code: int = 0):
        self.url = url
        self.status_code = status_code
        super().__init__(f"Failed to crawl {url}: {message}")


class ExtractionError(WebIntelligenceError):
    def __init__(self, url: str, message: str = "No content extracted"):
        self.url = url
        super().__init__(f"Extraction failed for {url}: {message}")


class EmbeddingError(WebIntelligenceError):
    def __init__(self, message: str):
        super().__init__(f"Embedding error: {message}")


class NoEmbedderError(WebIntelligenceError):
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
    def __init__(self, message: str):
        super().__init__(f"Vector store error: {message}")


class SearchProviderError(WebIntelligenceError):
    def __init__(self, provider: str, message: str):
        self.provider = provider
        super().__init__(f"Search provider '{provider}' error: {message}")


class ConfigError(WebIntelligenceError):
    def __init__(self, message: str):
        super().__init__(f"Configuration error: {message}")
