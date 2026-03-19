import copy
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


def _load_dotenv():
    try:
        from dotenv import load_dotenv
        for candidate in [Path.cwd() / ".env", Path(__file__).parent.parent / ".env"]:
            if candidate.exists():
                load_dotenv(candidate)
                return True
    except ImportError:
        pass
    return False


_load_dotenv()


def _env(key: str, default: Any = None, cast: type = str) -> Any:
    val = os.environ.get(key, default)
    if val is None:
        return None
    if cast == bool:
        return str(val).lower() in ("1", "true", "yes", "on")
    return cast(val)


@dataclass
class CrawlerConfig:
    max_concurrent: int = _env("WI_CRAWLER_MAX_CONCURRENT", 10, int)
    timeout: int = _env("WI_CRAWLER_TIMEOUT", 15, int)
    max_retries: int = _env("WI_CRAWLER_MAX_RETRIES", 3, int)
    requests_per_second: float = _env("WI_CRAWLER_REQUESTS_PER_SEC", 0.0, float)
    respect_robots: bool = _env("WI_CRAWLER_RESPECT_ROBOTS", True, bool)
    user_agent: str = _env(
        "WI_CRAWLER_USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )


@dataclass
class ChunkerConfig:
    chunk_size: int = _env("WI_CHUNK_SIZE", 400, int)
    chunk_overlap: int = _env("WI_CHUNK_OVERLAP", 50, int)


@dataclass
class EmbeddingConfig:
    model_name: str = _env("WI_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    device: Optional[str] = _env("WI_EMBEDDING_DEVICE", None)
    use_cache: bool = _env("WI_EMBEDDING_CACHE", True, bool)
    batch_size: int = _env("WI_EMBEDDING_BATCH_SIZE", 64, int)


@dataclass
class VectorStoreConfig:
    persist_directory: str = _env("WI_VECTOR_STORE_PATH", "./data")
    collection_name: str = _env("WI_COLLECTION_NAME", "web_content")


@dataclass
class SearchConfig:
    default_limit: int = _env("WI_SEARCH_LIMIT", 5, int)
    max_context_words: int = _env("WI_SEARCH_MAX_CONTEXT_WORDS", 3000, int)
    include_sources: bool = _env("WI_SEARCH_INCLUDE_SOURCES", True, bool)
    min_relevance_score: float = _env("WI_SEARCH_MIN_SCORE", 0.0, float)


@dataclass
class ServerConfig:
    host: str = _env("WI_SERVER_HOST", "0.0.0.0")
    port: int = _env("WI_SERVER_PORT", 8000, int)
    reload: bool = _env("WI_SERVER_RELOAD", False, bool)


@dataclass
class Config:
    crawler: CrawlerConfig = field(default_factory=CrawlerConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    cache_enabled: bool = _env("WI_CACHE_ENABLED", True, bool)

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    def copy(self) -> "Config":
        return copy.deepcopy(self)


def default_config() -> Config:
    return Config()
