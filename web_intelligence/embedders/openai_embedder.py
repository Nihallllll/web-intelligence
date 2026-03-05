"""
OpenAI embedding backend (cloud-based, requires API key).

Install:  pip install openai
"""

from __future__ import annotations

import logging
from typing import List, Optional, Dict

from ..cache import EmbeddingCache

logger = logging.getLogger("web_intelligence.embedders.openai")

# Dimensions for known OpenAI models
_KNOWN_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder:
    """
    Embedding backend using the OpenAI Embeddings API.

    Args:
        api_key: OpenAI API key.  Falls back to ``OPENAI_API_KEY`` env var.
        model_name: Model name (default ``text-embedding-3-small``).
        base_url: Optional custom base URL (for Azure OpenAI, proxies, etc.).
        use_cache: Persist embeddings to disk.
        cache_dir: Directory for the embedding cache.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "text-embedding-3-small",
        base_url: Optional[str] = None,
        use_cache: bool = True,
        cache_dir: str = "./data/cache/embeddings",
    ):
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "OpenAI backend requires 'openai'. Install with:  pip install openai"
            ) from exc

        import openai as _openai

        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        self._client = _openai.OpenAI(**kwargs)
        self.model_name: str = model_name
        self.dimension: int = _KNOWN_DIMS.get(model_name, 1536)

        # Cache
        self.use_cache = use_cache
        self.cache: Optional[EmbeddingCache] = EmbeddingCache(cache_dir=cache_dir) if use_cache else None
        self._cache_hits = 0
        self._cache_misses = 0

        logger.info("OpenAI embedder ready (model=%s, dim=%d)", model_name, self.dimension)

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #

    def embed(self, text: str) -> List[float]:
        if self.cache:
            cached = self.cache.get(text)
            if cached is not None:
                self._cache_hits += 1
                return cached
            self._cache_misses += 1

        resp = self._client.embeddings.create(input=[text], model=self.model_name)
        embedding = resp.data[0].embedding

        if self.cache:
            self.cache.set(text, embedding)
        return embedding

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 128,
        show_progress: bool = False,
    ) -> List[List[float]]:
        if not texts:
            return []

        results: list = [None] * len(texts)
        texts_to_embed: list[str] = []
        indices_to_embed: list[int] = []

        if self.cache:
            for i, text in enumerate(texts):
                cached = self.cache.get(text)
                if cached is not None:
                    results[i] = cached
                    self._cache_hits += 1
                else:
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
                    self._cache_misses += 1
        else:
            texts_to_embed = list(texts)
            indices_to_embed = list(range(len(texts)))

        if not texts_to_embed:
            return results

        # OpenAI accepts up to 2048 inputs per request
        for start in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[start : start + batch_size]
            resp = self._client.embeddings.create(input=batch, model=self.model_name)

            for j, item in enumerate(resp.data):
                global_idx = indices_to_embed[start + j]
                results[global_idx] = item.embedding
                if self.cache:
                    self.cache.set(texts_to_embed[start + j], item.embedding)

        return results

    # ------------------------------------------------------------------ #
    # Cache management
    # ------------------------------------------------------------------ #

    def get_cache_stats(self) -> Dict:
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "embeddings_saved": self.cache.stats() if self.cache else None,
        }

    def clear_cache(self) -> None:
        if self.cache:
            self.cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
