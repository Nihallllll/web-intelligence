from __future__ import annotations

import logging
from typing import List, Optional, Dict

from ..cache import EmbeddingCache

logger = logging.getLogger("web_intelligence.embedders.ollama")


class OllamaEmbedder:
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        use_cache: bool = True,
        cache_dir: str = "./data/cache/embeddings",
    ):
        import httpx  # already a dependency

        self._base_url = base_url.rstrip("/")
        self.model_name: str = model
        self._client = httpx.Client(timeout=60)

        self.dimension: int = self._get_dimension()

        self.use_cache = use_cache
        self.cache: Optional[EmbeddingCache] = EmbeddingCache(cache_dir=cache_dir) if use_cache else None
        self._cache_hits = 0
        self._cache_misses = 0

        logger.info("Ollama embedder ready (model=%s, dim=%d, url=%s)",
                     model, self.dimension, base_url)

    def _get_dimension(self) -> int:
        resp = self._client.post(
            f"{self._base_url}/api/embed",
            json={"model": self.model_name, "input": ["hello"]},
        )
        resp.raise_for_status()
        data = resp.json()
        return len(data["embeddings"][0])

    def _embed_via_api(self, texts: List[str]) -> List[List[float]]:
        resp = self._client.post(
            f"{self._base_url}/api/embed",
            json={"model": self.model_name, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["embeddings"]

    def embed(self, text: str) -> List[float]:
        if self.cache:
            cached = self.cache.get(text)
            if cached is not None:
                self._cache_hits += 1
                return cached
            self._cache_misses += 1

        embedding = self._embed_via_api([text])[0]

        if self.cache:
            self.cache.set(text, embedding)
        return embedding

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
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

        for start in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[start : start + batch_size]
            vectors = self._embed_via_api(batch)

            for j, vec in enumerate(vectors):
                global_idx = indices_to_embed[start + j]
                results[global_idx] = vec
                if self.cache:
                    self.cache.set(texts_to_embed[start + j], vec)

        return results

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
