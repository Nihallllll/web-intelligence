from __future__ import annotations

import logging
from typing import List, Optional, Dict

from ..cache import EmbeddingCache

logger = logging.getLogger("web_intelligence.embedders.fastembed")

_MODEL_MAP = {
    "all-MiniLM-L6-v2": "BAAI/bge-small-en-v1.5",
    "all-mpnet-base-v2": "BAAI/bge-base-en-v1.5",
}


class FastEmbedEmbedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        use_cache: bool = True,
        cache_dir: str = "./data/cache/embeddings",
        **kwargs,
    ):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ImportError(
                "FastEmbed backend requires 'fastembed'. "
                "Install with:  pip install fastembed"
            ) from exc

        resolved_name = _MODEL_MAP.get(model_name, model_name)
        if resolved_name != model_name:
            logger.info("Mapped model name '%s' → '%s'", model_name, resolved_name)

        logger.info("Loading fastembed model '%s'", resolved_name)
        self._model = TextEmbedding(model_name=resolved_name)
        self.model_name: str = resolved_name
        self.dimension: int = self._get_dimension()

        self.use_cache = use_cache
        self.cache: Optional[EmbeddingCache] = EmbeddingCache(cache_dir=cache_dir) if use_cache else None
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_dimension(self) -> int:
        probe = list(self._model.embed(["hello"]))[0]
        return len(probe)

    def embed(self, text: str) -> List[float]:
        if self.cache:
            cached = self.cache.get(text)
            if cached is not None:
                self._cache_hits += 1
                return cached
            self._cache_misses += 1

        vector = list(self._model.embed([text]))[0]
        embedding = vector.tolist()

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

        vectors = list(self._model.embed(texts_to_embed, batch_size=batch_size))

        for idx, (text, vector) in enumerate(zip(texts_to_embed, vectors)):
            embedding = vector.tolist()
            results[indices_to_embed[idx]] = embedding
            if self.cache:
                self.cache.set(text, embedding)

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
