"""
Sentence-Transformers embedding backend (requires ``torch`` + ``sentence-transformers``).

This is the most fully-featured backend with GPU support and large model selection.
Install:  pip install sentence-transformers torch
"""

from __future__ import annotations

import logging
from typing import List, Optional, Dict

from ..cache import EmbeddingCache

logger = logging.getLogger("web_intelligence.embedders.sentence_transformer")


class SentenceTransformerEmbedder:
    """
    Embedding backend powered by ``sentence-transformers``.

    Supports GPU acceleration, embedding normalization, and persistent caching.

    Args:
        model_name: Any model from https://huggingface.co/models?library=sentence-transformers
        device: ``'cuda'``, ``'cpu'``, or ``None`` for auto-detect.
        use_cache: Persist embeddings to disk to avoid recomputation.
        cache_dir: Directory for the embedding cache.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        use_cache: bool = True,
        cache_dir: str = "./data/cache/embeddings",
    ):
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers backend requires 'sentence-transformers' and 'torch'. "
                "Install with:  pip install sentence-transformers torch"
            ) from exc

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available — falling back to CPU")
            device = "cpu"

        logger.info("Loading model '%s' on %s", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension: int = self.model.get_sentence_embedding_dimension()
        self.device: str = device
        self.model_name: str = model_name

        # Cache
        self.use_cache = use_cache
        self.cache: Optional[EmbeddingCache] = EmbeddingCache(cache_dir=cache_dir) if use_cache else None
        self._cache_hits = 0
        self._cache_misses = 0

        if device == "cpu":
            import torch as _torch
            _torch.set_num_threads(_torch.get_num_threads())

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #

    def embed(self, text: str) -> List[float]:
        """Embed a single text. Normalized to match ``embed_batch``."""
        if self.cache:
            cached = self.cache.get(text)
            if cached is not None:
                self._cache_hits += 1
                return cached
            self._cache_misses += 1

        vector = self.model.encode(
            text,
            convert_to_tensor=False,
            show_progress_bar=False,
            normalize_embeddings=True,          # <-- FIX: was missing before
        )
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
        """Embed multiple texts with cache-aware batching."""
        if not texts:
            return []

        results: list = [None] * len(texts)
        texts_to_embed: list[str] = []
        indices_to_embed: list[int] = []

        # Check cache first
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

        # GPU benefits from larger batches
        if self.device == "cuda" and batch_size < 64:
            batch_size = 64

        vectors = self.model.encode(
            texts_to_embed,
            batch_size=batch_size,
            convert_to_tensor=False,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )

        for idx, (text, vector) in enumerate(zip(texts_to_embed, vectors)):
            embedding = vector.tolist()
            results[indices_to_embed[idx]] = embedding
            if self.cache:
                self.cache.set(text, embedding)

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
