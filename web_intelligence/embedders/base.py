"""
Abstract base class (Protocol) for all embedding backends.

Any object that implements ``embed()`` and ``embed_batch()`` with the
correct signatures can be used as an embedder — duck typing via Protocol.
"""

from typing import List, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class BaseEmbedder(Protocol):
    """
    Protocol that every embedder must satisfy.

    Implementations must provide:
        - embed(text) -> List[float]
        - embed_batch(texts) -> List[List[float]]
        - dimension: int  (dimensionality of the output vectors)
        - model_name: str

    Optional:
        - get_cache_stats() -> Dict
        - clear_cache()
    """

    model_name: str
    dimension: int

    def embed(self, text: str) -> List[float]:
        """Embed a single text string into a vector."""
        ...

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> List[List[float]]:
        """Embed multiple texts. Returns list of vectors (same order)."""
        ...

    # Optional methods — provide defaults so they aren't required.

    def get_cache_stats(self) -> Dict:
        """Return cache hit/miss stats (optional)."""
        return {}

    def clear_cache(self) -> None:
        """Clear the embedding cache (optional)."""
        pass
