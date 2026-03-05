"""
Pluggable embedding backends for Web Intelligence.

The library auto-detects the best available backend on import.
Users can also explicitly choose one:

    from web_intelligence.embedders import SentenceTransformerEmbedder
    embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

    from web_intelligence.embedders import FastEmbedEmbedder
    embedder = FastEmbedEmbedder()  # lightweight, no torch

    from web_intelligence.embedders import OpenAIEmbedder
    embedder = OpenAIEmbedder(api_key="sk-...")

    from web_intelligence.embedders import OllamaEmbedder
    embedder = OllamaEmbedder(model="nomic-embed-text")
"""

from .base import BaseEmbedder

# Always importable — they each guard their own imports.
from .sentence_transformer import SentenceTransformerEmbedder
from .fastembed_embedder import FastEmbedEmbedder
from .openai_embedder import OpenAIEmbedder
from .ollama_embedder import OllamaEmbedder


def auto_detect_embedder(**kwargs) -> BaseEmbedder:
    """
    Return the best available embedder, in order of preference:

    1. sentence-transformers (if torch is installed — most capable)
    2. fastembed (lightweight ONNX alternative)
    3. Raise NoEmbedderError with install instructions.
    """
    # Try sentence-transformers first (user likely already has torch)
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        import torch  # noqa: F401
        return SentenceTransformerEmbedder(**kwargs)
    except ImportError:
        pass

    # Try fastembed (lightweight)
    try:
        import fastembed  # noqa: F401
        return FastEmbedEmbedder(**kwargs)
    except ImportError:
        pass

    from ..exceptions import NoEmbedderError
    raise NoEmbedderError()


__all__ = [
    "BaseEmbedder",
    "SentenceTransformerEmbedder",
    "FastEmbedEmbedder",
    "OpenAIEmbedder",
    "OllamaEmbedder",
    "auto_detect_embedder",
]
