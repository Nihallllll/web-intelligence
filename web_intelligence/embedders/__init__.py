from .base import BaseEmbedder

from .sentence_transformer import SentenceTransformerEmbedder
from .fastembed_embedder import FastEmbedEmbedder
from .openai_embedder import OpenAIEmbedder
from .ollama_embedder import OllamaEmbedder


def auto_detect_embedder(**kwargs) -> BaseEmbedder:
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        import torch  # noqa: F401
        return SentenceTransformerEmbedder(**kwargs)
    except ImportError:
        pass

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
