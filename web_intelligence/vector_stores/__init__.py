from .base import BaseVectorStore
from .chroma_store import ChromaVectorStore
from .numpy_store import NumpyVectorStore


def auto_detect_vector_store(persist_directory: str = "./data", **kwargs) -> BaseVectorStore:
    try:
        import chromadb  # noqa: F401
        return ChromaVectorStore(persist_directory=persist_directory + "/chroma", **kwargs)
    except ImportError:
        pass

    return NumpyVectorStore(persist_path=persist_directory + "/numpy_store.pkl", **kwargs)


__all__ = [
    "BaseVectorStore",
    "ChromaVectorStore",
    "NumpyVectorStore",
    "auto_detect_vector_store",
]
