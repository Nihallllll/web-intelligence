from typing import List, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class BaseEmbedder(Protocol):
    model_name: str
    dimension: int

    def embed(self, text: str) -> List[float]: ...

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> List[List[float]]: ...

    def get_cache_stats(self) -> Dict:
        return {}

    def clear_cache(self) -> None:
        pass
