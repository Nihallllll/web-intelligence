from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, runtime_checkable


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str


@runtime_checkable
class BaseSearchProvider(Protocol):
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]: ...
