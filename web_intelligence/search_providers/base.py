"""
Abstract interface for web search providers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, runtime_checkable


@dataclass
class SearchResult:
    """A single web search result."""

    url: str
    title: str
    snippet: str


@runtime_checkable
class BaseSearchProvider(Protocol):
    """Protocol that every search provider must satisfy."""

    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Search the web and return a list of results.

        Args:
            query: Natural language search query.
            max_results: Maximum number of results to return.

        Returns:
            List of SearchResult objects.
        """
        ...
