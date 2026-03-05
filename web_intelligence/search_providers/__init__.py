"""
Pluggable web search backends for Web Intelligence.

    from web_intelligence.search_providers import DuckDuckGoSearchProvider

    provider = DuckDuckGoSearchProvider()
    results = provider.search("quantum computing", max_results=5)
    # → [{"url": "...", "title": "...", "snippet": "..."}, ...]
"""

from .base import BaseSearchProvider, SearchResult
from .duckduckgo_provider import DuckDuckGoSearchProvider

__all__ = [
    "BaseSearchProvider",
    "SearchResult",
    "DuckDuckGoSearchProvider",
]
