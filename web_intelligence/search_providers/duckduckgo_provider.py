from __future__ import annotations

import logging
from typing import List

from .base import BaseSearchProvider, SearchResult

logger = logging.getLogger("web_intelligence.search_providers.duckduckgo")


def _import_ddgs():
    try:
        from ddgs import DDGS
        return DDGS
    except ImportError:
        pass
    try:
        from duckduckgo_search import DDGS
        return DDGS
    except ImportError:
        raise ImportError(
            "DuckDuckGo provider requires 'ddgs'. "
            "Install with:  pip install ddgs\n"
            "Or:  pip install web-intelligence[search]"
        )


class DuckDuckGoSearchProvider:
    def __init__(self, region: str = "wt-wt", safesearch: str = "moderate"):
        self._DDGS = _import_ddgs()
        self.region = region
        self.safesearch = safesearch

    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        DDGS = self._DDGS

        logger.info("Searching DuckDuckGo: '%s' (max_results=%d)", query, max_results)

        results: List[SearchResult] = []
        try:
            with DDGS() as ddgs:
                raw = ddgs.text(
                    query,
                    region=self.region,
                    safesearch=self.safesearch,
                    max_results=max_results,
                )
                for item in raw:
                    results.append(
                        SearchResult(
                            url=item.get("href", ""),
                            title=item.get("title", ""),
                            snippet=item.get("body", ""),
                        )
                    )
        except Exception as e:
            logger.error("DuckDuckGo search failed: %s", e)
            from ..exceptions import SearchProviderError
            raise SearchProviderError("duckduckgo", str(e))

        logger.info("DuckDuckGo returned %d results", len(results))
        return results
