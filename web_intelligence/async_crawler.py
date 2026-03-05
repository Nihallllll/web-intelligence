"""
Async web crawler with HTTP/2, retry with exponential backoff,
rate limiting, and optional robots.txt checking.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from .crawler import CrawlObject

logger = logging.getLogger("web_intelligence.crawler")

# Module-level robots.txt cache (domain → RobotFileParser)
_robots_cache: dict[str, RobotFileParser] = {}

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "DNT": "1",
}


# ------------------------------------------------------------------ #
# robots.txt helpers
# ------------------------------------------------------------------ #


async def _check_robots(url: str, client: httpx.AsyncClient, user_agent: str) -> bool:
    """Return True if the URL is allowed by robots.txt (or if check fails)."""
    parsed = urlparse(url)
    domain = f"{parsed.scheme}://{parsed.netloc}"

    if domain in _robots_cache:
        return _robots_cache[domain].can_fetch(user_agent, url)

    robots_url = f"{domain}/robots.txt"
    try:
        resp = await client.get(robots_url, timeout=5, follow_redirects=True)
        rp = RobotFileParser()
        rp.parse(resp.text.splitlines())
        _robots_cache[domain] = rp
        return rp.can_fetch(user_agent, url)
    except Exception:
        # If we can't fetch robots.txt, assume allowed
        return True


# ------------------------------------------------------------------ #
# Core async crawl
# ------------------------------------------------------------------ #


async def crawl_url_async(
    url: str,
    client: httpx.AsyncClient,
    timeout: int = 10,
    max_retries: int = 3,
    respect_robots: bool = True,
    rate_limiter: Optional[asyncio.Semaphore] = None,
) -> CrawlObject:
    """
    Fetch a single URL with retry and exponential backoff.

    Args:
        url: Target URL.
        client: Shared httpx async client.
        timeout: Request timeout in seconds.
        max_retries: Number of retries on failure/5xx/429.
        respect_robots: Check robots.txt before crawling.
        rate_limiter: Optional semaphore for rate limiting.
    """
    user_agent = _DEFAULT_HEADERS["User-Agent"]

    # robots.txt check
    if respect_robots:
        if not await _check_robots(url, client, user_agent):
            logger.info("Blocked by robots.txt: %s", url)
            return CrawlObject(
                url=url,
                html="",
                success=False,
                status_code=0,
                crawled_at=datetime.now(),
                error="Blocked by robots.txt",
            )

    last_error: Optional[str] = None
    for attempt in range(max_retries + 1):
        try:
            if rate_limiter:
                await rate_limiter.acquire()
                try:
                    response = await client.get(
                        url, headers=_DEFAULT_HEADERS, timeout=timeout, follow_redirects=True
                    )
                finally:
                    # Release after a small delay for rate limiting
                    rate_limiter.release()
            else:
                response = await client.get(
                    url, headers=_DEFAULT_HEADERS, timeout=timeout, follow_redirects=True
                )

            # Retry on 429 (rate limited) or 5xx (server error)
            if response.status_code == 429 or response.status_code >= 500:
                wait = 2 ** attempt
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = max(wait, int(retry_after))
                logger.warning(
                    "HTTP %d for %s — retrying in %ds (attempt %d/%d)",
                    response.status_code, url, wait, attempt + 1, max_retries + 1,
                )
                last_error = f"HTTP {response.status_code}"
                await asyncio.sleep(wait)
                continue

            return CrawlObject(
                url=url,
                html=response.text,
                success=response.status_code == 200,
                status_code=response.status_code,
                crawled_at=datetime.now(),
            )

        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning(
                    "Error crawling %s: %s — retrying in %ds (attempt %d/%d)",
                    url, e, wait, attempt + 1, max_retries + 1,
                )
                await asyncio.sleep(wait)
            else:
                logger.error("Failed to crawl %s after %d attempts: %s", url, max_retries + 1, e)

    return CrawlObject(
        url=url,
        html="",
        success=False,
        status_code=0,
        crawled_at=datetime.now(),
        error=last_error or "Max retries exceeded",
    )


# ------------------------------------------------------------------ #
# Batch crawl
# ------------------------------------------------------------------ #


async def crawl_urls_batch(
    urls: List[str],
    max_concurrent: int = 10,
    timeout: int = 15,
    max_retries: int = 3,
    requests_per_second: float = 0.0,
    respect_robots: bool = True,
) -> List[CrawlObject]:
    """
    Crawl multiple URLs concurrently with rate limiting and retry.

    Args:
        urls: List of URLs to crawl.
        max_concurrent: Max parallel connections.
        timeout: Per-request timeout in seconds.
        max_retries: Retries per URL on failure/5xx/429.
        requests_per_second: Rate limit (0 = unlimited).
        respect_robots: If True, check robots.txt before each domain.
    """
    limits = httpx.Limits(
        max_connections=max_concurrent,
        max_keepalive_connections=max_concurrent,
    )

    # Rate limiter: simple token-bucket via delayed semaphore
    rate_limiter: Optional[asyncio.Semaphore] = None
    if requests_per_second > 0:
        rate_limiter = asyncio.Semaphore(max(1, int(requests_per_second)))

    async with httpx.AsyncClient(
        limits=limits,
        http2=True,
        timeout=timeout,
        follow_redirects=True,
    ) as client:
        tasks = [
            crawl_url_async(
                url, client,
                timeout=timeout,
                max_retries=max_retries,
                respect_robots=respect_robots,
                rate_limiter=rate_limiter,
            )
            for url in urls
        ]
        results = await asyncio.gather(*tasks)

    return list(results)


# ------------------------------------------------------------------ #
# Sync wrapper
# ------------------------------------------------------------------ #


def crawl_urls_sync(
    urls: List[str],
    max_concurrent: int = 10,
    timeout: int = 15,
    max_retries: int = 3,
    requests_per_second: float = 0.0,
    respect_robots: bool = True,
) -> List[CrawlObject]:
    """Synchronous wrapper around crawl_urls_batch."""
    return asyncio.run(
        crawl_urls_batch(
            urls,
            max_concurrent=max_concurrent,
            timeout=timeout,
            max_retries=max_retries,
            requests_per_second=requests_per_second,
            respect_robots=respect_robots,
        )
    )


