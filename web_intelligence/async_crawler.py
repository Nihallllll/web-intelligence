"""Async web crawler with HTTP/2 support and concurrent request handling."""

import httpx
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
from .crawler import CrawlObject


async def crawl_url_async(url: str, client: httpx.AsyncClient, timeout: int = 10) -> CrawlObject:
    """Fetch a single URL asynchronously using a shared client."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'DNT': '1'
    }
    try:
        response = await client.get(url, headers=headers, timeout=timeout, follow_redirects=True)
        return CrawlObject(
            url=url,
            html=response.text,
            success=response.status_code == 200,
            status_code=response.status_code,
            crawled_at=datetime.now()
        )
    except Exception as e:
        return CrawlObject(
            url=url,
            html="",
            success=False,
            status_code=0,
            crawled_at=datetime.now(),
            error=str(e)
        )


async def crawl_urls_batch(urls: List[str], max_concurrent: int = 10, timeout: int = 15) -> List[CrawlObject]:
    """Crawl multiple URLs concurrently. Returns a list of CrawlObjects."""
    limits = httpx.Limits(
        max_connections=max_concurrent,
        max_keepalive_connections=max_concurrent
    )
    
    async with httpx.AsyncClient(
        limits=limits,
        http2=True,
        timeout=timeout,
        follow_redirects=True
    ) as client:
        tasks = [crawl_url_async(url, client, timeout) for url in urls]
        results = await asyncio.gather(*tasks)
        
    return results


def crawl_urls_sync(urls: List[str], max_concurrent: int = 10, timeout: int = 15) -> List[CrawlObject]:
    """Synchronous wrapper around crawl_urls_batch."""
    return asyncio.run(crawl_urls_batch(urls, max_concurrent, timeout))


