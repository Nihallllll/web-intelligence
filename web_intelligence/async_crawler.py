# src/async_crawler.py - FAST async crawler
"""
🚗 THE SUPER FAST WEBSITE VISITOR 🚗

Imagine you need to visit 5 friends' houses to collect toys.

SLOW way (old):
    Go to house 1 → wait → get toy → come back
    Go to house 2 → wait → get toy → come back
    Go to house 3 → wait → get toy → come back
    (Takes 15 minutes!)

FAST way (this code!):
    Send 5 robots to ALL houses at the SAME TIME!
    All robots come back together
    (Takes only 3 minutes!) ⚡

This is called "ASYNC" (asynchronous) - doing many things at once!

Also uses HTTP/2:
    HTTP/1: Like having separate phone calls with each friend
    HTTP/2: Like a GROUP CALL with everyone at once! 📞
"""

import httpx
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
from .crawler import CrawlObject


async def crawl_url_async(url: str, client: httpx.AsyncClient, timeout: int = 10) -> CrawlObject:
    """
    🤖 SEND ONE ROBOT TO ONE HOUSE 🤖
    
    This robot visits one website and brings back the content.
    It's fast because it doesn't block other robots!
    
    Args:
        url: The website address (like a house address)
        client: The shared car all robots use (connection pool)
        timeout: How long to wait before giving up (in seconds)
    
    Returns:
        CrawlObject with the website content (the toy we collected!)
    """
    try:
        # Visit the website and get the content
        response = await client.get(url, timeout=timeout, follow_redirects=True)
        return CrawlObject(
            url=url,
            html=response.text,  # The content (like the toy)
            success=response.status_code == 200,  # 200 = success!
            status_code=response.status_code,
            crawled_at=datetime.now()
        )
    except Exception as e:
        # Something went wrong (house was locked, nobody home, etc.)
        return CrawlObject(
            url=url,
            html="",
            success=False,
            status_code=0,
            crawled_at=datetime.now(),
            error=str(e)
        )


async def crawl_urls_batch(urls: List[str], max_concurrent: int = 10, timeout: int = 15) -> List[CrawlObject]:
    """    
    Instead of visiting websites one-by-one ,
    we visit ALL of them at the SAME TIME 
    
    Like sending 10 robots to 10 houses simultaneously!
    
    List of CrawlObjects 
        
    Example:
        >>> urls = ["https://site1.com", "https://site2.com", "https://site3.com"]
        >>> results = await crawl_urls_batch(urls)
        >>> # All 3 visited at once! Takes ~3 seconds instead of ~9 seconds
    """
    # Configure the shared "car" for all robots
    # HTTP/2 = faster, like a group call instead of individual calls!
    limits = httpx.Limits(
        max_connections=max_concurrent,         # How many robots can go at once
        max_keepalive_connections=max_concurrent  # Keep the car running between trips
    )
    
    # Create the super-fast client with HTTP/2 support
    async with httpx.AsyncClient(
        limits=limits,
        http2=True,  # 🚀 USE HTTP/2 - faster for multiple requests!
        timeout=timeout,
        follow_redirects=True
    ) as client:
        # Send ALL robots at the same time!
        tasks = [crawl_url_async(url, client, timeout) for url in urls]
        
        # Wait for ALL robots to come back
        # asyncio.gather = "wait for everyone to finish"
        results = await asyncio.gather(*tasks)
        
    return results


def crawl_urls_sync(urls: List[str], max_concurrent: int = 10, timeout: int = 15) -> List[CrawlObject]:
    """
    🔄 EASY-TO-USE VERSION 🔄
    
    This wraps the async function so you don't need to use 'await'.
    Like a simple button that does all the magic inside!
    
    Example:
        >>> urls = ["https://site1.com", "https://site2.com"]
        >>> results = crawl_urls_sync(urls)  # Simple! No await needed!
    """
    return asyncio.run(crawl_urls_batch(urls, max_concurrent, timeout))


