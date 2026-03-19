import httpx
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class CrawlObject:
    url : str
    html : str
    status_code : int
    success :bool
    error : Optional[str] = None
    crawled_at : datetime =None


def crawl_url(url : str , timeout=10):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    try:
        response = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
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