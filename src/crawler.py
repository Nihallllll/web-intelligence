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
    """Fetch the html from the url"""
    try:
        response = httpx.get(url , timeout= timeout, follow_redirects=True)
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
        html=response.text,
        success=False,
        status_code=response.status_code,
        crawled_at=datetime.now(),
        error=str(e)
        )