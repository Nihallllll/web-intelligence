import trafilatura
from dataclasses import dataclass
from typing import Optional

@dataclass
class ExtractedContent:
    text: str
    title: str
    url: str
    word_count: int
    published_date: Optional[str] = None

def extract_content(html: str, url: str) -> ExtractedContent:
    text = trafilatura.extract(html, include_comments=False, include_tables=False)
    
    if not text:
        text = trafilatura.extract(html, output_format='txt', no_fallback=False)
    
    metadata = trafilatura.extract_metadata(html)
    title = metadata.title if metadata and metadata.title else "Untitled"
    
    return ExtractedContent(
        text=text or "",
        title=title,
        url=url,
        word_count=len(text.split()) if text else 0,
        published_date=metadata.date if metadata else None
    )