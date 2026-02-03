"""Test if the crawler fix works"""
import sys
sys.path.insert(0, '.')

from web_intelligence.crawler import crawl_url
from web_intelligence.extractor import extract_content

# Test crawling Wikipedia
url = "https://en.wikipedia.org/wiki/Nostradamus"
print(f"Testing crawler on: {url}\n")

result = crawl_url(url)

print(f"Status Code: {result.status_code}")
print(f"Success: {result.success}")
print(f"HTML Length: {len(result.html):,} characters")

if result.success:
    print(f"\n{'='*60}")
    print("SUCCESS! Wikipedia accepted our request")
    print(f"{'='*60}")
    
    # Extract content
    extracted = extract_content(result.html, result.url)
    
    print(f"\nTitle: {extracted.title}")
    print(f"Word Count: {extracted.word_count:,}")
    print(f"Text Length: {len(extracted.text):,} characters")
    print(f"\nFirst 500 characters of extracted text:")
    print("-" * 60)
    print(extracted.text[:500])
    print("-" * 60)
else:
    print(f"\n{'='*60}")
    print(f"FAILED: {result.error or 'HTTP ' + str(result.status_code)}")
    print(f"{'='*60}")
    print("\nReceived HTML:")
    print(result.html[:500])
