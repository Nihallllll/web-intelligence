"""Test indexing Wikipedia specifically"""

from web_intelligence import FastPipeline
from langchain_google_genai import ChatGoogleGenerativeAI
pipeline = FastPipeline()  # Disable cache to force fresh crawl

url = "https://www.britannica.com/topic/astrology"
print(f"Indexing: {url}")

result = pipeline.index_url(url)

if result['success']:
    print(f"✓ Success! Indexed {result['chunks_count']} chunks")
    print(f"  Title: {result['title']}")
else:
    print(f"✗ Failed: {result.get('error', 'Unknown error')}")

# Search for creator
print("\nSearching for 'who is nostrademus'...")
results = pipeline.search("who is nostradamous", limit=5)

for i, r in enumerate(results, 1):
    print(f"\n{i}. {r['text'][:200]}...")
