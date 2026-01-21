"""Test indexing Wikipedia specifically"""

from web_intelligence import FastPipeline

pipeline = FastPipeline(cache_enabled=False, use_gpu=None)  # Disable cache to force fresh crawl

url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
print(f"Indexing: {url}")

result = pipeline.index_url(url)

if result['success']:
    print(f"✓ Success! Indexed {result['chunks_count']} chunks")
    print(f"  Title: {result['title']}")
else:
    print(f"✗ Failed: {result.get('error', 'Unknown error')}")

# Search for creator
print("\nSearching for 'who created python'...")
results = pipeline.search("who created python guido", limit=5)

for i, r in enumerate(results, 1):
    print(f"\n{i}. {r['text'][:200]}...")
