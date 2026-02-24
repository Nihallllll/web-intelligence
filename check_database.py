"""Check what's in the ChromaDB database."""

from web_intelligence import FastPipeline

pipeline = FastPipeline(cache_enabled=True, use_gpu=None)
stats = pipeline.stats()

print(f"Total chunks: {stats['total_chunks_in_database']}")
print(f"Device: {stats['device']}, Model: {stats['embedding_model']}")

results = pipeline.search("python", limit=10)
print(f"\nSearch 'python': {len(results)} results")
for i, r in enumerate(results, 1):
    url = r['metadata'].get('url', 'Unknown')
    text_preview = r['text'][:100].replace('\n', ' ')
    score = r.get('score', 0)
    print(f"\n{i}. Score: {score:.3f} | {url}")
    print(f"   {text_preview}...")

all_chunks = pipeline.search("", limit=100)
unique_urls = set(r['metadata']['url'] for r in all_chunks if 'url' in r['metadata'])

print(f"\nIndexed URLs ({len(unique_urls)}):")
if unique_urls:
    for url in sorted(unique_urls):
        chunks_for_url = sum(1 for r in all_chunks if r['metadata'].get('url') == url)
        print(f"  {url} ({chunks_for_url} chunks)")
else:
    print("  No URLs found in database.")
