"""Quick script to check what's in the ChromaDB database"""

from web_intelligence import FastPipeline

# Initialize pipeline
pipeline = FastPipeline(cache_enabled=True, use_gpu=None)

# Get stats
stats = pipeline.stats()

print("="*60)
print("DATABASE STATS")
print("="*60)
print(f"Total chunks in database: {stats['total_chunks_in_database']}")
print(f"Device: {stats['device']}")
print(f"Embedding model: {stats['embedding_model']}")

# Try to search for anything
print("\n" + "="*60)
print("SAMPLE SEARCH: 'python'")
print("="*60)

results = pipeline.search("python", limit=10)
print(f"Found {len(results)} chunks")

for i, r in enumerate(results, 1):
    url = r['metadata'].get('url', 'Unknown')
    text_preview = r['text'][:100].replace('\n', ' ')
    score = r.get('score', 0)
    print(f"\n{i}. Score: {score:.3f}")
    print(f"   URL: {url}")
    print(f"   Text: {text_preview}...")

# List all unique URLs
print("\n" + "="*60)
print("ALL INDEXED URLs")
print("="*60)

# Get all chunks (limit to avoid too much output)
all_chunks = pipeline.search("", limit=100)  # Empty query gets all
unique_urls = set(r['metadata']['url'] for r in all_chunks if 'url' in r['metadata'])

if unique_urls:
    for url in sorted(unique_urls):
        chunks_for_url = sum(1 for r in all_chunks if r['metadata'].get('url') == url)
        print(f"  - {url} ({chunks_for_url} chunks)")
else:
    print("  No URLs found in database!")
