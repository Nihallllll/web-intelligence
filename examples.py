"""
Simple usage examples for Web Intelligence library
"""

# Example 1: Quick Start - Index a single URL
def example_quick_start():
    from src.optimized_pipeline import FastPipeline
    
    # Initialize (auto-detects GPU)
    pipeline = FastPipeline()
    
    # Index a URL
    result = pipeline.index_url("https://example.com")
    
    print(f"✓ Indexed: {result['title']}")
    print(f"  Chunks: {result['chunks_count']}")
    

# Example 2: Batch Indexing - Multiple URLs at once (FAST!)
def example_batch_indexing():
    from src.optimized_pipeline import FastPipeline
    
    pipeline = FastPipeline(cache_enabled=True)
    
    # Index multiple URLs concurrently
    urls = [
        "https://python.org",
        "https://github.com",
        "https://stackoverflow.com"
    ]
    
    results = pipeline.index_batch(urls)
    
    for r in results:
        if r['success']:
            print(f"✓ {r['url']}: {r['chunks_count']} chunks")
        else:
            print(f"✗ {r['url']}: {r['error']}")


# Example 3: Search indexed content
def example_search():
    from src.optimized_pipeline import FastPipeline
    
    pipeline = FastPipeline()
    
    # Index some content first
    pipeline.index_url("https://example.com")
    
    # Search
    results = pipeline.search("python programming", limit=5)
    
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['metadata']['title']}")
        print(f"   URL: {r['metadata']['url']}")
        print(f"   Score: {r['score']:.3f}")
        print(f"   Text: {r['text'][:150]}...")


# Example 4: Using caching for speed
def example_caching():
    from src.optimized_pipeline import FastPipeline
    import time
    
    pipeline = FastPipeline(cache_enabled=True)
    
    url = "https://example.com"
    
    # First time - slow
    start = time.time()
    result1 = pipeline.index_url(url)
    time1 = time.time() - start
    
    # Second time - cached (instant!)
    start = time.time()
    result2 = pipeline.index_url(url)
    time2 = time.time() - start
    
    print(f"First run: {time1:.2f}s")
    print(f"Cached run: {time2:.2f}s")
    print(f"Speedup: {time1/time2:.1f}x faster!")


# Example 5: View stored content
def example_view_content():
    from src.optimized_pipeline import FastPipeline
    
    pipeline = FastPipeline()
    
    # Check stats
    stats = pipeline.stats()
    print(f"Total chunks indexed: {stats['total_chunks']}")
    print(f"Embedding model: {stats['embedding_model']}")
    print(f"Device: {stats['device']}")


# Example 6: Original pipeline (simple, slower)
def example_original():
    from src.pipeline import IndexingPipeLine
    
    pipeline = IndexingPipeLine()
    pipeline.index_url("https://example.com")
    

if __name__ == "__main__":
    print("Web Intelligence - Usage Examples\n")
    
    print("=" * 60)
    print("Example 1: Quick Start")
    print("=" * 60)
    example_quick_start()
    
    print("\n" + "=" * 60)
    print("Example 2: Batch Indexing")
    print("=" * 60)
    example_batch_indexing()
    
    print("\n" + "=" * 60)
    print("Example 3: Search")
    print("=" * 60)
    example_search()
    
    print("\n" + "=" * 60)
    print("Example 4: Caching")
    print("=" * 60)
    example_caching()
    
    print("\n" + "=" * 60)
    print("Example 5: Stats")
    print("=" * 60)
    example_view_content()
