"""Web content indexing pipeline with async crawling and multi-tier caching."""

from typing import List, Dict, Optional
import asyncio
from .async_crawler import crawl_urls_batch
from .fast_embedder import FastEmbedder
from .cache import URLCache, ContentCache
from .extractor import extract_content
from .chunker import chunk_text
from .vector_store import VectorStore
import uuid
from datetime import datetime


class FastPipeline:
    """End-to-end pipeline: crawl URLs, extract text, embed, and store in vector DB."""
    
    def __init__(
        self,
        storage_path: str = "./data/chroma",
        cache_enabled: bool = True,
        use_gpu: bool = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_embedding_cache: bool = True
    ):
        """Initialize pipeline components."""
        print("Initializing pipeline...")
        
        self.embedder = FastEmbedder(
            model_name=embedding_model,
            device="cuda" if use_gpu else None,
            use_cache=use_embedding_cache
        )
        
        self.vector_store = VectorStore(persist_directory=storage_path)
        
        self.url_cache = URLCache() if cache_enabled else None
        
        self.content_cache = ContentCache() if cache_enabled else None
        
        self.stats_data = {
            'urls_processed': 0,
            'urls_cached': 0,
            'duplicates_skipped': 0,
            'chunks_created': 0
        }
        
        print("✓ Pipeline ready!")
    
    def index_url(self, url: str, skip_cache: bool = False) -> Dict:
        """Index a single URL. Returns result dict with success, title, chunks_count."""
        if self.url_cache and not skip_cache:
            if self.url_cache.is_cached(url):
                cached_data = self.url_cache.get(url)
                print(f"Using cached data for {url}")
                self.stats_data['urls_cached'] += 1
                return {
                    'success': True,
                    'url': url,
                    'cached': True,
                    'title': cached_data.get('data', {}).get('title', 'Cached'),
                    'chunks_count': cached_data.get('data', {}).get('chunks', 0),
                    'indexed_at': cached_data['cached_at']
                }
                    
        results = asyncio.run(self._index_urls_async([url]))
        result = results[0]
        
        if self.url_cache and result['success']:
            self.url_cache.set(url, {
                'doc_id': result['doc_id'],
                'chunks': result['chunks_count'],
                'title': result['title']
            })
        
        self.stats_data['urls_processed'] += 1
        return result
    
    async def _index_urls_async(self, urls: List[str]) -> List[Dict]:
        """Crawl, extract, chunk, embed, and store content for a list of URLs."""
        results = []
        
        crawl_results = await crawl_urls_batch(urls, max_concurrent=10)
        
        for crawl_result in crawl_results:
            if not crawl_result.success:
                results.append({
                    'success': False,
                    'url': crawl_result.url,
                    'error': crawl_result.error or 'Failed to crawl'
                })
                continue
            
            extracted = extract_content(crawl_result.html, crawl_result.url)
            if not extracted.text:
                results.append({
                    'success': False,
                    'url': crawl_result.url,
                    'error': 'No content extracted'
                })
                continue
            
            if self.content_cache:
                if self.content_cache.is_duplicate(extracted.text):
                    existing = self.content_cache.get_existing(extracted.text)
                    print(f"Duplicate content detected, skipping (same as {existing['url']})")
                    self.stats_data['duplicates_skipped'] += 1
                    results.append({
                        'success': True,
                        'url': crawl_result.url,
                        'duplicate': True,
                        'original_url': existing['url']
                    })
                    continue
            
            doc_id = str(uuid.uuid4())
            chunks = chunk_text(extracted.text, doc_id, crawl_result.url)
            
            if len(chunks) == 0:
                results.append({
                    'success': False,
                    'url': crawl_result.url,
                    'error': 'No chunks created'
                })
                continue
            
            chunk_texts = [c.text for c in chunks]
            vectors = self.embedder.embed_batch(chunk_texts, batch_size=64)
            
            metadatas = [{
                "text": c.text,
                "chunk_index": c.chunk_index,
                "title": extracted.title,
                "url": c.url,
                "doc_id": doc_id,
                "word_count": c.token_count,
                "indexed_at": datetime.now().isoformat()
            } for c in chunks]
            
            ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            
            self.vector_store.add(vectors, metadatas, ids)
            
            if self.content_cache:
                self.content_cache.mark_as_indexed(extracted.text, doc_id, crawl_result.url)
            
            self.stats_data['chunks_created'] += len(chunks)
            
            results.append({
                'success': True,
                'url': crawl_result.url,
                'doc_id': doc_id,
                'title': extracted.title,
                'chunks_count': len(chunks),
                'word_count': extracted.word_count,
                'cached': False,
                'duplicate': False
            })
        
        return results
    
    def index_batch(self, urls: List[str], skip_cached: bool = True) -> List[Dict]:
        """Index multiple URLs concurrently. Skips already-cached URLs by default."""
        import time
        
        urls_to_process = []
        cached_results = []
        
        if self.url_cache and skip_cached:
            for url in urls:
                if self.url_cache.is_cached(url):
                    cached_data = self.url_cache.get(url)
                    cached_results.append({
                        'success': True,
                        'url': url,
                        'cached': True,
                        'title': cached_data.get('data', {}).get('title', 'Cached'),
                        'chunks_count': cached_data.get('data', {}).get('chunks', 0),
                        'indexed_at': cached_data['cached_at']
                    })
                    self.stats_data['urls_cached'] += 1
                else:
                    urls_to_process.append(url)
        else:
            urls_to_process = urls
        
        if len(urls_to_process) == 0:
            return cached_results
        
        print(f"Processing {len(urls_to_process)} URLs...")
        if cached_results:
            print(f"{len(cached_results)} URLs already cached, skipping")
        
        start = time.time()
        new_results = asyncio.run(self._index_urls_async(urls_to_process))
        elapsed = time.time() - start
        
        success_count = sum(1 for r in new_results if r.get('success'))
        duplicate_count = sum(1 for r in new_results if r.get('duplicate'))
        
        print(f"\n{'='*50}")
        print(f"✓ Processed {len(urls_to_process)} URLs in {elapsed:.2f}s")
        print(f"  ├─ Success: {success_count}")
        print(f"  ├─ Duplicates skipped: {duplicate_count}")
        print(f"  ├─ Speed: {elapsed/max(len(urls_to_process), 1):.2f}s per URL")
        print(f"  └─ Cached for next time: {len(cached_results)}")
        print(f"{'='*50}")
        
        for result in new_results:
            if result.get('success') and not result.get('duplicate') and self.url_cache:
                self.url_cache.set(result['url'], {
                    'doc_id': result.get('doc_id', ''),
                    'chunks': result.get('chunks_count', 0),
                    'title': result.get('title', '')
                })
        
        self.stats_data['urls_processed'] += len(urls_to_process)
        return cached_results + new_results
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Semantic search across indexed content. Returns ranked chunks."""
        query_vector = self.embedder.embed(query)
        
        results = self.vector_store.search(query_vector, limit=limit)
        
        return results
    
    def stats(self) -> Dict:
        """Return pipeline statistics: DB chunk count, model info, cache metrics."""
        collection_count = self.vector_store.collection.count()
        
        stats = {
            'total_chunks_in_database': collection_count,
            'embedding_model': self.embedder.model_name,
            'embedding_dimension': self.embedder.dimension,
            'device': self.embedder.device,
            'session_stats': self.stats_data,
            'embedding_cache': self.embedder.get_cache_stats()
        }
        
        if self.url_cache:
            stats['url_cache'] = self.url_cache.stats()
        
        return stats
    
    def clear_all_caches(self):
        """Clear URL, content, and embedding caches."""
        if self.url_cache:
            self.url_cache.clear()
        if self.content_cache:
            self.content_cache.clear()
        if self.embedder:
            self.embedder.clear_cache()


if __name__ == "__main__":
    print("="*60)
    print("WEB INTELLIGENCE PIPELINE DEMO")
    print("="*60)
    
    pipeline = FastPipeline()
    
    urls = [
        "https://example.com",
        "https://www.python.org/about/"
    ]
    
    print("\nIndexing URLs...")
    results = pipeline.index_batch(urls)
    
    print("\nResults:")
    for r in results:
        status = "✓" if r.get('success') else "✗"
        cached = " (cached)" if r.get('cached') else ""
        print(f"  {status} {r['url']}{cached}")
    
    print("\nSearching for 'python'...")
    search_results = pipeline.search("python", limit=3)
    for i, r in enumerate(search_results, 1):
        print(f"  {i}. {r['metadata']['title'][:50]}")
    
    print("\nPipeline Stats:")
    stats = pipeline.stats()
    print(f"  Total chunks: {stats['total_chunks_in_database']}")
    print(f"  Device: {stats['device']}")
