"""
Fast text embedding with GPU acceleration and caching support.

.. deprecated:: 0.4.0
    Import from ``web_intelligence.embedders`` instead::

        from web_intelligence.embedders import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder()

    This module is kept for backward compatibility.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict
import torch
from .cache import EmbeddingCache

import warnings

warnings.warn(
    "web_intelligence.fast_embedder is deprecated. "
    "Use web_intelligence.embedders.SentenceTransformerEmbedder instead.",
    DeprecationWarning,
    stacklevel=2,
)


class FastEmbedder:
    """Text embedder with GPU support and persistent embedding cache."""
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        use_cache: bool = True
    ):
        """Initialize embedder. device: 'cuda', 'cpu', or None for auto-detect."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.device = device
        self.model_name = model_name
        
        self.use_cache = use_cache
        self.cache = EmbeddingCache() if use_cache else None
        self.cache_hits = 0
        self.cache_misses = 0
        
        if device == "cpu":
            torch.set_num_threads(torch.get_num_threads())


    def embed(self, text: str) -> List[float]:
        """Embed a single text string into a vector."""
        if self.cache:
            cached = self.cache.get(text)
            if cached is not None:
                self.cache_hits += 1
                return cached
            self.cache_misses += 1
        
        vector = self.model.encode(
            text, 
            convert_to_tensor=False,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        embedding = vector.tolist()
        
        if self.cache:
            self.cache.set(text, embedding)
        
        return embedding

    def embed_batch(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress: bool = False
    ) -> List[List[float]]:
        """Embed multiple texts. Uses cache when available."""
        if len(texts) == 0:
            return []
        
        results = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []
        
        if self.cache:
            for i, text in enumerate(texts):
                cached = self.cache.get(text)
                if cached is not None:
                    results[i] = cached
                    self.cache_hits += 1
                else:
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
                    self.cache_misses += 1
        else:
            texts_to_embed = texts
            indices_to_embed = list(range(len(texts)))
        
        if len(texts_to_embed) == 0:
            return results
            
        if self.device == "cuda" and batch_size < 64:
            batch_size = 64
            
        vectors = self.model.encode(
            texts_to_embed,
            batch_size=batch_size,
            convert_to_tensor=False,
            show_progress_bar=show_progress,
            normalize_embeddings=True
        )
        
        for idx, (text, vector) in enumerate(zip(texts_to_embed, vectors)):
            embedding = vector.tolist()
            results[indices_to_embed[idx]] = embedding
            
            if self.cache:
                self.cache.set(text, embedding)
        
        return results
    
    def get_cache_stats(self) -> Dict:
        """Return cache hit/miss statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': f"{hit_rate:.1f}%",
            'embeddings_saved': self.cache.stats() if self.cache else None
        }
    
    def clear_cache(self):
        """Clear all cached embeddings and reset statistics."""
        if self.cache:
            self.cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0


def benchmark_embedder(model_name: str = "all-MiniLM-L6-v2", num_texts: int = 100):
    """Benchmark embedding speed on CPU, GPU, and with cache."""
    import time
    
    texts = ["This is a test sentence for benchmarking."] * num_texts
    
    print("\n" + "="*60)
    print("EMBEDDER PERFORMANCE BENCHMARK")
    print("="*60)
    
    print("\nCPU Test (no cache):")
    embedder_cpu = FastEmbedder(model_name, device="cpu", use_cache=False)
    start = time.time()
    embedder_cpu.embed_batch(texts)
    cpu_time = time.time() - start
    print(f"   Time: {cpu_time:.2f}s for {num_texts} texts")
    print(f"   Speed: {num_texts/cpu_time:.1f} texts/second")
    
    if torch.cuda.is_available():
        print("\nGPU Test (no cache):")
        embedder_gpu = FastEmbedder(model_name, device="cuda", use_cache=False)
        start = time.time()
        embedder_gpu.embed_batch(texts)
        gpu_time = time.time() - start
        print(f"   Time: {gpu_time:.2f}s for {num_texts} texts")
        print(f"   Speed: {num_texts/gpu_time:.1f} texts/second")
        print(f"\n   GPU is {cpu_time/gpu_time:.1f}x faster")
    
    print("\nCache Test:")
    embedder_cached = FastEmbedder(model_name, device="cpu", use_cache=True)
    embedder_cached.embed_batch(texts)
    
    start = time.time()
    embedder_cached.embed_batch(texts)
    cache_time = time.time() - start
    print(f"   Cached time: {cache_time:.4f}s for {num_texts} texts")
    print(f"   Cache is {cpu_time/cache_time:.0f}x faster than CPU")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    benchmark_embedder()
