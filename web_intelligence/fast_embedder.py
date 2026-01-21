"""
Fast text embedding with GPU acceleration and caching support.

This module provides efficient text-to-vector conversion using sentence transformers.
Embeddings are 384-dimensional vectors that enable semantic search and similarity
comparisons. GPU acceleration provides 10-50x speedup over CPU-only processing.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict
import torch
from .cache import EmbeddingCache


class FastEmbedder:
    """
    High-performance text embedding with GPU support and intelligent caching.
    
    Converts text into numerical vector representations (embeddings) for semantic
    search and similarity analysis. Supports GPU acceleration and caches computed
    embeddings to avoid redundant calculations.
    
    Performance characteristics:
    - CPU only: ~500ms per 100 texts
    - GPU accelerated: ~50ms per 100 texts (10x faster)
    - Cached results: <1ms (instant retrieval)
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        use_cache: bool = True
    ):
        """
        Initialize the embedding model with specified configuration.
        
        Args:
            model_name: SentenceTransformer model identifier. Options:
                - "all-MiniLM-L6-v2": Fast, 384-dim vectors (recommended)
                - "all-MiniLM-L12-v2": Higher quality, slower
            device: Compute device specification:
                - "cuda": Force GPU usage
                - "cpu": Force CPU usage
                - None: Auto-detect (prefers GPU if available)
            use_cache: Enable persistent caching of computed embeddings
        """
        print(f"Loading embedding model: {model_name}...")
        
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            
        if device == "cuda" and torch.cuda.is_available():
            print(f"✓ GPU detected! Using CUDA for 10-50x speedup")
        else:
            print(f"Using CPU (consider GPU for better performance)")
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
        
        print(f"✓ Model loaded ({self.dimension} dimensions, device: {device})")
        if use_cache:
            print(f"✓ Embedding cache enabled")

    def embed(self, text: str) -> List[float]:
        """
        Convert a single text string into an embedding vector.
        
        Args:
            text: Input text to convert
            
        Returns:
            384-dimensional embedding vector as list of floats
        """
        if self.cache:
            cached = self.cache.get(text)
            if cached is not None:
                self.cache_hits += 1
                return cached
            self.cache_misses += 1
        
        vector = self.model.encode(
            text, 
            convert_to_tensor=False,
            show_progress_bar=False
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
        """
        Convert multiple texts into embedding vectors efficiently.
        
        Batch processing is significantly faster than individual conversions
        due to GPU parallelization and reduced overhead.
        
        Args:
            texts: List of text strings to convert
            batch_size: Number of texts to process simultaneously
            show_progress: Display progress bar for large batches
            
        Returns:
            List of embedding vectors, one per input text
        """
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
        """
        Retrieve cache performance statistics.
        
        Returns:
            Dictionary containing cache hits, misses, hit rate percentage,
            and total number of cached embeddings.
        """
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
    """
    Benchmark embedding performance on current hardware.
    
    Tests CPU, GPU (if available), and cache performance to measure
    throughput and speedup factors.
    """
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
