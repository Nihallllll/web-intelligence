"""
Caching system for URL deduplication, content detection, and embedding storage.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta


class URLCache:
    """
    Persistent cache for tracking processed URLs with TTL support.
    """
    
    def __init__(self, cache_dir: str = "./data/cache"):
        """Initialize URL cache with persistent storage in cache_dir."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "index.json"
        self.index = self._load_index()
        
    def _load_index(self) -> Dict:
        """Load and return the cache index from disk."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        """Persist cache index to disk."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _url_hash(self, url: str) -> str:
        """Return a 16-char SHA-256 hash for a URL."""
        return hashlib.sha256(url.encode()).hexdigest()[:16]
    
    def is_cached(self, url: str, ttl_hours: Optional[int] = None) -> bool:
        """Return True if url is cached and not expired (ttl_hours=None means no expiry)."""
        url_hash = self._url_hash(url)
        
        if url_hash not in self.index:
            return False
        
        if ttl_hours is not None:
            cached_at = datetime.fromisoformat(self.index[url_hash]['cached_at'])
            expiry = cached_at + timedelta(hours=ttl_hours)
            if datetime.now() > expiry:
                  return False
        
        return True
    
    def get(self, url: str) -> Optional[Dict[str, Any]]:
        """Return cached data for a URL, or None."""
        url_hash = self._url_hash(url)
        return self.index.get(url_hash)
    
    def set(self, url: str, data: Dict[str, Any]):
        """Store JSON-serializable data for a URL."""
        url_hash = self._url_hash(url)
        
        self.index[url_hash] = {
            'url': url,
            'cached_at': datetime.now().isoformat(),
            'data': data
        }
        
        self._save_index()
    
    def delete(self, url: str):
        """Remove a URL from cache."""
        url_hash = self._url_hash(url)
        if url_hash in self.index:
            del self.index[url_hash]
            self._save_index()
    
    def clear(self):
        """Clear all cached entries."""
        self.index = {}
        self._save_index()
    
    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return {
            'total_cached': len(self.index),
            'cache_dir': str(self.cache_dir),
            'index_size_bytes': self.index_file.stat().st_size if self.index_file.exists() else 0
        }


class ContentCache:
    """
    Content-based deduplication cache using SHA-256 hashing.
    
    Detects duplicate content across different URLs by comparing content hashes.
    """
    
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.content_file = self.cache_dir / "content_hashes.json"
        self.hashes = self._load_hashes()
    
    def _load_hashes(self) -> Dict:
        if self.content_file.exists():
            with open(self.content_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_hashes(self):
        with open(self.content_file, 'w') as f:
            json.dump(self.hashes, f, indent=2)
    
    def get_content_hash(self, text: str) -> str:
        """Return a 32-char SHA-256 hash of the text."""
        return hashlib.sha256(text.encode()).hexdigest()[:32]
    
    def is_duplicate(self, text: str) -> bool:
        """Return True if this text has already been indexed."""
        content_hash = self.get_content_hash(text)
        return content_hash in self.hashes
    
    def mark_as_indexed(self, text: str, doc_id: str, url: str):
        """Record that this text content has been indexed."""
        content_hash = self.get_content_hash(text)
        self.hashes[content_hash] = {
            'doc_id': doc_id,
            'url': url,
            'indexed_at': datetime.now().isoformat()
        }
        self._save_hashes()
    
    def get_existing(self, text: str) -> Optional[Dict]:
        """Return existing index entry for text, or None."""
        content_hash = self.get_content_hash(text)
        return self.hashes.get(content_hash)
    
    def clear(self):
        """Clear all content hashes."""
        self.hashes = {}
        self._save_hashes()


class EmbeddingCache:
    """
    Two-tier cache for text embeddings with memory and disk storage.
    
    Caches computed embeddings to avoid redundant model inference.
    Uses in-memory cache for fast access and disk storage for persistence.
    """
    
    def __init__(self, cache_dir: str = "./data/cache/embeddings", max_memory_items: int = 10000):
        """Initialize embedding cache with memory and disk tiers."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        self.max_memory_items = max_memory_items
        
    def _text_hash(self, text: str) -> str:
        """Generate SHA-256 hash for text content."""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[List[float]]:
        """Retrieve embedding for text from memory or disk. Returns None if not cached."""
        text_hash = self._text_hash(text)
        
        if text_hash in self.cache:
            return self.cache[text_hash]
        
        cache_file = self.cache_dir / f"{text_hash[:16]}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                embedding = json.load(f)
                if len(self.cache) < self.max_memory_items:
                    self.cache[text_hash] = embedding
                return embedding
        
        return None
    
    def set(self, text: str, embedding: List[float]):
        """Store embedding in memory and disk cache."""
        text_hash = self._text_hash(text)
        
        if len(self.cache) < self.max_memory_items:
            self.cache[text_hash] = embedding
        
        cache_file = self.cache_dir / f"{text_hash[:16]}.json"
        with open(cache_file, 'w') as f:
            json.dump(embedding, f)
    
    def get_batch(self, texts: List[str]) -> Dict[str, Optional[List[float]]]:
        """Retrieve embeddings for multiple texts."""
        results = {}
        for text in texts:
            results[text] = self.get(text)
        return results
    
    def set_batch(self, texts: List[str], embeddings: List[List[float]]):
        """Store embeddings for multiple texts."""
        for text, embedding in zip(texts, embeddings):
            self.set(text, embedding)
    
    def clear(self):
        """Clear all cached embeddings from memory and disk."""
        self.cache = {}
        for file in self.cache_dir.glob("*.json"):
            file.unlink()
    
    def stats(self) -> Dict:
        """Return memory and disk cache item counts."""
        disk_count = len(list(self.cache_dir.glob("*.json")))
        return {
            'memory_cached': len(self.cache),
            'disk_cached': disk_count,
            'cache_dir': str(self.cache_dir)
        }
