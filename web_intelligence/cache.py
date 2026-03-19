import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta


class URLCache:
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "index.json"
        self.index = self._load_index()
        
    def _load_index(self) -> Dict:
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _url_hash(self, url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()[:16]
    
    def is_cached(self, url: str, ttl_hours: Optional[int] = None) -> bool:
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
        url_hash = self._url_hash(url)
        return self.index.get(url_hash)
    
    def set(self, url: str, data: Dict[str, Any]):
        url_hash = self._url_hash(url)
        
        self.index[url_hash] = {
            'url': url,
            'cached_at': datetime.now().isoformat(),
            'data': data
        }
        
        self._save_index()
    
    def delete(self, url: str):
        url_hash = self._url_hash(url)
        if url_hash in self.index:
            del self.index[url_hash]
            self._save_index()
    
    def clear(self):
        self.index = {}
        self._save_index()
    
    def stats(self) -> Dict[str, Any]:
        return {
            'total_cached': len(self.index),
            'cache_dir': str(self.cache_dir),
            'index_size_bytes': self.index_file.stat().st_size if self.index_file.exists() else 0
        }


class ContentCache:
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
        return hashlib.sha256(text.encode()).hexdigest()[:32]
    
    def is_duplicate(self, text: str) -> bool:
        content_hash = self.get_content_hash(text)
        return content_hash in self.hashes
    
    def mark_as_indexed(self, text: str, doc_id: str, url: str):
        content_hash = self.get_content_hash(text)
        self.hashes[content_hash] = {
            'doc_id': doc_id,
            'url': url,
            'indexed_at': datetime.now().isoformat()
        }
        self._save_hashes()
    
    def get_existing(self, text: str) -> Optional[Dict]:
        content_hash = self.get_content_hash(text)
        return self.hashes.get(content_hash)
    
    def clear(self):
        self.hashes = {}
        self._save_hashes()


class EmbeddingCache:
    
    def __init__(self, cache_dir: str = "./data/cache/embeddings", max_memory_items: int = 10000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        self.max_memory_items = max_memory_items
        
    def _text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[List[float]]:
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
        text_hash = self._text_hash(text)
        
        if len(self.cache) < self.max_memory_items:
            self.cache[text_hash] = embedding
        
        cache_file = self.cache_dir / f"{text_hash[:16]}.json"
        with open(cache_file, 'w') as f:
            json.dump(embedding, f)
    
    def get_batch(self, texts: List[str]) -> Dict[str, Optional[List[float]]]:
        results = {}
        for text in texts:
            results[text] = self.get(text)
        return results
    
    def set_batch(self, texts: List[str], embeddings: List[List[float]]):
        for text, embedding in zip(texts, embeddings):
            self.set(text, embedding)
    
    def clear(self):
        self.cache = {}
        for file in self.cache_dir.glob("*.json"):
            file.unlink()
    
    def stats(self) -> Dict:
        disk_count = len(list(self.cache_dir.glob("*.json")))
        return {
            'memory_cached': len(self.cache),
            'disk_cached': disk_count,
            'cache_dir': str(self.cache_dir)
        }
