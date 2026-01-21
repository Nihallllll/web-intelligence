# src/vector_store.py
import chromadb
from typing import List, Dict, Optional

class VectorStore:
    def __init__(self, persist_directory: str = "./data/chroma"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection("web_content")
    
    def add(self, vectors: List[List[float]], metadatas: List[Dict], ids: List[str]):
        """Add vectors to the store."""
        self.collection.add(
            embeddings=vectors,
            metadatas=metadatas,
            ids=ids
        )
    
    def search(self, query_vector: List[float], limit: int = 5, 
               filter: Optional[Dict] = None) -> List[Dict]:
        """Search for similar vectors."""
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            where=filter
        )
        
        # Format results
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                'id': results['ids'][0][i],
                'text': results['metadatas'][0][i].get('text', ''),
                'source': results['metadatas'][0][i].get('source_url', ''),
                'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                'metadata': results['metadatas'][0][i]
            })
        
        return formatted