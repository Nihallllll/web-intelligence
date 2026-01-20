from embedder import Embedder
from vector_store import VectorStore
from crawler import crawl_url
from chunker import chunk_text
from extractor import extract_content
from typing import List
import uuid
class IndexingPipeLine:
    def __init__(self):
        self.embedder =  Embedder()
        self.vector_store = VectorStore()
    
    def index_url(self , url:str):
        print("Crawling URL")
        result = crawl_url(url)
        if not result.success:
            print(f"Failed to crawl due to : {result.error}")
            return
        
        print("Extracting Content...")
        extracted_result = extract_content(result.html,url)
        if extracted_result.text == "":
            print("No content extracted")
            return
        
        print(f"Chunking {extracted_result.word_count} number of chunks")
        unique_id = str(uuid.uuid4())
        chunks = chunk_text(extracted_result.text,unique_id,url)

        print(f"Embedding {len(chunks)} number of chunks")
        chunk_texts = [c.text for c in chunks] 
        vectors = self.embedder.embed_batch(chunk_texts)

        print("Storing in the Vector DB...")
        metadatas = [{
            "text":c.text,
            "chunk_index": c.chunk_index,
            "title":extracted_result.title,
            "url" : c.url
        }for c in chunks]
        ids = [f"{unique_id}_chunk_{i}" for i in range(len(chunks))]
        
        self.vector_store.add(vectors, metadatas, ids)
        print(f"✓ Indexed {url} ({len(chunks)} chunks)")
    
    def view_content(self, limit=None):
        """View stored content in the vector database"""
        results = self.vector_store.collection.get(
            limit=limit,
            include=["metadatas", "documents"]
        )
        
        print(f"\n{'='*80}")
        print(f"VECTOR STORE CONTENT ({len(results['ids'])} items)")
        print(f"{'='*80}\n")
        
        for i, (id, metadata) in enumerate(zip(results['ids'], results['metadatas'])):
            print(f"[{i+1}] ID: {id}")
            print(f"    Title: {metadata.get('title', 'N/A')}")
            print(f"    URL: {metadata.get('url', 'N/A')}")
            print(f"    Chunk Index: {metadata.get('chunk_index', 'N/A')}")
            print(f"    Text Preview: {metadata.get('text', '')[:200]}...")
            print()



pipeline =IndexingPipeLine()
pipeline.index_url("https://sitecorediaries.org/about/")

# View the stored content
pipeline.view_content(limit=5)  # Show first 5 chunks, remove limit to see all