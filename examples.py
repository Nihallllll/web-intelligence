"""
Two main functions for web data extraction using our crawler:
1. extract_url_data() - Crawls a URL and returns raw extracted content
2. get_cleaned_data() - Retrieves indexed/cleaned data from vector store
"""

from web_intelligence.crawler import crawl_url
from web_intelligence.extractor import extract_content
from web_intelligence import FastPipeline


def extract_url_data(url):
    """
    Crawl a URL and extract raw content using our crawler.
    
    Args:
        url: The URL to crawl and extract
        
    Returns:
        dict with:
            - success: bool
            - url: str
            - title: str
            - text: str (raw extracted text)
            - word_count: int
            - html_length: int
            - status_code: int
            - error: str (if failed)
    """
    print(f"Crawling {url}...")
    
    # Use our crawler with proper headers
    crawl_result = crawl_url(url)
    
    if not crawl_result.success:
        return {
            'success': False,
            'url': url,
            'error': crawl_result.error or f'HTTP {crawl_result.status_code}',
            'status_code': crawl_result.status_code
        }
    
    print(f"✓ Crawled successfully ({len(crawl_result.html):,} chars)")
    
    # Extract clean content
    extracted = extract_content(crawl_result.html, url)
    
    if not extracted.text:
        return {
            'success': False,
            'url': url,
            'error': 'No content extracted from HTML'
        }
    
    print(f"✓ Extracted: {extracted.title}")
    print(f"  Word count: {extracted.word_count:,}")
    
    return {
        'success': True,
        'url': url,
        'title': extracted.title,
        'text': extracted.text,
        'word_count': extracted.word_count,
        'html_length': len(crawl_result.html),
        'status_code': crawl_result.status_code,
        'published_date': extracted.published_date
    }


def get_cleaned_data(url, pipeline=None):
    """
    Get cleaned and chunked data from the vector store.
    Automatically indexes the URL if not already indexed.
    
    Args:
        url: The URL to retrieve cleaned data for
        pipeline: Optional FastPipeline instance (creates one if not provided)
        
    Returns:
        dict with:
            - success: bool
            - url: str
            - title: str
            - chunks: list of chunk dicts
            - full_text: str (all chunks combined)
            - total_words: int
            - total_chunks: int
            - indexed_at: str
    """
    if pipeline is None:
        pipeline = FastPipeline()
    
    # Check if URL is already indexed
    collection = pipeline.vector_store.collection
    results = collection.get(where={"url": url})
    
    # If not indexed, index it now
    if len(results['ids']) == 0:
        print(f"URL not indexed. Indexing now...")
        index_result = pipeline.index_url(url)
        
        if not index_result['success']:
            return {
                'success': False,
                'url': url,
                'error': index_result.get('error', 'Failed to index')
            }
        
        print(f"✓ Indexed successfully")
        # Retrieve again
        results = collection.get(where={"url": url})
    
    # Build chunks list
    chunks = []
    for i in range(len(results['ids'])):
        chunks.append({
            'chunk_id': results['ids'][i],
            'text': results['metadatas'][i]['text'],
            'chunk_index': results['metadatas'][i]['chunk_index'],
            'title': results['metadatas'][i]['title'],
            'url': results['metadatas'][i]['url'],
            'word_count': results['metadatas'][i]['word_count'],
            'indexed_at': results['metadatas'][i]['indexed_at']
        })
    
    # Sort by chunk_index to maintain order
    chunks.sort(key=lambda x: x['chunk_index'])
    
    # Combine all chunks into full text
    full_text = "\n\n".join([c['text'] for c in chunks])
    
    return {
        'success': True,
        'url': url,
        'title': chunks[0]['title'] if chunks else '',
        'chunks': chunks,
        'full_text': full_text,
        'total_words': sum(c['word_count'] for c in chunks),
        'total_chunks': len(chunks),
        'indexed_at': chunks[0]['indexed_at'] if chunks else None
    }


# Example usage
if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/Nostradamus"
    
    print("="*60)
    print("FUNCTION 1: Extract URL Data (Raw Crawling)")
    print("="*60)
    
    # Function 1: Extract raw data using our crawler
    raw_data = extract_url_data(url)
    
    if raw_data['success']:
        print(f"\n✅ Successfully extracted data:")
        print(f"   Title: {raw_data['title']}")
        print(f"   Word count: {raw_data['word_count']:,}")
        print(f"   HTML size: {raw_data['html_length']:,} bytes")
        print(f"\n   First 300 chars:")
        print(f"   {raw_data['text'][:300]}...")
    else:
        print(f"\n❌ Failed: {raw_data['error']}")
        exit(1)
    
    print(f"\n{'='*60}")
    print("FUNCTION 2: Get Cleaned Data (Indexed/Chunked)")
    print("="*60)
    
    # Function 2: Get cleaned/chunked data from vector store
    cleaned_data = get_cleaned_data(url)
    
    if cleaned_data['success']:
        print(f"\n✅ Retrieved cleaned data:")
        print(f"   Title: {cleaned_data['title']}")
        print(f"   Total chunks: {cleaned_data['total_chunks']}")
        print(f"   Total words: {cleaned_data['total_words']:,}")
        print(f"   Total characters: {len(cleaned_data['full_text']):,}")
        print(f"   Indexed at: {cleaned_data['indexed_at']}")
        
        print(f"\n   First 5 chunks:")
        for chunk in cleaned_data['chunks'][:5]:
            print(f"\n   --- Chunk {chunk['chunk_index']} ({chunk['word_count']} words) ---")
            print(f"   {chunk['text'][:200]}...")
        
        if cleaned_data['total_chunks'] > 5:
            print(f"\n   ... and {cleaned_data['total_chunks'] - 5} more chunks")
        
        # Optional: Save to file
        save_to_file = input("\n\nSave full cleaned text to file? (y/n): ").lower().strip()
        if save_to_file == 'y':
            filename = "cleaned_data.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Title: {cleaned_data['title']}\n")
                f.write(f"URL: {url}\n")
                f.write(f"Indexed: {cleaned_data['indexed_at']}\n")
                f.write(f"Total Words: {cleaned_data['total_words']:,}\n")
                f.write(f"Total Chunks: {cleaned_data['total_chunks']}\n")
                f.write("="*60 + "\n\n")
                f.write(cleaned_data['full_text'])
            print(f"✅ Saved to {filename}")
    else:
        print(f"\n❌ Failed: {cleaned_data['error']}")