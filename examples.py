"""Utility functions for extracting and retrieving web content."""

from web_intelligence.crawler import crawl_url
from web_intelligence.extractor import extract_content
from web_intelligence import FastPipeline


def extract_url_data(url):
    """Crawl a URL and return extracted text, title, and metadata."""
    crawl_result = crawl_url(url)
    
    if not crawl_result.success:
        return {
            'success': False,
            'url': url,
            'error': crawl_result.error or f'HTTP {crawl_result.status_code}',
            'status_code': crawl_result.status_code
        }
    

    extracted = extract_content(crawl_result.html, url)
    
    if not extracted.text:
        return {
            'success': False,
            'url': url,
            'error': 'No content extracted from HTML'
        }
    
    print(f"Extracted: {extracted.title} ({extracted.word_count:,} words)")
    
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
    """Return indexed/chunked content for a URL, indexing it first if needed."""
    if pipeline is None:
        pipeline = FastPipeline()
    
    collection = pipeline.vector_store.collection
    results = collection.get(where={"url": url})
    
    if len(results['ids']) == 0:
        index_result = pipeline.index_url(url)
        
        if not index_result['success']:
            return {
                'success': False,
                'url': url,
                'error': index_result.get('error', 'Failed to index')
            }
        
        print("Indexed successfully")
        results = collection.get(where={"url": url})
    
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
    
    chunks.sort(key=lambda x: x['chunk_index'])
    
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


if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/Nostradamus"
    
    raw_data = extract_url_data(url)
    
    if raw_data['success']:
        print(f"Title: {raw_data['title']}")
        print(f"Word count: {raw_data['word_count']:,}")
        print(f"HTML size: {raw_data['html_length']:,} bytes")
        print(f"\n{raw_data['text'][:300]}...")
    else:
        print(f"Failed: {raw_data['error']}")
        exit(1)
    
    cleaned_data = get_cleaned_data(url)
    
    if cleaned_data['success']:
        print(f"Title: {cleaned_data['title']}")
        print(f"Total chunks: {cleaned_data['total_chunks']}")
        print(f"Total words: {cleaned_data['total_words']:,}")
        print(f"Indexed at: {cleaned_data['indexed_at']}")
        
        for chunk in cleaned_data['chunks'][:5]:
            print(f"\nChunk {chunk['chunk_index']} ({chunk['word_count']} words):")
            print(f"{chunk['text'][:200]}...")
        
        if cleaned_data['total_chunks'] > 5:
            print(f"\n... and {cleaned_data['total_chunks'] - 5} more chunks")
        
        save_to_file = input("\nSave full cleaned text to file? (y/n): ").lower().strip()
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
            print(f"Saved to {filename}")
    else:
        print(f"Failed: {cleaned_data['error']}")