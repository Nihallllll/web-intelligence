from tavily import TavilyClient
import json

tavily_client = TavilyClient(api_key="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")
url = "https://en.wikipedia.org/wiki/Nostradamus"

print(f"Extracting content from: {url}\n")
response = tavily_client.extract(url)

# Display response structure
print("="*60)
print("TAVILY EXTRACT RESPONSE")
print("="*60)

# Check what's in the response
if 'results' in response and len(response['results']) > 0:
    result = response['results'][0]
    
    print(f"\nURL: {result.get('url', 'N/A')}")
    print(f"Title: {result.get('title', 'N/A')}")
    
    # Get the cleaned content
    raw_content = result.get('raw_content', '')
    
    print(f"\n{'='*60}")
    print(f"CLEANED DATA STATISTICS")
    print(f"{'='*60}")
    print(f"Total characters: {len(raw_content):,}")
    print(f"Total words: {len(raw_content.split()):,}")
    print(f"Total lines: {len(raw_content.splitlines()):,}")
    
    print(f"\n{'='*60}")
    print(f"FIRST 1000 CHARACTERS OF CLEANED CONTENT")
    print(f"{'='*60}")
    print(raw_content[:1000])
    print("\n...")
    
    print(f"\n{'='*60}")
    print(f"LAST 500 CHARACTERS OF CLEANED CONTENT")
    print(f"{'='*60}")
    print("...")
    print(raw_content[-500:])
    
    # Check for images if available
    if 'images' in result:
        print(f"\n{'='*60}")
        print(f"IMAGES FOUND: {len(result['images'])}")
        print(f"{'='*60}")
    
    # Save to file
    save_file = "tavily_cleaned_data.txt"
    with open(save_file, 'w', encoding='utf-8') as f:
        f.write(f"Title: {result.get('title', 'N/A')}\n")
        f.write(f"URL: {result.get('url', 'N/A')}\n")
        f.write("="*60 + "\n\n")
        f.write(raw_content)
    
    print(f"\n✅ Full cleaned content saved to: {save_file}")
    
else:
    print("No results found!")
    print("\nFull response:")
    print(json.dumps(response, indent=2))