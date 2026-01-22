"""
Examples of using Web Intelligence with LangChain for RAG (Retrieval-Augmented Generation)
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from web_intelligence import FastPipeline
from dotenv import load_dotenv

load_dotenv()

# Initialize components
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # Stable model with better rate limits
    temperature=0.7
)
pipeline = FastPipeline(cache_enabled=True, use_gpu=None)


# Example 1: Simple RAG - Index a URL and answer questions about it
def example_simple_rag():
    """Index a webpage and answer questions using its content."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Simple RAG")
    print("="*60)
    
    # Index a webpage
    url = "https://www.python.org/about/"
    print(f"\nIndexing: {url}")
    result = pipeline.index_url(url)
    print(f"✓ Indexed: {result['title']}, Chunks: {result['chunks_count']}")
    
    # Ask a question
    query = "What is Python used for?"
    print(f"\nQuestion: {query}")
    
    # Search indexed content
    search_results = pipeline.search(query, limit=2)  # Reduced from 3
    
    # Build context from search results (truncate to save tokens)
    context = "\n\n".join([r['text'][:500] + "..." for r in search_results])
    
    # Ask LLM with context
    prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
    
    response = llm.invoke(prompt)
    print(f"\nAnswer: {response.content}")


# Example 2: Multi-URL RAG - Index multiple sources
def example_multi_source_rag():
    """Index multiple webpages and answer questions using all of them."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Multi-Source RAG")
    print("="*60)
    
    # Index multiple URLs
    urls = [
        "https://www.python.org/about/",
        "https://docs.python.org/3/tutorial/index.html",
    ]
    
    print(f"\nIndexing {len(urls)} URLs...")
    results = pipeline.index_batch(urls)
    
    for r in results:
        if r['success']:
            print(f"✓ {r['url']}: {r['chunks_count']} chunks")
    
    # Ask a question
    query = "How do I get started with Python?"
    print(f"\nQuestion: {query}")
    
    # Search across all indexed content
    search_results = pipeline.search(query, limit=3)  # Reduced from 5
    
    # Build context with sources (truncate to save tokens)
    context_parts = []
    for i, r in enumerate(search_results, 1):
        context_parts.append(f"[Source {i}] {r['metadata']['url']}\n{r['text'][:400]}...")
    
    context = "\n\n".join(context_parts)
    
    # Ask LLM with context and sources
    prompt = f"""Based on the following sources, answer the question. Include source numbers in your answer.

{context}

Question: {query}

Answer with citations:"""
    
    response = llm.invoke(prompt)
    print(f"\nAnswer:\n{response.content}")


# Example 3: Conversational RAG - Chat with your indexed content
def example_conversational_rag():
    """Have a conversation with the LLM using indexed content."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Conversational RAG")
    print("="*60)
    
    # Index content (using cached if already indexed)
    url = "https://www.python.org/about/"
    pipeline.index_url(url)
    
    # Conversation loop
    conversation_history = []
    
    questions = [
        "What is Python?",
        "What are its main features?",
        "Is it good for beginners?"
    ]
    
    for question in questions:
        print(f"\nUser: {question}")
        
        # Search for relevant content
        search_results = pipeline.search(question, limit=3)
        context = "\n\n".join([r['text'] for r in search_results])
        
        # Build conversation context
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
        
        prompt = f"""You are a helpful assistant. Use the provided context to answer questions.

Previous conversation:
{history_text if history_text else "None"}

Current context:
{context}

User question: {question}

Answer:"""
        
        response = llm.invoke(prompt)
        print(f"Assistant: {response.content}")
        
        # Add to conversation history
        conversation_history.append({"role": "user", "content": question})
        conversation_history.append({"role": "assistant", "content": response.content})


# Example 4: Build a custom chatbot class
class WebIntelligenceChatbot:
    """A chatbot that uses indexed web content to answer questions."""
    
    def __init__(self, llm, pipeline):
        self.llm = llm
        self.pipeline = pipeline
        self.conversation_history = []
    
    def index_urls(self, urls):
        """Index multiple URLs for the chatbot to use."""
        print(f"Indexing {len(urls)} URLs...")
        results = self.pipeline.index_batch(urls)
        successful = sum(1 for r in results if r['success'])
        failed = [r for r in results if not r['success']]
        
        print(f"✓ Successfully indexed {successful}/{len(urls)} URLs")
        
        # Show failed URLs
        if failed:
            print("\n⚠ Failed to index:")
            for r in failed:
                error = r.get('error', 'Unknown error')
                print(f"  - {r['url']}: {error}")
        
        return results
    
    def ask(self, question, num_results=5, debug=False):
        """Ask a question and get an answer based on indexed content."""
        # Search for relevant content
        search_results = self.pipeline.search(question, limit=num_results)
        
        if not search_results:
            return "I don't have enough information to answer that question. Please index some relevant web pages first."
        
        # Debug: show what chunks were found
        if debug:
            print("\n[DEBUG] Found chunks:")
            for i, r in enumerate(search_results, 1):
                print(f"  {i}. Score: {r.get('score', 0):.3f} | {r['text'][:100]}...")
            print()
        
        # Build context from search results (full text, no truncation)
        context_parts = []
        for i, r in enumerate(search_results, 1):
            source = r['metadata']['url']
            text = r['text']  # Use full text
            score = r.get('score', 0)
            context_parts.append(f"[Source {i}] (Relevance: {score:.2f})\n{source}\n{text}")
        
        context = "\n\n".join(context_parts)
        
        # Build prompt with conversation history
        history_text = ""
        if self.conversation_history:
            history_text = "Previous conversation:\n" + "\n".join([
                f"{msg['role'].title()}: {msg['content']}" 
                for msg in self.conversation_history[-4:]  # Last 2 exchanges
            ]) + "\n\n"
        
        prompt = f"""You are a helpful assistant. Answer the question based on the provided sources.
Be concise and accurate. If you use information from the sources, mention which source number.

{history_text}Relevant sources:
{context}

User question: {question}

Answer:"""
        
        # Get response from LLM
        response = self.llm.invoke(prompt)
        answer = response.content
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})
        
        return answer
    
    def get_stats(self):
        """Get statistics about indexed content."""
        return self.pipeline.stats()
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("Conversation history cleared.")


def example_chatbot_class():
    """Use the custom chatbot class."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Custom Chatbot Class")
    print("="*60)
    
    # Create chatbot
    chatbot = WebIntelligenceChatbot(llm, pipeline)
    
    # Index some URLs
    urls = [
        "https://www.python.org/about/",
        "https://developers.google.com/merchant/ucp",
    ]
    chatbot.index_urls(urls)
    
    # Ask questions
    questions = [
        "What is Python?",
        "what is ucp?",
        "What are its key features?"
    ]
    
    for question in questions:
        print(f"\nUser: {question}")
        answer = chatbot.ask(question, debug=True)
        print(f"Bot: {answer}")
    
    # Show stats
    print("\n" + "-"*60)
    stats = chatbot.get_stats()
    print(f"Total chunks indexed: {stats['total_chunks_in_database']}")
    print(f"Device: {stats['device']}")


# Example 5: Real-time web research assistant
def example_research_assistant():
    """Research a topic by indexing relevant URLs and answering questions."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Research Assistant")
    print("="*60)
    
    topic = "machine learning"
    
    # You could programmatically find URLs (e.g., from search API)
    # For demo, using predefined URLs
    urls = [
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://www.python.org/about/",
    ]
    
    print(f"\nResearching topic: {topic}")
    print(f"Indexing {len(urls)} sources...")
    
    results = pipeline.index_batch(urls)
    print(f"✓ Indexed {sum(1 for r in results if r['success'])} pages")
    
    # Generate a summary
    search_results = pipeline.search(topic, limit=5)
    context = "\n\n".join([r['text'] for r in search_results[:3]])
    
    prompt = f"""Based on the following information, provide a comprehensive summary about {topic}.
Include key points and important details and at the last tell me who created python and also tell me the url you are referring to.

Information:
{context}

Summary:"""
    
    response = llm.invoke(prompt)
    print(f"\nSummary of '{topic}':")
    print(response.content)


if __name__ == "__main__":
    # Run examples
    print("WEB INTELLIGENCE + LANGCHAIN EXAMPLES")
    print("="*60)
    
    # Uncomment the examples you want to run:
    
   #  example_simple_rag()
    # example_multi_source_rag()
    #example_conversational_rag()
    example_chatbot_class()
    # example_research_assistant()
