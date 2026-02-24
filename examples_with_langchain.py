"""Web Intelligence + LangChain RAG examples."""

from langchain_google_genai import ChatGoogleGenerativeAI
from web_intelligence import FastPipeline
from dotenv import load_dotenv
from litellm import completion
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
    
    # Index a webpage
    url = "https://en.wikipedia.org/wiki/Nostradamus"
    result = pipeline.index_url(url)
    print(f"Indexed: {result['title']}, Chunks: {result['chunks_count']}")
    
    # Ask a question
    query = "who is nostradamus and tell me his 5 predictions"
    print(f"Question: {query}")
    
    # Search indexed content
    search_results = pipeline.search(query, limit=2)
    
    # Build context from search results (truncate to save tokens)
    context = "\n\n".join([r['text'][:500] + "..." for r in search_results])
    prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
    
    # response = llm.invoke(prompt)
    response = completion(
        model="groq/openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"\nAnswer: {response.choices[0].message.content}")


# Example 2: Multi-URL RAG - Index multiple sources
def example_multi_source_rag():
    """Index multiple webpages and answer a question across all of them."""
    
    # Index multiple URLs
    urls = [
        "https://www.python.org/about/",
        "https://docs.python.org/3/tutorial/index.html",
    ]
    
    results = pipeline.index_batch(urls)
    
    for r in results:
        if r['success']:
            print(f"✓ {r['url']}: {r['chunks_count']} chunks")
    
    # Ask a question
    query = "How do I get started with Python?"
    print(f"Question: {query}")
    
    search_results = pipeline.search(query, limit=3)
    
    # Build context with sources (truncate to save tokens)
    context_parts = []
    for i, r in enumerate(search_results, 1):
        context_parts.append(f"[Source {i}] {r['metadata']['url']}\n{r['text'][:400]}...")
    prompt = f"""Based on the following sources, answer the question. Include source numbers in your answer.

{context}

Question: {query}

Answer with citations:"""
    
    response = llm.invoke(prompt)
    print(f"\nAnswer:\n{response.content}")


# Example 3: Conversational RAG - Chat with your indexed content
def example_conversational_rag():
    """Have a conversation with the LLM using indexed content."""
    
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
        
        conversation_history.append({"role": "user", "content": question})
        conversation_history.append({"role": "assistant", "content": response.content})


class WebIntelligenceChatbot:
    """A chatbot that uses indexed web content to answer questions."""
    
    def __init__(self, llm, pipeline):
        self.llm = llm
        self.pipeline = pipeline
        self.conversation_history = []
    
    def index_urls(self, urls):
        """Index multiple URLs."""
        results = self.pipeline.index_batch(urls)
        successful = sum(1 for r in results if r['success'])
        failed = [r for r in results if not r['success']]
        
        print(f"Indexed {successful}/{len(urls)} URLs")
        if failed:
            for r in failed:
                print(f"  Failed: {r['url']}: {r.get('error', 'Unknown error')}")
        return results
    
    def ask(self, question, num_results=5, debug=False):
        """Answer a question from indexed content."""
        search_results = self.pipeline.search(question, limit=num_results)
        
        if not search_results:
            return "I don't have enough information to answer that question. Please index some relevant web pages first."
        
        # Debug: show what chunks were found
        if debug:
            print("\n[DEBUG] Found chunks:")
            for i, r in enumerate(search_results, 1):
                print(f"  {i}. Score: {r.get('score', 0):.3f} | {r['text'][:100]}...")
            print()
        
        context_parts = []
        for i, r in enumerate(search_results, 1):
            source = r['metadata']['url']
            text = r['text']
            score = r.get('score', 0)
            context_parts.append(f"[Source {i}] (Relevance: {score:.2f})\n{source}\n{text}")
        
        context = "\n\n".join(context_parts)
        
        history_text = ""
        if self.conversation_history:
            history_text = "Previous conversation:\n" + "\n".join([
                f"{msg['role'].title()}: {msg['content']}" 
                for msg in self.conversation_history[-4:]
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
    print(f"Total chunks: {stats['total_chunks_in_database']}, Device: {stats['device']}")


# Example 5: Real-time web research assistant
def example_research_assistant():
    """Research a topic by indexing relevant URLs and summarizing."""
    print("\nResearching topic: machine learning")
    
    topic = "machine learning"
    urls = [
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://www.python.org/about/",
    ]
    results = pipeline.index_batch(urls)
    print(f"Indexed {sum(1 for r in results if r['success'])} pages")
    
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
    example_simple_rag()
    # example_multi_source_rag()
    # example_conversational_rag()
    # example_chatbot_class()
    # example_research_assistant()
