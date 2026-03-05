"""
Example: Web Intelligence + Groq LLM

This script shows how to:
  1. Search the web for a topic (no URL needed)
  2. Get formatted context from Web Intelligence
  3. Feed that context to Groq's LLM via LangChain
  4. Get an AI answer grounded in real web data

Setup:
  1. Add your Groq API key to .env file:
       GROQ_API_KEY=gsk_...
  2. Install dependencies:
       pip install langchain-groq python-dotenv
  3. Run:
       python example_use.py
"""

from dotenv import load_dotenv

# Load .env file so GROQ_API_KEY is available
load_dotenv()

from web_intelligence import FastPipeline
from langchain_groq import ChatGroq


def main():
    # ─────────────────────────────────────────────────────
    # Step 1: Create the pipeline
    # ─────────────────────────────────────────────────────
    print("Setting up Web Intelligence pipeline...")
    pipeline = FastPipeline()

    # ─────────────────────────────────────────────────────
    # Step 2: Search the web — library does everything
    #   - Searches DuckDuckGo for your question
    #   - Crawls the top results
    #   - Extracts clean text
    #   - Chunks, embeds, and stores it
    #   - Retrieves the most relevant pieces
    # ─────────────────────────────────────────────────────
    question = "What is FastAPI and why is it popular for building APIs in Python?"

    print(f"\nSearching the web for: '{question}'")
    print("(This searches DuckDuckGo → crawls pages → indexes → retrieves)\n")

    ctx = pipeline.search_web(question, max_results=3, limit=5)

    # Show what the library found
    print("=" * 60)
    print("WEB INTELLIGENCE RESULTS")
    print("=" * 60)
    print(f"Sources found: {len(ctx.sources)}")
    for s in ctx.sources:
        print(f"  - {s['title']}: {s['url']}")
    print(f"Context chunks: {ctx.total_chunks}")
    print(f"Context words:  {ctx.total_words}")
    print(f"\nContext preview (first 500 chars):")
    print(ctx.context_text[:500])
    print("...\n")

    # ─────────────────────────────────────────────────────
    # Step 3: Send context to Groq LLM
    #
    # ctx.as_messages() gives you OpenAI-compatible messages:
    #   [
    #     {"role": "system", "content": "You are a helpful assistant..."},
    #     {"role": "user", "content": "Context: ...\nQuestion: ..."}
    #   ]
    #
    # This works with ANY LLM that accepts this format:
    #   - Groq, OpenAI, Anthropic, Ollama, LiteLLM, etc.
    # ─────────────────────────────────────────────────────
    print("=" * 60)
    print("SENDING TO GROQ LLM (llama-3.3-70b-versatile)")
    print("=" * 60)

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    # as_messages() returns the perfect format for any chat LLM
    messages = ctx.as_messages()
    response = llm.invoke(messages)

    print(f"\nLLM ANSWER:\n")
    print(response.content)

    # ─────────────────────────────────────────────────────
    # Bonus: You can also index a specific URL and ask about it
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BONUS: Index a specific page and ask a question")
    print("=" * 60)

    pipeline.index_url("https://en.wikipedia.org/wiki/Python_(programming_language)")
    ctx2 = pipeline.retrieve("Who created Python and when?")

    response2 = llm.invoke(ctx2.as_messages())
    print(f"\nQuestion: Who created Python and when?")
    print(f"Answer: {response2.content}")


if __name__ == "__main__":
    main()
