from dotenv import load_dotenv

load_dotenv()

from web_intelligence import FastPipeline
from langchain_groq import ChatGroq


def main():
    print("Setting up Web Intelligence pipeline...")
    pipeline = FastPipeline()

    question = "what is lcm of 9 and 17"

    print(f"\nSearching the web for: '{question}'")
    print("(This searches DuckDuckGo → crawls pages → indexes → retrieves)\n")

    ctx = pipeline.search_web(question, max_results=3, limit=5)

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

    print("=" * 60)
    print("SENDING TO GROQ LLM (llama-3.3-70b-versatile)")
    print("=" * 60)

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    messages = ctx.as_messages()
    response = llm.invoke(messages)

    print(f"\nLLM ANSWER:\n")
    print(response.content)

    print("\n" + "=" * 60)
    print("BONUS: Index a specific page and ask a question")
    print("=" * 60)

    pipeline.index_url("https://en.wikipedia.org/wiki/Python_(programming_language)")
    ctx2 = pipeline.retrieve("Who created Python and when?")

    response2 = llm.invoke(ctx2.as_messages())
    print(f"\nQuestion: Who created Python and when?")
    print(f"Answer: {response2.content}")

    print("\n" + "=" * 60)
    print("INTERACTIVE WEB Q&A")
    print("=" * 60)
    print("Ask anything. Type 'exit' or 'quit' to stop.")

    while True:
        user_question = input("\nYour question: ").strip()

        if user_question.lower() in {"exit", "quit"}:
            print("Exiting interactive session.")
            break

        if not user_question:
            print("Please enter a question or type 'exit'.")
            continue

        print("\nSearching the web and building context...")
        ctx_loop = pipeline.search_web(user_question, max_results=3, limit=5)

        print(f"Sources found: {len(ctx_loop.sources)}")
        for source in ctx_loop.sources:
            print(f"  - {source['title']}: {source['url']}")

        response_loop = llm.invoke(ctx_loop.as_messages(user_question=user_question))
        print("\nAnswer:")
        print(response_loop.content)


if __name__ == "__main__":
    main()
