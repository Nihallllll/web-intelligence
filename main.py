import sys
import json

HELP_TEXT = """\
Web Intelligence CLI

Usage:
    python main.py index <url>             Index a single URL
    python main.py index <url1> <url2>     Index multiple URLs
    python main.py search <query>          Semantic search
    python main.py retrieve <query>        Get LLM-ready context
    python main.py documents               List indexed documents
    python main.py delete <doc_id>         Delete a document
    python main.py stats                   Show pipeline stats
    python main.py serve                   Start the REST API server
    python main.py clear                   Clear all data
"""


def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help", "help"):
        print(HELP_TEXT.strip())
        return

    command = args[0].lower()

    # Lazy import to avoid slow startup for help text
    from web_intelligence import FastPipeline

    if command == "serve":
        from web_intelligence.server import start_server
        host = None
        port = None
        for i, a in enumerate(args[1:], 1):
            if a in ("--host",) and i + 1 < len(args):
                host = args[i + 1]
            if a in ("--port", "-p") and i + 1 < len(args):
                port = int(args[i + 1])
        start_server(host=host, port=port)
        return

    pipeline = FastPipeline()

    if command == "index":
        urls = args[1:]
        if not urls:
            print("Usage: python main.py index <url> [url2] [url3] ...")
            return
        if len(urls) == 1:
            result = pipeline.index_url(urls[0])
            print(json.dumps(result, indent=2, default=str))
        else:
            results = pipeline.index_batch(urls)
            for r in results:
                status = "OK" if r.get("success") else "FAIL"
                cached = " (cached)" if r.get("cached") else ""
                print(f"  [{status}] {r['url']}{cached}")

    elif command == "search":
        query = " ".join(args[1:])
        if not query:
            print("Usage: python main.py search <query>")
            return
        limit = 5
        for i, a in enumerate(args):
            if a in ("--limit", "-n") and i + 1 < len(args):
                limit = int(args[i + 1])
                query = " ".join(a for a in args[1:] if a not in ("--limit", "-n", args[i + 1]))
        results = pipeline.search(query, limit=limit)
        for i, r in enumerate(results, 1):
            print(f"\n--- Result {i} (score: {r['score']:.3f}) ---")
            print(f"Source: {r['metadata'].get('url', '')}")
            print(f"Title:  {r['metadata'].get('title', '')}")
            print(r["text"][:500])

    elif command == "retrieve":
        query = " ".join(args[1:])
        if not query:
            print("Usage: python main.py retrieve <query>")
            return
        ctx = pipeline.retrieve(query)
        print(f"Query: {ctx.query}")
        print(f"Sources: {len(ctx.sources)} | Chunks: {ctx.total_chunks} | Words: {ctx.total_words}")
        print(f"\n{'='*60}")
        print(ctx.context_text)
        print(f"{'='*60}")
        print("\nOpenAI-compatible messages:")
        print(json.dumps(ctx.as_messages(), indent=2)[:500] + "...")

    elif command == "documents":
        docs = pipeline.list_documents()
        if not docs:
            print("No documents indexed.")
            return
        print(f"Total: {len(docs)} documents\n")
        for d in docs:
            print(f"  {d['doc_id'][:12]}...  {d['title'][:40]:<40}  {d['chunk_count']} chunks  {d['url']}")

    elif command == "delete":
        if len(args) < 2:
            print("Usage: python main.py delete <doc_id or url>")
            return
        target = args[1]
        if target.startswith("http"):
            count = pipeline.delete_url(target)
            print(f"Deleted {count} chunks from {target}")
        else:
            ok = pipeline.delete_document(target)
            print(f"Deleted: {ok}")

    elif command == "stats":
        stats = pipeline.stats()
        print(json.dumps(stats, indent=2, default=str))

    elif command == "clear":
        confirm = input("This will delete ALL indexed data. Type 'yes' to confirm: ")
        if confirm.strip().lower() == "yes":
            pipeline.clear_all()
        else:
            print("Aborted.")

    else:
        print(f"Unknown command: {command}")
        print(HELP_TEXT.strip())


if __name__ == "__main__":
    main()
