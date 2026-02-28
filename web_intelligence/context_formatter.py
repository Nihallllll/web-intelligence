"""
Context formatters that turn search results into LLM-ready text.

This module provides multiple output formats so users can feed retrieved
content to any LLM (local Ollama/Llama, OpenAI, Anthropic, etc.) in the
format that works best for their use case.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class RetrievedContext:
    """
    Structured search result ready to be fed to any LLM.

    Attributes:
        query: The original search query.
        chunks: Raw list of chunk dicts from the vector store.
        context_text: Pre-formatted context string for direct LLM use.
        sources: Deduplicated list of source URLs with titles.
        total_words: Word count of the context text.
        total_chunks: Number of chunks included.
    """
    query: str
    chunks: List[Dict]
    context_text: str
    sources: List[Dict]
    total_words: int
    total_chunks: int

    def to_dict(self) -> Dict:
        """Serialize to a plain dict (JSON-safe)."""
        return {
            "query": self.query,
            "context_text": self.context_text,
            "sources": self.sources,
            "total_words": self.total_words,
            "total_chunks": self.total_chunks,
            "chunks": [
                {
                    "text": c.get("text", ""),
                    "score": c.get("score", 0),
                    "url": c.get("metadata", {}).get("url", ""),
                    "title": c.get("metadata", {}).get("title", ""),
                }
                for c in self.chunks
            ],
        }

    def as_messages(self, system_prompt: Optional[str] = None, user_question: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Return an OpenAI-compatible messages list ready for any chat model.

        Works with: OpenAI, Anthropic, Ollama, LiteLLM, LangChain, etc.

        Args:
            system_prompt: Custom system prompt. Defaults to a generic RAG instruction.
            user_question: Override the question (defaults to self.query).

        Returns:
            List of {"role": ..., "content": ...} dicts.
        """
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant. Answer the user's question based on "
                "the provided context. If the context doesn't contain enough information, "
                "say so. Cite sources when possible."
            )

        question = user_question or self.query
        source_list = "\n".join(
            f"  - [{s['title']}]({s['url']})" for s in self.sources
        )
        user_content = (
            f"Context:\n{self.context_text}\n\n"
            f"Sources:\n{source_list}\n\n"
            f"Question: {question}"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]


def format_context_plain(chunks: List[Dict], query: str,
                         max_words: int = 3000,
                         include_sources: bool = True) -> RetrievedContext:
    """
    Format search results as plain concatenated text.

    Best for: simple prompt injection, any LLM.

    Args:
        chunks: Search results from FastPipeline.search().
        query: The original query string.
        max_words: Truncate context to this many words.
        include_sources: Append source URLs at the end.

    Returns:
        RetrievedContext with plain text formatting.
    """
    parts = []
    sources_seen = {}
    word_count = 0

    for chunk in chunks:
        text = chunk.get("text", "")
        words = text.split()
        if word_count + len(words) > max_words:
            remaining = max_words - word_count
            if remaining > 0:
                parts.append(" ".join(words[:remaining]))
                word_count += remaining
            break
        parts.append(text)
        word_count += len(words)

        meta = chunk.get("metadata", {})
        url = meta.get("url", "")
        if url and url not in sources_seen:
            sources_seen[url] = meta.get("title", "Untitled")

    context = "\n\n".join(parts)

    if include_sources and sources_seen:
        source_lines = "\n".join(f"- {title}: {url}" for url, title in sources_seen.items())
        context += f"\n\nSources:\n{source_lines}"

    sources = [{"url": url, "title": title} for url, title in sources_seen.items()]

    return RetrievedContext(
        query=query,
        chunks=chunks,
        context_text=context,
        sources=sources,
        total_words=word_count,
        total_chunks=len(parts),
    )


def format_context_numbered(chunks: List[Dict], query: str,
                            max_words: int = 3000) -> RetrievedContext:
    """
    Format search results with numbered source citations.

    Best for: prompts that need citation references like [1], [2].

    Each chunk is prefixed with [Source N] and its URL.
    """
    parts = []
    sources_seen = {}
    word_count = 0

    for i, chunk in enumerate(chunks, 1):
        text = chunk.get("text", "")
        meta = chunk.get("metadata", {})
        url = meta.get("url", "")
        title = meta.get("title", "Untitled")
        score = chunk.get("score", 0)

        words = text.split()
        if word_count + len(words) > max_words:
            remaining = max_words - word_count
            if remaining > 20:
                parts.append(f"[Source {i}] ({url})\n" + " ".join(words[:remaining]))
                word_count += remaining
            break

        parts.append(f"[Source {i}] (relevance: {score:.2f}) {url}\n{text}")
        word_count += len(words)

        if url and url not in sources_seen:
            sources_seen[url] = title

    context = "\n\n---\n\n".join(parts)
    sources = [{"url": url, "title": title} for url, title in sources_seen.items()]

    return RetrievedContext(
        query=query,
        chunks=chunks,
        context_text=context,
        sources=sources,
        total_words=word_count,
        total_chunks=len(parts),
    )


def format_context_structured(chunks: List[Dict], query: str,
                              max_words: int = 3000) -> RetrievedContext:
    """
    Format search results as a structured JSON-like block.

    Best for: LLMs that handle structured/XML input well (Claude, GPT-4).
    """
    parts = []
    sources_seen = {}
    word_count = 0

    for i, chunk in enumerate(chunks, 1):
        text = chunk.get("text", "")
        meta = chunk.get("metadata", {})
        url = meta.get("url", "")
        title = meta.get("title", "Untitled")
        score = chunk.get("score", 0)

        words = text.split()
        if word_count + len(words) > max_words:
            break

        block = (
            f"<document index=\"{i}\">\n"
            f"  <source>{url}</source>\n"
            f"  <title>{title}</title>\n"
            f"  <relevance>{score:.2f}</relevance>\n"
            f"  <content>{text}</content>\n"
            f"</document>"
        )
        parts.append(block)
        word_count += len(words)

        if url and url not in sources_seen:
            sources_seen[url] = title

    context = "\n\n".join(parts)
    sources = [{"url": url, "title": title} for url, title in sources_seen.items()]

    return RetrievedContext(
        query=query,
        chunks=chunks,
        context_text=context,
        sources=sources,
        total_words=word_count,
        total_chunks=len(parts),
    )
