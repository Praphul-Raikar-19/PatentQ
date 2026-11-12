"""Shared formatting and prompt-preview helpers for PatentQ agents."""

from __future__ import annotations

from typing import List

from src.models import PriorArtDocument


def format_docs_for_prompt(docs: List[PriorArtDocument], max_docs: int = 8) -> str:
    """Format retrieved ChromaDB documents with explicit [DOC_ID] citation tags."""
    if not docs:
        return "No prior-art documents retrieved from ChromaDB."
    blocks: List[str] = []
    for doc in docs[:max_docs]:
        score_str = f" | Similarity: {doc.similarity_score:.2f}" if doc.similarity_score is not None else ""
        blocks.append(
            f"[{doc.doc_id}] ({doc.doc_type.upper()}, {doc.year}, {doc.source_api}{score_str})\n"
            f"Title: {doc.title}\n"
            f"Authors/Assignee: {doc.authors_or_assignee}\n"
            f"URL: {doc.url or 'N/A'}\n"
            f"Excerpt: {doc.abstract[:750]}"
        )
    return "\n\n---\n\n".join(blocks)


def format_references_appendix(docs: List[PriorArtDocument]) -> str:
    """Render a clean Markdown bibliography for the final screening note."""
    if not docs:
        return "_No external references were retrieved or indexed during this run._"
    lines: List[str] = []
    for doc in docs:
        score = f" (Sim: {doc.similarity_score:.2f})" if doc.similarity_score is not None else ""
        link = f"[{doc.url}]({doc.url})" if doc.url else "No URL"
        lines.append(
            f"- **`[{doc.doc_id}]`** **{doc.title}** — *{doc.doc_type.title()}* "
            f"| {doc.authors_or_assignee} ({doc.year}, {doc.source_api}){score} — {link}"
        )
    return "\n".join(lines)

