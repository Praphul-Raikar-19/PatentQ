"""Search Agent: formulates search queries via LCEL, queries patent/paper APIs, and indexes into ChromaDB."""

from __future__ import annotations

from typing import List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from src.llm_factory import extract_json_from_text
from src.models import InventionIdea, PriorArtDocument
from src.prompts import build_search_query_prompt
from src.tools.search_tools import run_multi_source_search
from src.vector_store import ChromaPriorArtStore


class SearchAgent:
    """Queries free patent and paper APIs and persists excerpts in ChromaDB."""

    def __init__(self, llm: BaseChatModel, vector_store: ChromaPriorArtStore) -> None:
        self.llm = llm
        self.vector_store = vector_store
        self.query_chain = build_search_query_prompt() | self.llm | StrOutputParser()

    def generate_queries(self, idea: InventionIdea) -> Tuple[List[str], str]:
        """Generate 3 targeted search queries using LangChain LCEL."""
        fallback_queries = [
            idea.title,
            f"{idea.title} {idea.keywords}".strip(),
            " ".join(idea.solution.split()[:10]),
        ]
        try:
            raw = self.query_chain.invoke(
                {
                    "title": idea.title,
                    "domain": idea.domain,
                    "problem": idea.problem,
                    "solution": idea.solution,
                    "components": idea.components or "Not specified",
                    "keywords": idea.keywords or "Not specified",
                }
            )
            parsed = extract_json_from_text(raw)
            queries = [str(q).strip() for q in parsed.get("queries", []) if str(q).strip()]
            rationale = str(parsed.get("rationale", "Generated via LCEL Search Agent."))
            if queries:
                return queries[:3], rationale
        except Exception:
            pass
        return [q for q in fallback_queries if q][:3], "Used structured fallback query extraction."

    def run(
        self,
        idea: Optional[InventionIdea] = None,
        explicit_query: str = "",
        sources: Optional[List[str]] = None,
        max_per_source: int = 4,
        patentsview_api_key: Optional[str] = None,
        max_queries: int = 3,
    ) -> Tuple[List[str], List[PriorArtDocument], List[str], str]:
        """Execute multi-source search and upsert results into ChromaDB."""
        if explicit_query.strip() and idea is None:
            queries = [explicit_query.strip()]
            rationale = "Used direct user search query."
        elif idea is not None:
            queries, rationale = self.generate_queries(idea)
            if explicit_query.strip():
                queries = [explicit_query.strip()] + queries[:2]
        else:
            return [], [], ["No search query or invention idea supplied."], "Skipped empty search."

        queries = queries[:max(1, max_queries)]
        docs, warnings = run_multi_source_search(
            queries=queries,
            sources=sources,
            max_per_source=max_per_source,
            patentsview_api_key=patentsview_api_key,
        )
        indexed_count = self.vector_store.upsert_documents(docs)
        if indexed_count == 0 and not warnings:
            warnings.append("External search returned 0 results for these queries. Try broader keywords or add custom excerpts.")
        return queries, docs, warnings, rationale

