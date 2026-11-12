"""Report Agent: drafts a cited preliminary prior-art & novelty screening note."""

from __future__ import annotations

import json
from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from src.agents.utils import format_docs_for_prompt, format_references_appendix
from src.config import DISCLAIMER_TEXT
from src.models import (
    GapAnalysisResult,
    InventionIdea,
    NoveltyComparisonResult,
    PriorArtDocument,
)
from src.prompts import build_report_prompt


class ReportAgent:
    """Synthesizes comparison, gaps, and citations into a Markdown screening note via LCEL."""

    def __init__(self, llm: BaseChatModel, technique: str = "CoT") -> None:
        self.llm = llm
        self.technique = technique
        self.prompt = build_report_prompt(technique)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def run(
        self,
        idea: InventionIdea,
        search_queries: List[str],
        retrieved_docs: List[PriorArtDocument],
        comparison: Optional[NoveltyComparisonResult],
        gap_analysis: Optional[GapAnalysisResult],
    ) -> str:
        comparison_json = (
            comparison.model_dump_json(indent=2)
            if comparison
            else json.dumps({"note": "Comparison skipped."})
        )
        gap_json = (
            gap_analysis.model_dump_json(indent=2)
            if gap_analysis
            else json.dumps({"note": "Gap analysis skipped."})
        )
        references_block = format_docs_for_prompt(retrieved_docs, max_docs=10)

        body = self.chain.invoke(
            {
                "title": idea.title,
                "domain": idea.domain,
                "problem": idea.problem,
                "solution": idea.solution,
                "components": idea.components or "Not specified",
                "search_queries": ", ".join(search_queries) if search_queries else "ChromaDB existing index",
                "comparison_json": comparison_json,
                "gap_json": gap_json,
                "references_block": references_block,
            }
        ).strip()

        verified_refs = format_references_appendix(retrieved_docs)
        if "## Cited References" not in body and "## 7. Cited References" not in body:
            body = f"{body}\n\n## Cited References (Verified from ChromaDB)\n{verified_refs}"
        else:
            body = f"{body}\n\n### Verified Reference Links\n{verified_refs}"

        if "screening scope notice" not in body.lower() and "registered patent attorney" not in body.lower():
            body = f"{body}\n\n---\n*{DISCLAIMER_TEXT}*"
        return body

