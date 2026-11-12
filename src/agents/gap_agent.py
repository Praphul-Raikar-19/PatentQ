"""Gap Agent: highlights novel aspects, white-space angles, and evidence sufficiency."""

from __future__ import annotations

from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from src.agents.utils import format_docs_for_prompt
from src.llm_factory import extract_json_from_text
from src.models import (
    GapAnalysisResult,
    GapItem,
    InventionIdea,
    NoveltyComparisonResult,
    PriorArtDocument,
)
from src.prompts import build_gap_prompt


class GapAgent:
    """Identifies potential novelty gaps and under-explored areas via LCEL."""

    def __init__(self, llm: BaseChatModel, technique: str = "Reflection") -> None:
        self.llm = llm
        self.technique = technique
        self.prompt = build_gap_prompt(technique)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def run(
        self,
        idea: InventionIdea,
        retrieved_docs: List[PriorArtDocument],
        comparison: Optional[NoveltyComparisonResult] = None,
    ) -> GapAnalysisResult:
        context = format_docs_for_prompt(retrieved_docs)
        comparison_summary = comparison.summary if comparison else "No prior comparison summary provided."
        raw = self.chain.invoke(
            {
                "title": idea.title,
                "solution": idea.solution,
                "components": idea.components or "Not specified",
                "comparison_summary": comparison_summary,
                "context": context,
            }
        )
        data = extract_json_from_text(raw)
        if not data:
            return GapAnalysisResult(
                summary=raw[:600] if raw else "Gap analysis completed.",
                reasoning_trace="Fallback raw-text gap output.",
            )

        gaps: List[GapItem] = []
        for item in data.get("novel_gaps", []):
            if not isinstance(item, dict):
                continue
            signal = str(item.get("novelty_signal", "Moderate"))
            if signal not in {"Strong", "Moderate", "Exploratory"}:
                signal = "Moderate"
            gaps.append(
                GapItem(
                    aspect=str(item.get("aspect", "Novel aspect")),
                    novelty_signal=signal,  # type: ignore[arg-type]
                    evidence_basis=str(item.get("evidence_basis", "")),
                    suggested_claim_or_experiment=str(item.get("suggested_claim_or_experiment", "")),
                )
            )

        sufficiency = str(data.get("evidence_sufficiency", "Sufficient"))
        if sufficiency not in {"Sufficient", "Needs_More_Search"}:
            sufficiency = "Sufficient"

        return GapAnalysisResult(
            summary=str(data.get("summary", "Novelty gap analysis completed.")),
            novel_gaps=gaps,
            under_explored_combinations=[str(x) for x in data.get("under_explored_combinations", [])],
            evidence_sufficiency=sufficiency,  # type: ignore[arg-type]
            followup_search_queries=[str(x) for x in data.get("followup_search_queries", []) if str(x).strip()],
            reasoning_trace=str(data.get("reasoning_trace", "")),
        )

