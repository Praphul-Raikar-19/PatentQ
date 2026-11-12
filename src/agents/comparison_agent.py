"""Novelty Comparison Agent: compares invention features against ChromaDB-retrieved prior art."""

from __future__ import annotations

from typing import List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from src.agents.utils import format_docs_for_prompt
from src.llm_factory import extract_json_from_text
from src.models import (
    FTORiskItem,
    InventionIdea,
    NoveltyComparisonResult,
    OverlapItem,
    PriorArtDocument,
)
from src.prompts import build_comparison_prompt


class NoveltyComparisonAgent:
    """Compares user invention against retrieved prior art (§102/§103 + FTO) using LCEL."""

    def __init__(self, llm: BaseChatModel, technique: str = "CoT") -> None:
        self.llm = llm
        self.technique = technique
        self.prompt = build_comparison_prompt(technique)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def run(self, idea: InventionIdea, retrieved_docs: List[PriorArtDocument]) -> NoveltyComparisonResult:
        context = format_docs_for_prompt(retrieved_docs)
        raw = self.chain.invoke(
            {
                "title": idea.title,
                "domain": idea.domain,
                "problem": idea.problem,
                "solution": idea.solution,
                "components": idea.components or "Not specified",
                "context": context,
            }
        )
        data = extract_json_from_text(raw)
        if not data:
            return NoveltyComparisonResult(
                overall_overlap_risk="Moderate",
                fto_risk_level="Moderate",
                summary=raw[:600] if raw else "Comparison completed; structured JSON unavailable.",
                novelty_102_assessment="Single-reference §102 anticipation requires claim-level review of top retrieved citations.",
                obviousness_103_assessment="Multi-reference §103 combination risk depends on motivation to combine retrieved mechanisms.",
                closest_references=[d.doc_id for d in retrieved_docs[:3]],
                reasoning_trace="Fallback raw-text comparison output.",
            )

        items: List[OverlapItem] = []
        for raw_item in data.get("overlap_items", []):
            if not isinstance(raw_item, dict):
                continue
            level = str(raw_item.get("overlap_level", "Moderate"))
            if level not in {"High", "Moderate", "Low", "None"}:
                level = "Moderate"
            items.append(
                OverlapItem(
                    feature=str(raw_item.get("feature", "Feature")),
                    prior_art_id=str(raw_item.get("prior_art_id", "N/A")),
                    prior_art_title=str(raw_item.get("prior_art_title", "")),
                    overlap_level=level,  # type: ignore[arg-type]
                    explanation=str(raw_item.get("explanation", "")),
                )
            )

        fto_items: List[FTORiskItem] = []
        for raw_fto in data.get("fto_items", []):
            if not isinstance(raw_fto, dict):
                continue
            f_risk = str(raw_fto.get("fto_risk", "Moderate"))
            if f_risk not in {"High", "Moderate", "Low"}:
                f_risk = "Moderate"
            fto_items.append(
                FTORiskItem(
                    patent_id=str(raw_fto.get("patent_id", "N/A")),
                    title=str(raw_fto.get("title", "")),
                    assignee=str(raw_fto.get("assignee", "Unknown")),
                    status_or_jurisdiction=str(
                        raw_fto.get("status_or_jurisdiction", "Published/Registered Patent — Verify Legal Status")
                    ),
                    fto_risk=f_risk,  # type: ignore[arg-type]
                    blocking_claim_focus=str(raw_fto.get("blocking_claim_focus", "")),
                    design_around_note=str(raw_fto.get("design_around_note", "")),
                )
            )

        # If model didn't populate fto_items explicitly, derive entries from retrieved patent docs
        if not fto_items:
            patent_docs = [d for d in retrieved_docs if d.doc_type == "patent"][:3]
            for p in patent_docs:
                fto_items.append(
                    FTORiskItem(
                        patent_id=p.doc_id,
                        title=p.title,
                        assignee=p.authors_or_assignee,
                        status_or_jurisdiction=f"Published {p.year} ({p.source_api})",
                        fto_risk="Moderate",
                        blocking_claim_focus="Verify independent claims covering overlapping sub-system.",
                        design_around_note="Emphasize distinct architecture/control parameters identified in Gap Analysis.",
                    )
                )

        risk = str(data.get("overall_overlap_risk", "Moderate"))
        if risk not in {"High", "Moderate", "Low"}:
            risk = "Moderate"

        fto_risk = str(data.get("fto_risk_level", "Moderate" if fto_items else "Low"))
        if fto_risk not in {"High", "Moderate", "Low"}:
            fto_risk = "Moderate"

        return NoveltyComparisonResult(
            overall_overlap_risk=risk,  # type: ignore[arg-type]
            fto_risk_level=fto_risk,  # type: ignore[arg-type]
            summary=str(data.get("summary", "Prior-art comparison completed.")),
            novelty_102_assessment=str(
                data.get(
                    "novelty_102_assessment",
                    "No single retrieved reference appears to identically disclose every combined element.",
                )
            ),
            obviousness_103_assessment=str(
                data.get(
                    "obviousness_103_assessment",
                    "Evaluate §103 combination risk across closest patents and academic references.",
                )
            ),
            overlap_items=items,
            fto_items=fto_items,
            closest_references=[str(x) for x in data.get("closest_references", [])],
            reasoning_trace=str(data.get("reasoning_trace", "")),
        )
