"""Pydantic data models for PatentQ agents, prior-art documents, routing, and reports."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


IntentType = Literal[
    "prior_art_search",
    "novelty_comparison",
    "gap_analysis",
    "full_screening_report",
    "followup_qa",
]


class InventionIdea(BaseModel):
    """Structured invention disclosure submitted by the user."""

    title: str = Field(..., min_length=3, description="Short descriptive title of the invention.")
    problem: str = Field(..., min_length=10, description="Technical problem addressed.")
    solution: str = Field(..., min_length=10, description="Core mechanism or technical solution.")
    components: str = Field(default="", description="Key components, steps, or draft claim elements.")
    keywords: str = Field(default="", description="Comma-separated technical keywords.")
    domain: str = Field(default="General Engineering", description="Technical domain or IPC area.")

    def to_query_text(self) -> str:
        parts = [self.title, self.problem, self.solution]
        if self.components:
            parts.append(f"Key components: {self.components}")
        if self.keywords:
            parts.append(f"Keywords: {self.keywords}")
        return "\n".join(parts)


class PriorArtDocument(BaseModel):
    """Normalized patent, research paper, or commercial product record indexed in ChromaDB."""

    doc_id: str = Field(..., description="Unique identifier (e.g., ARXIV-2401.12345 or PAT-US11223344).")
    doc_type: Literal["patent", "paper", "product", "custom"] = Field(default="paper")
    title: str
    abstract: str
    authors_or_assignee: str = "Unknown"
    year: str = "N/A"
    url: str = ""
    image_url: str = ""
    price: str = ""
    source_api: str = "manual"
    similarity_score: Optional[float] = None

    def to_citation_tag(self) -> str:
        return f"[{self.doc_id}]"

    def to_embedding_text(self) -> str:
        return (
            f"Title: {self.title}\n"
            f"Type: {self.doc_type.upper()} | Source: {self.source_api} | Year: {self.year}\n"
            f"Authors/Assignee: {self.authors_or_assignee}\n"
            f"Abstract/Excerpt: {self.abstract}"
        )


class IntentDetectionResult(BaseModel):
    """Result of the Intent Detection & Routing step."""

    intent: IntentType
    confidence: float = Field(ge=0.0, le=1.0, default=0.85)
    rationale: str
    routed_pipeline: str
    extracted_query: str = ""


class OverlapItem(BaseModel):
    """Single feature-level comparison against a prior-art reference."""

    feature: str
    prior_art_id: str
    prior_art_title: str = ""
    overlap_level: Literal["High", "Moderate", "Low", "None"] = "Moderate"
    explanation: str


class FTORiskItem(BaseModel):
    """Freedom-to-Operate (FTO) risk screening entry for a registered/published patent."""

    patent_id: str
    title: str = ""
    assignee: str = "Unknown"
    status_or_jurisdiction: str = "Published / Active Check Recommended"
    fto_risk: Literal["High", "Moderate", "Low"] = "Moderate"
    blocking_claim_focus: str
    design_around_note: str = ""


class NoveltyComparisonResult(BaseModel):
    """Output of the Novelty Comparison & FTO Screening Agent."""

    overall_overlap_risk: Literal["High", "Moderate", "Low"] = "Moderate"
    fto_risk_level: Literal["High", "Moderate", "Low"] = "Moderate"
    summary: str
    novelty_102_assessment: str = ""
    obviousness_103_assessment: str = ""
    overlap_items: List[OverlapItem] = Field(default_factory=list)
    fto_items: List[FTORiskItem] = Field(default_factory=list)
    closest_references: List[str] = Field(default_factory=list)
    reasoning_trace: str = ""


class GapItem(BaseModel):
    """Specific novelty gap or under-explored direction identified by the Gap Agent."""

    aspect: str
    novelty_signal: Literal["Strong", "Moderate", "Exploratory"] = "Moderate"
    evidence_basis: str
    suggested_claim_or_experiment: str


class GapAnalysisResult(BaseModel):
    """Output of the Gap Agent."""

    summary: str
    novel_gaps: List[GapItem] = Field(default_factory=list)
    under_explored_combinations: List[str] = Field(default_factory=list)
    evidence_sufficiency: Literal["Sufficient", "Needs_More_Search"] = "Sufficient"
    followup_search_queries: List[str] = Field(default_factory=list)
    reasoning_trace: str = ""


class AgentTraceEntry(BaseModel):
    """Execution trace entry for transparent inspection in the Streamlit UI."""

    step: int
    agent: str
    action: str
    prompt_technique: str
    system_prompt_preview: str = ""
    user_prompt_preview: str = ""
    decision_or_output: str = ""
    retrieved_doc_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScreeningResult(BaseModel):
    """Complete bundle returned by the PatentQ Orchestrator to the Streamlit UI."""

    intent_result: IntentDetectionResult
    search_queries_used: List[str] = Field(default_factory=list)
    newly_indexed_docs: List[PriorArtDocument] = Field(default_factory=list)
    retrieved_docs: List[PriorArtDocument] = Field(default_factory=list)
    comparison: Optional[NoveltyComparisonResult] = None
    gap_analysis: Optional[GapAnalysisResult] = None
    report_markdown: str = ""
    qa_answer: str = ""
    traces: List[AgentTraceEntry] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

