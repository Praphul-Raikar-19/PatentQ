"""Intent Detection and LCEL RunnableBranch Routing for PatentQ."""

from __future__ import annotations

from typing import Any, Callable, Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda

from src.llm_factory import extract_json_from_text
from src.models import IntentDetectionResult, IntentType
from src.prompts import build_intent_prompt

PIPELINE_LABELS: Dict[IntentType, str] = {
    "prior_art_search": "Search Agent -> ChromaDB Indexing",
    "novelty_comparison": "Agentic RAG (ChromaDB) -> Novelty Comparison Agent",
    "gap_analysis": "Agentic RAG (ChromaDB) -> Novelty Comparison -> Gap Agent",
    "full_screening_report": "Search Agent -> ChromaDB -> Comparison -> Gap -> Report Agent",
    "followup_qa": "Agentic RAG Decision -> ChromaDB Retrieval / Tool Fallback -> Q&A Synthesis",
}

VALID_INTENTS = set(PIPELINE_LABELS.keys())


def heuristic_intent_fallback(user_request: str, has_idea: bool) -> IntentDetectionResult:
    """Deterministic keyword fallback if LLM JSON parsing fails."""
    text = (user_request or "").lower()
    if any(k in text for k in ["search only", "only search", "find patents", "find papers", "prior art search"]):
        intent: IntentType = "prior_art_search"
    elif any(k in text for k in ["gap", "white space", "under-explored", "novel aspects"]):
        intent = "gap_analysis"
    elif any(k in text for k in ["compare", "overlap", "novelty check", "similarity"]):
        intent = "novelty_comparison"
    elif any(k in text for k in ["?", "why", "which patent", "explain", "what did", "summarize"]):
        intent = "followup_qa"
    elif has_idea:
        intent = "full_screening_report"
    else:
        intent = "prior_art_search"

    return IntentDetectionResult(
        intent=intent,
        confidence=0.75,
        rationale="Heuristic intent guardrail classified request from action keywords and context.",
        routed_pipeline=PIPELINE_LABELS[intent],
        extracted_query=user_request[:120].strip(),
    )


def detect_intent_lcel(
    llm: BaseChatModel,
    user_request: str,
    has_idea: bool,
    indexed_count: int,
    mode_override: str = "Auto-Detect",
) -> IntentDetectionResult:
    """Detect user intent via an LCEL chain, respecting explicit UI overrides."""
    override_map: Dict[str, IntentType] = {
        "Prior-Art Search Only": "prior_art_search",
        "Novelty Comparison": "novelty_comparison",
        "Gap Analysis": "gap_analysis",
        "Full Screening Report": "full_screening_report",
        "Follow-up Q&A": "followup_qa",
    }

    if mode_override in override_map:
        chosen = override_map[mode_override]
        return IntentDetectionResult(
            intent=chosen,
            confidence=1.0,
            rationale=f"Explicit user mode selected: '{mode_override}'.",
            routed_pipeline=PIPELINE_LABELS[chosen],
            extracted_query=user_request[:150].strip(),
        )

    def _parse_intent(raw: str) -> IntentDetectionResult:
        data = extract_json_from_text(raw)
        raw_intent = str(data.get("intent", "")).strip()
        if raw_intent not in VALID_INTENTS:
            return heuristic_intent_fallback(user_request, has_idea)
        intent_typed: IntentType = raw_intent  # type: ignore[assignment]
        conf = float(data.get("confidence", 0.85))
        conf = max(0.0, min(1.0, conf))
        return IntentDetectionResult(
            intent=intent_typed,
            confidence=conf,
            rationale=str(data.get("rationale", "Classified via PatentQ LCEL Intent Router.")),
            routed_pipeline=PIPELINE_LABELS[intent_typed],
            extracted_query=str(data.get("extracted_query", user_request[:120])).strip(),
        )

    intent_chain = build_intent_prompt() | llm | StrOutputParser() | RunnableLambda(_parse_intent)
    try:
        return intent_chain.invoke(
            {
                "user_request": user_request,
                "has_idea": str(has_idea),
                "indexed_count": indexed_count,
                "mode_override": mode_override,
            }
        )
    except Exception:
        return heuristic_intent_fallback(user_request, has_idea)


def build_intent_router_branch(handlers: Dict[IntentType, Callable[[Dict[str, Any]], Any]]) -> RunnableBranch:
    """Build an LCEL RunnableBranch routing state payloads to the selected agent pipeline."""
    return RunnableBranch(
        (lambda x: x["intent_result"].intent == "prior_art_search", RunnableLambda(handlers["prior_art_search"])),
        (lambda x: x["intent_result"].intent == "novelty_comparison", RunnableLambda(handlers["novelty_comparison"])),
        (lambda x: x["intent_result"].intent == "gap_analysis", RunnableLambda(handlers["gap_analysis"])),
        (lambda x: x["intent_result"].intent == "followup_qa", RunnableLambda(handlers["followup_qa"])),
        RunnableLambda(handlers["full_screening_report"]),
    )

