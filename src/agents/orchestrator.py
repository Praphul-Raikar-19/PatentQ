"""PatentQ Orchestrator: coordinates Intent Routing, Agentic RAG, and the 4-agent team."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from src.agents.comparison_agent import NoveltyComparisonAgent
from src.agents.gap_agent import GapAgent
from src.agents.report_agent import ReportAgent
from src.agents.search_agent import SearchAgent
from src.agents.utils import format_docs_for_prompt
from src.llm_factory import extract_json_from_text
from src.models import (
    AgentTraceEntry,
    InventionIdea,
    PriorArtDocument,
    ScreeningResult,
)
from src.prompts import (
    AGENTIC_RAG_SYSTEM_PROMPT,
    AGENTIC_RAG_USER_PROMPT,
    TECHNIQUE_DIRECTIVES,
    build_agentic_rag_prompt,
    build_qa_prompt,
)
from src.router import build_intent_router_branch, detect_intent_lcel
from src.vector_store import ChromaPriorArtStore


class PatentQOrchestrator:
    """Coordinates Search, Comparison, Gap, and Report agents with Agentic RAG over ChromaDB."""

    def __init__(
        self,
        llm: BaseChatModel,
        vector_store: ChromaPriorArtStore,
        prompt_technique: str = "CoT",
        sources: Optional[List[str]] = None,
        top_k: int = 6,
        max_per_source: int = 4,
        patentsview_api_key: Optional[str] = None,
        fast_mode: bool = True,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> None:
        self.llm = llm
        self.vector_store = vector_store
        self.prompt_technique = prompt_technique
        self.sources = sources or [
            "EuropePMC Patents",
            "Google Patents",
            "arXiv",
            "OpenAlex",
            "Crossref",
            "Online Shopping Products",
        ]
        self.top_k = top_k
        self.max_per_source = max_per_source
        self.patentsview_api_key = patentsview_api_key
        self.fast_mode = fast_mode
        self.progress_cb = progress_cb

        self.search_agent = SearchAgent(llm=llm, vector_store=vector_store)
        self.comparison_agent = NoveltyComparisonAgent(llm=llm, technique=prompt_technique)
        self.gap_agent = GapAgent(llm=llm, technique=prompt_technique)
        self.report_agent = ReportAgent(llm=llm, technique=prompt_technique)

        self.agentic_rag_chain = build_agentic_rag_prompt("ReAct") | self.llm | StrOutputParser()
        self.qa_chain = build_qa_prompt(prompt_technique) | self.llm | StrOutputParser()

    def _notify(self, pct: float, msg: str) -> None:
        if self.progress_cb:
            try:
                self.progress_cb(pct, msg)
            except Exception:
                pass

    def _add_trace(
        self,
        traces: List[AgentTraceEntry],
        agent: str,
        action: str,
        decision_or_output: str,
        technique: Optional[str] = None,
        system_preview: str = "",
        user_preview: str = "",
        retrieved_ids: Optional[List[str]] = None,
    ) -> None:
        traces.append(
            AgentTraceEntry(
                step=len(traces) + 1,
                agent=agent,
                action=action,
                prompt_technique=technique or self.prompt_technique,
                system_prompt_preview=system_preview[:500],
                user_prompt_preview=user_preview[:500],
                decision_or_output=decision_or_output,
                retrieved_doc_ids=retrieved_ids or [],
            )
        )

    def _agentic_rag_retrieve(
        self,
        query_text: str,
        idea: Optional[InventionIdea],
        traces: List[AgentTraceEntry],
        warnings: List[str],
        allow_external_search: bool = True,
    ) -> tuple[List[PriorArtDocument], List[PriorArtDocument], List[str]]:
        """Agentic RAG step: retrieve from ChromaDB, evaluate sufficiency, and call search tools if needed."""
        col_size = self.vector_store.count()
        initial_hits = self.vector_store.similarity_search(query_text, top_k=self.top_k)
        preview = format_docs_for_prompt(initial_hits, max_docs=4)

        decision = "USE_RETRIEVED"
        reason = "Sufficient evidence found in ChromaDB."
        refined_query = ""

        if col_size == 0 or not initial_hits:
            decision = "CALL_SEARCH_TOOL" if allow_external_search else "USE_RETRIEVED"
            reason = "ChromaDB collection is empty; invoking external patent/paper search tools first."
            refined_query = idea.title if idea else query_text
        elif allow_external_search:
            try:
                raw = self.agentic_rag_chain.invoke(
                    {
                        "task_description": query_text[:400],
                        "collection_size": col_size,
                        "retrieved_preview": preview[:1500],
                    }
                )
                parsed = extract_json_from_text(raw)
                cand = str(parsed.get("decision", "")).strip()
                if cand in {"USE_RETRIEVED", "CALL_SEARCH_TOOL"}:
                    decision = cand
                reason = str(parsed.get("reason", reason))
                refined_query = str(parsed.get("search_query", "")).strip()
            except Exception:
                pass

        self._add_trace(
            traces=traces,
            agent="Agentic RAG Controller",
            action="Evaluate ChromaDB Retrieval Sufficiency",
            technique="ReAct",
            system_preview=AGENTIC_RAG_SYSTEM_PROMPT.format(
                technique_directive=TECHNIQUE_DIRECTIVES["ReAct"]
            ),
            user_preview=AGENTIC_RAG_USER_PROMPT.format(
                task_description=query_text[:200],
                collection_size=col_size,
                retrieved_preview=preview[:200],
            ),
            decision_or_output=f"{decision}: {reason}",
            retrieved_ids=[d.doc_id for d in initial_hits],
        )

        newly_indexed: List[PriorArtDocument] = []
        queries_used: List[str] = []

        if decision == "CALL_SEARCH_TOOL" and allow_external_search:
            queries_used, newly_indexed, search_warns, q_rationale = self.search_agent.run(
                idea=idea,
                explicit_query=refined_query or query_text,
                sources=self.sources,
                max_per_source=self.max_per_source,
                patentsview_api_key=self.patentsview_api_key,
            )
            warnings.extend(search_warns)
            self._add_trace(
                traces=traces,
                agent="Search Agent",
                action="External Patent & Paper Tool Search + ChromaDB Upsert",
                decision_or_output=(
                    f"Queries: {queries_used} | Rationale: {q_rationale} | "
                    f"Indexed {len(newly_indexed)} documents into ChromaDB."
                ),
                retrieved_ids=[d.doc_id for d in newly_indexed],
            )
            initial_hits = self.vector_store.similarity_search(query_text, top_k=self.top_k)

        return initial_hits, newly_indexed, queries_used

    def run(
        self,
        user_request: str,
        idea: Optional[InventionIdea] = None,
        mode_override: str = "Auto-Detect",
        report_context_excerpt: str = "",
    ) -> ScreeningResult:
        """Detect intent, route via LCEL RunnableBranch, and execute the required agents."""
        traces: List[AgentTraceEntry] = []
        warnings: List[str] = []

        intent_result = detect_intent_lcel(
            llm=self.llm,
            user_request=user_request,
            has_idea=idea is not None,
            indexed_count=self.vector_store.count(),
            mode_override=mode_override,
        )
        self._add_trace(
            traces=traces,
            agent="Intent Router (LCEL)",
            action="Classify Intent & Select RunnableBranch Route",
            decision_or_output=(
                f"Intent='{intent_result.intent}' (confidence={intent_result.confidence:.2f}) "
                f"-> Pipeline: {intent_result.routed_pipeline}. Rationale: {intent_result.rationale}"
            ),
        )

        def _handle_search(state: Dict[str, Any]) -> ScreeningResult:
            query = intent_result.extracted_query or (idea.to_query_text() if idea else user_request)
            queries_used, new_docs, warns, rationale = self.search_agent.run(
                idea=idea,
                explicit_query="" if idea else query,
                sources=self.sources,
                max_per_source=self.max_per_source,
                patentsview_api_key=self.patentsview_api_key,
            )
            warnings.extend(warns)
            self._add_trace(
                traces=traces,
                agent="Search Agent",
                action="Multi-Source Search + ChromaDB Indexing",
                decision_or_output=f"Queries: {queries_used} | {rationale} | Indexed {len(new_docs)} records.",
                retrieved_ids=[d.doc_id for d in new_docs],
            )
            retrieved = self.vector_store.similarity_search(query, top_k=self.top_k)
            return ScreeningResult(
                intent_result=intent_result,
                search_queries_used=queries_used,
                newly_indexed_docs=new_docs,
                retrieved_docs=retrieved,
                traces=traces,
                warnings=warnings,
            )

        def _handle_comparison(state: Dict[str, Any]) -> ScreeningResult:
            active_idea = idea or InventionIdea(
                title=user_request[:80],
                problem=user_request,
                solution=user_request,
            )
            retrieved, new_docs, queries_used = self._agentic_rag_retrieve(
                query_text=active_idea.to_query_text(),
                idea=active_idea,
                traces=traces,
                warnings=warnings,
            )
            comparison = self.comparison_agent.run(active_idea, retrieved)
            self._add_trace(
                traces=traces,
                agent="Novelty Comparison Agent",
                action=f"Feature-Level Overlap Assessment ({self.prompt_technique})",
                decision_or_output=(
                    f"Overlap Risk: {comparison.overall_overlap_risk}. {comparison.summary}"
                ),
                retrieved_ids=[d.doc_id for d in retrieved],
            )
            return ScreeningResult(
                intent_result=intent_result,
                search_queries_used=queries_used,
                newly_indexed_docs=new_docs,
                retrieved_docs=retrieved,
                comparison=comparison,
                traces=traces,
                warnings=warnings,
            )

        def _handle_gap(state: Dict[str, Any]) -> ScreeningResult:
            active_idea = idea or InventionIdea(
                title=user_request[:80],
                problem=user_request,
                solution=user_request,
            )
            retrieved, new_docs, queries_used = self._agentic_rag_retrieve(
                query_text=active_idea.to_query_text(),
                idea=active_idea,
                traces=traces,
                warnings=warnings,
            )
            comparison = self.comparison_agent.run(active_idea, retrieved)
            self._add_trace(
                traces=traces,
                agent="Novelty Comparison Agent",
                action=f"Baseline Overlap Check ({self.prompt_technique})",
                decision_or_output=f"Overlap Risk: {comparison.overall_overlap_risk}.",
                retrieved_ids=[d.doc_id for d in retrieved],
            )
            gap = self.gap_agent.run(active_idea, retrieved, comparison)
            self._add_trace(
                traces=traces,
                agent="Gap Agent",
                action=f"Identify Novelty Gaps & White Space ({self.prompt_technique})",
                decision_or_output=f"Identified {len(gap.novel_gaps)} gaps. {gap.summary}",
                retrieved_ids=[d.doc_id for d in retrieved],
            )
            return ScreeningResult(
                intent_result=intent_result,
                search_queries_used=queries_used,
                newly_indexed_docs=new_docs,
                retrieved_docs=retrieved,
                comparison=comparison,
                gap_analysis=gap,
                traces=traces,
                warnings=warnings,
            )

        def _handle_full_report(state: Dict[str, Any]) -> ScreeningResult:
            active_idea = idea or InventionIdea(
                title=user_request[:80],
                problem=user_request,
                solution=user_request,
            )
            # Step 1: Parallel Search Agent
            self._notify(0.18, "Searching registered patents, academic papers & online products in parallel...")
            queries_used, new_docs, warns, rationale = self.search_agent.run(
                idea=active_idea,
                sources=self.sources,
                max_per_source=self.max_per_source,
                patentsview_api_key=self.patentsview_api_key,
                max_queries=2 if self.fast_mode else 3,
            )
            warnings.extend(warns)
            self._add_trace(
                traces=traces,
                agent="Search Agent",
                action="Parallel Patent, Paper & Product Search + ChromaDB Upsert",
                decision_or_output=f"Queries: {queries_used} | {rationale} | Indexed {len(new_docs)} documents.",
                retrieved_ids=[d.doc_id for d in new_docs],
            )

            # Step 2: ChromaDB Semantic Retrieval
            self._notify(0.45, "Retrieving closest prior art via ChromaDB vector similarity...")
            retrieved = self.vector_store.similarity_search(active_idea.to_query_text(), top_k=self.top_k)
            self._add_trace(
                traces=traces,
                agent="ChromaDB Vector Retriever",
                action="Semantic Similarity Search",
                decision_or_output=f"Retrieved top-{len(retrieved)} chunks from ChromaDB (total={self.vector_store.count()}).",
                retrieved_ids=[d.doc_id for d in retrieved],
            )

            # Step 3 & 4: Novelty Comparison (§102/§103 + FTO) + Gap Agent
            if self.fast_mode:
                self._notify(0.68, "Running §102/§103 Comparison, FTO Matrix & Gap Agents in parallel...")
                with ThreadPoolExecutor(max_workers=2) as pool:
                    f_comp = pool.submit(self.comparison_agent.run, active_idea, retrieved)
                    f_gap = pool.submit(self.gap_agent.run, active_idea, retrieved, None)
                    comparison = f_comp.result()
                    gap = f_gap.result()
            else:
                self._notify(0.60, "Running Novelty Comparison (§102/§103 + FTO) Agent...")
                comparison = self.comparison_agent.run(active_idea, retrieved)
                self._notify(0.75, "Running Novelty Gap & Sufficiency Agent...")
                gap = self.gap_agent.run(active_idea, retrieved, comparison)

            self._add_trace(
                traces=traces,
                agent="Novelty Comparison Agent",
                action=f"Feature Overlap & FTO Assessment ({self.prompt_technique})",
                decision_or_output=f"Overall Risk: {comparison.overall_overlap_risk} | FTO: {comparison.fto_risk_level}. {comparison.summary}",
                retrieved_ids=[d.doc_id for d in retrieved],
            )
            self._add_trace(
                traces=traces,
                agent="Gap Agent",
                action=f"Novelty Gap & Sufficiency Check ({self.prompt_technique})",
                decision_or_output=(
                    f"Gaps: {len(gap.novel_gaps)} | Sufficiency: {gap.evidence_sufficiency}. {gap.summary}"
                ),
                retrieved_ids=[d.doc_id for d in retrieved],
            )

            if (not self.fast_mode or len(retrieved) < 2) and gap.evidence_sufficiency == "Needs_More_Search" and gap.followup_search_queries:
                extra_q, extra_docs, extra_warns, _ = self.search_agent.run(
                    explicit_query=gap.followup_search_queries[0],
                    sources=self.sources,
                    max_per_source=3,
                    patentsview_api_key=self.patentsview_api_key,
                    max_queries=1,
                )
                warnings.extend(extra_warns)
                queries_used.extend(extra_q)
                new_docs.extend(extra_docs)
                retrieved = self.vector_store.similarity_search(active_idea.to_query_text(), top_k=self.top_k)
                self._add_trace(
                    traces=traces,
                    agent="Agentic RAG Follow-up",
                    action="Secondary Search Triggered by Gap Agent",
                    decision_or_output=f"Executed follow-up query {extra_q}; indexed {len(extra_docs)} more docs.",
                    retrieved_ids=[d.doc_id for d in retrieved],
                )

            # Step 5: Report Agent
            self._notify(0.88, "Drafting Formal Patentability & FTO Screening Memorandum...")
            report_md = self.report_agent.run(
                idea=active_idea,
                search_queries=queries_used,
                retrieved_docs=retrieved,
                comparison=comparison,
                gap_analysis=gap,
            )
            self._add_trace(
                traces=traces,
                agent="Report Agent",
                action=f"Draft Cited Memorandum ({self.prompt_technique})",
                decision_or_output=f"Generated {len(report_md)} chars of cited Markdown screening report.",
                retrieved_ids=[d.doc_id for d in retrieved],
            )
            self._notify(1.0, "Screening complete!")

            return ScreeningResult(
                intent_result=intent_result,
                search_queries_used=queries_used,
                newly_indexed_docs=new_docs,
                retrieved_docs=retrieved,
                comparison=comparison,
                gap_analysis=gap,
                report_markdown=report_md,
                traces=traces,
                warnings=warnings,
            )

        def _handle_followup_qa(state: Dict[str, Any]) -> ScreeningResult:
            query_for_rag = intent_result.extracted_query or user_request
            retrieved, new_docs, queries_used = self._agentic_rag_retrieve(
                query_text=query_for_rag,
                idea=idea,
                traces=traces,
                warnings=warnings,
                allow_external_search=True,
            )
            answer = self.qa_chain.invoke(
                {
                    "question": user_request,
                    "title": idea.title if idea else "General Prior-Art Session",
                    "context": format_docs_for_prompt(retrieved),
                    "report_excerpt": report_context_excerpt[:1500] or "No prior report drafted yet.",
                }
            )
            self._add_trace(
                traces=traces,
                agent="Agentic RAG Q&A",
                action=f"Grounded Answer Synthesis ({self.prompt_technique})",
                decision_or_output=answer[:300],
                retrieved_ids=[d.doc_id for d in retrieved],
            )
            return ScreeningResult(
                intent_result=intent_result,
                search_queries_used=queries_used,
                newly_indexed_docs=new_docs,
                retrieved_docs=retrieved,
                qa_answer=answer,
                traces=traces,
                warnings=warnings,
            )

        router_branch = build_intent_router_branch(
            {
                "prior_art_search": _handle_search,
                "novelty_comparison": _handle_comparison,
                "gap_analysis": _handle_gap,
                "full_screening_report": _handle_full_report,
                "followup_qa": _handle_followup_qa,
            }
        )
        return router_branch.invoke({"intent_result": intent_result})

