"""Automated tests for PatentQ backend, LCEL chains, ChromaDB, Agentic RAG, and Streamlit UI."""

from __future__ import annotations

from pathlib import Path

from src.agents.orchestrator import PatentQOrchestrator
from src.config import SAMPLE_INVENTIONS
from src.llm_factory import (
    PatentQDemoChatModel,
    extract_json_from_text,
    verify_llm_connection,
)
from src.models import InventionIdea, PriorArtDocument
from src.prompts import (
    build_comparison_prompt,
    build_gap_prompt,
    build_intent_prompt,
    get_prompt_catalog,
)
from src.router import detect_intent_lcel
from src.vector_store import ChromaPriorArtStore


def _sample_idea() -> InventionIdea:
    preset = next(iter(SAMPLE_INVENTIONS.values()))
    return InventionIdea(**preset)


def _sample_docs() -> list[PriorArtDocument]:
    return [
        PriorArtDocument(
            doc_id="PAT-US10000001",
            doc_type="patent",
            title="Portable Solar Vaccine Refrigerator with Battery Backup",
            abstract="A portable solar-powered medical cooler using lithium batteries and Peltier thermoelectric cooling.",
            authors_or_assignee="MedCool Corp",
            year="2022",
            url="https://patents.google.com/patent/US10000001/en",
            source_api="Google Patents",
        ),
        PriorArtDocument(
            doc_id="ARXIV-2401.99999",
            doc_type="paper",
            title="Paraffin Phase-Change Materials for Cold-Chain Logistics",
            abstract="Evaluates dual-layer paraffin PCM jackets maintaining 2-8 C thermal stability during transit.",
            authors_or_assignee="A. Researcher et al.",
            year="2024",
            url="https://arxiv.org/abs/2401.99999",
            source_api="arXiv",
        ),
        PriorArtDocument(
            doc_id="PROD-MIC-12345678",
            doc_type="product",
            title="Portable 12V Solar Medical Vaccine Cooler Box",
            abstract="Commercial product listed on Made-in-China (US$ 230.00).",
            authors_or_assignee="Made-in-China Marketplace",
            year="Active Listing",
            url="https://example.com/product/solar-vaccine-cooler",
            image_url="https://example.com/images/solar-cooler.jpg",
            price="US$ 230.00",
            source_api="Made-in-China",
        ),
    ]


def test_prompt_templates_and_catalog():
    catalog = get_prompt_catalog("CoT")
    assert "Novelty Comparison Agent" in catalog
    assert "system" in catalog["Novelty Comparison Agent"]
    assert "user" in catalog["Novelty Comparison Agent"]

    for tech in ["CoT", "Few-Shot", "ReAct", "Reflection"]:
        p = build_comparison_prompt(tech)
        formatted = p.format_messages(
            title="Test",
            domain="Tech",
            problem="Problem statement",
            solution="Solution mechanism",
            components="Comp A",
            context="[PAT-1] Sample",
        )
        assert len(formatted) >= 2

    assert build_intent_prompt() is not None
    assert build_gap_prompt("Reflection") is not None


def test_chromadb_crud_and_similarity_search():
    store = ChromaPriorArtStore(persist_dir=None, collection_name="test_patentq_col")
    assert store.count() == 0

    docs = _sample_docs()
    inserted = store.upsert_documents(docs)
    assert inserted == len(docs)
    assert store.count() == len(docs)

    hits = store.similarity_search("solar vaccine cooler phase change material", top_k=2)
    assert len(hits) == 2
    assert hits[0].similarity_score is not None

    patent_only = store.similarity_search("vaccine cooler", top_k=5, doc_type_filter="patent")
    assert len(patent_only) == 1
    assert patent_only[0].doc_type == "patent"

    product_only = store.similarity_search("solar cooler", top_k=5, doc_type_filter="product")
    assert len(product_only) == 1
    assert product_only[0].image_url.startswith("https://")

    store.delete_document("PAT-US10000001")
    assert store.count() == len(docs) - 1

    store.reset()
    assert store.count() == 0


def test_intent_detection_and_routing():
    llm = PatentQDemoChatModel()

    r1 = detect_intent_lcel(llm, "Only search for patents on solar vaccine coolers", False, 0)
    assert r1.intent == "prior_art_search"

    r2 = detect_intent_lcel(llm, "Compare this invention against indexed prior art", True, 5)
    assert r2.intent == "novelty_comparison"

    r3 = detect_intent_lcel(llm, "Find novelty gaps and white space", True, 5)
    assert r3.intent == "gap_analysis"

    r4 = detect_intent_lcel(llm, "Which retrieved patent covers PCM cooling?", True, 5)
    assert r4.intent == "followup_qa"

    r5 = detect_intent_lcel(llm, "Screen my invention", True, 5, mode_override="Full Screening Report")
    assert r5.intent == "full_screening_report"
    assert r5.confidence == 1.0


def test_orchestrator_end_to_end_with_seeded_chroma(monkeypatch):
    store = ChromaPriorArtStore(persist_dir=None, collection_name="test_orchestrator_col")
    store.upsert_documents(_sample_docs())

    monkeypatch.setattr(
        "src.agents.search_agent.run_multi_source_search",
        lambda queries, sources=None, max_per_source=4, patentsview_api_key=None: (_sample_docs(), []),
    )

    llm = PatentQDemoChatModel()
    orch = PatentQOrchestrator(llm=llm, vector_store=store, prompt_technique="CoT")

    res = orch.run(
        user_request="Run full screening report",
        idea=_sample_idea(),
        mode_override="Full Screening Report",
    )
    assert res.intent_result.intent == "full_screening_report"
    assert len(res.retrieved_docs) >= 2
    assert res.comparison is not None
    assert res.comparison.novelty_102_assessment
    assert res.comparison.obviousness_103_assessment
    assert len(res.comparison.fto_items) >= 1
    assert res.gap_analysis is not None
    assert "Formal Patentability & FTO Screening Memorandum" in res.report_markdown
    assert len(res.traces) >= 5

    qa_res = orch.run(
        user_request="Which retrieved reference discusses paraffin PCM?",
        idea=_sample_idea(),
        mode_override="Follow-up Q&A",
        report_context_excerpt=res.report_markdown,
    )
    assert qa_res.intent_result.intent == "followup_qa"
    assert qa_res.qa_answer


def test_llm_connection_and_json_extractor():
    ok, msg = verify_llm_connection("demo")
    assert ok is True
    assert "Connected" in msg

    parsed = extract_json_from_text('```json\n{"intent": "gap_analysis", "confidence": 0.9}\n```')
    assert parsed["intent"] == "gap_analysis"


def test_streamlit_app_buttons_and_validation(monkeypatch, tmp_path):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.cache_resource.clear()
    monkeypatch.setattr(
        "src.agents.search_agent.run_multi_source_search",
        lambda queries, sources=None, max_per_source=4, patentsview_api_key=None: (_sample_docs(), []),
    )
    monkeypatch.setattr("src.vector_store.CHROMA_PERSIST_DIR", str(tmp_path / "chroma_test"))

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=45)
    at.run()
    assert not at.exception

    # Switch sidebar provider to Offline Demo LLM for automated UI button checks
    at.sidebar.selectbox[0].set_value("demo").run()
    assert not at.exception

    # Sidebar: Test LLM Connection
    test_conn_btn = [b for b in at.sidebar.button if "Test LLM Connection" in b.label][0]
    test_conn_btn.click().run()
    assert not at.exception
    assert any("Connected" in s.value for s in at.sidebar.success)

    # Tab 1: Empty required fields should trigger validation warnings, not crash
    run_btn = [b for b in at.button if "Run Screening Pipeline" in b.label][0]
    run_btn.click().run()
    assert not at.exception
    assert len(at.warning) >= 1

    # Tab 1: Load Sample and Run Full Screening Pipeline
    load_btn = [b for b in at.button if "Load Sample" in b.label][0]
    load_btn.click().run()
    assert not at.exception

    run_btn = [b for b in at.button if "Run Screening Pipeline" in b.label][0]
    run_btn.click().run()
    assert not at.exception
    rendered_md = "\n".join(m.value for m in at.markdown)
    assert "LCEL Intent Detection & Agentic Routing" in rendered_md
    assert "Agentic Pipeline Flowchart" in rendered_md
    assert "Patentability &amp; FTO Matrices" in rendered_md or "Patentability & FTO Matrices" in rendered_md
    assert "Antigravity" in rendered_md
    assert "35 U.S.C." in rendered_md
    assert "Freedom-to-Operate (FTO)" in rendered_md
    assert "View Product" in rendered_md

    # Pin an overlap/gap/prior-art card into the Modular Memorandum Builder
    pin_btns = [b for b in at.button if "Pin to Memo" in b.label]
    assert len(pin_btns) >= 1
    pin_btns[0].click().run()
    assert not at.exception
    assert any("Pinned to Memo" in b.label for b in at.button)

    # Tab 1: Follow-up Q&A Send Request button
    followup_input = [ti for ti in at.text_input if ti.key == "followup_query"][0]
    followup_input.set_value("Which patent is closest to the solar PCM cooler?").run()
    send_btn = [b for b in at.button if "Send Request" in b.label][0]
    send_btn.click().run()
    assert not at.exception

    # Tab 2: ChromaDB Semantic Search button
    kb_input = [ti for ti in at.text_input if ti.key == "kb_search_query"][0]
    kb_input.set_value("vaccine cooler phase change").run()
    kb_search_btn = [b for b in at.button if b.key == "btn_kb_search"][0]
    kb_search_btn.click().run()
    assert not at.exception

    # Tab 1: Clear Form button
    clear_btn = [b for b in at.button if "Clear Form" in b.label][0]
    clear_btn.click().run()
    assert not at.exception

    # Sidebar: Reset ChromaDB Index button
    reset_btn = [b for b in at.sidebar.button if "Reset ChromaDB Index" in b.label][0]
    reset_btn.click().run()
    assert not at.exception
