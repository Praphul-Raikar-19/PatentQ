"""PatentQ — Prior-Art, Patentability (§102/§103) & FTO Registered-Patent Screener."""

from __future__ import annotations

import html
import json
import os
import time
import urllib.parse
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from src.agents.orchestrator import PatentQOrchestrator
from src.config import (
    APP_NAME,
    APP_SUBTITLE,
    DISCLAIMER_TEXT,
    LLM_PROVIDERS,
    PROMPT_TECHNIQUES,
    SAMPLE_INVENTIONS,
    load_custom_presets,
    save_custom_preset,
)
from src.llm_factory import (
    create_chat_model,
    format_llm_error,
    list_ollama_models,
    pick_default_provider,
    verify_llm_connection,
)
from src.models import InventionIdea, PriorArtDocument, ScreeningResult
from src.prompts import TECHNIQUE_DIRECTIVES, get_prompt_catalog
from src.tools.search_tools import get_langchain_tools
from src.vector_store import ChromaPriorArtStore

st.set_page_config(
    page_title=f"{APP_NAME} — {APP_SUBTITLE}",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_custom_css() -> None:
    """Inject Antigravity Neo-Glassmorphism styling, micro-animations, flowchart & heatmap CSS."""
    st.markdown(
        """
        <style>
        @keyframes pqPulseGlow {
            0%, 100% {
                box-shadow: 0 0 12px rgba(0, 191, 255, 0.35), inset 0 0 8px rgba(0, 191, 255, 0.12);
            }
            50% {
                box-shadow: 0 0 24px rgba(0, 191, 255, 0.75), inset 0 0 14px rgba(0, 191, 255, 0.25);
            }
        }
        @keyframes pqFloatSubtle {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-2px); }
        }

        /* Deep Space Canvas */
        .stApp {
            background:
                radial-gradient(circle at 15% 18%, rgba(0, 191, 255, 0.12), transparent 42%),
                radial-gradient(circle at 86% 12%, rgba(139, 92, 246, 0.14), transparent 44%),
                radial-gradient(circle at 75% 85%, rgba(16, 185, 129, 0.08), transparent 42%),
                linear-gradient(155deg, #070B16 0%, #131B32 52%, #0B1224 100%);
            color: #E2E8F0;
        }
        .block-container {
            padding-top: 1.35rem;
            padding-bottom: 2.75rem;
            max-width: 1360px;
        }

        /* Glassmorphic Sidebar */
        [data-testid="stSidebar"] {
            background: rgba(11, 17, 34, 0.82) !important;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-right: 1px solid rgba(0, 191, 255, 0.22);
            box-shadow: 6px 0 30px rgba(0, 0, 0, 0.45);
        }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #F8FAFC !important;
        }

        /* Native Inputs & Containers Dark Glass Treatment */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.035) !important;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(0, 191, 255, 0.2) !important;
            border-radius: 16px !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
            transition: border-color 0.25s ease, box-shadow 0.25s ease;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: rgba(0, 191, 255, 0.42) !important;
        }

        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div,
        .stMultiSelect > div > div,
        .stNumberInput > div > div > input {
            background: rgba(15, 23, 42, 0.72) !important;
            color: #F1F5F9 !important;
            border: 1px solid rgba(148, 163, 184, 0.28) !important;
            border-radius: 10px !important;
        }
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: rgba(0, 191, 255, 0.8) !important;
            box-shadow: 0 0 0 2px rgba(0, 191, 255, 0.2) !important;
        }

        /* Micro-Animation Buttons */
        .stButton > button, .stDownloadButton > button {
            background: rgba(255, 255, 255, 0.06);
            color: #F8FAFC;
            border: 1px solid rgba(0, 191, 255, 0.32);
            border-radius: 11px;
            font-weight: 600;
            backdrop-filter: blur(8px);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease, background 0.2s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            transform: translateY(-3px);
            border-color: rgba(0, 191, 255, 0.9);
            box-shadow: 0 8px 22px rgba(0, 191, 255, 0.3);
            background: rgba(0, 191, 255, 0.14);
            color: #FFFFFF;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, rgba(0, 191, 255, 0.85) 0%, rgba(79, 70, 229, 0.9) 100%);
            border: 1px solid rgba(125, 211, 252, 0.85);
            color: #FFFFFF;
            box-shadow: 0 6px 20px rgba(0, 191, 255, 0.35);
        }
        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 10px 28px rgba(0, 191, 255, 0.55);
            transform: translateY(-4px);
        }

        /* Neo-Glass Hero Header */
        .pq-hero {
            background: linear-gradient(135deg, rgba(19, 27, 50, 0.86) 0%, rgba(23, 37, 84, 0.72) 55%, rgba(15, 23, 42, 0.88) 100%);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            color: #FFFFFF;
            padding: 1.45rem 1.75rem;
            border-radius: 18px;
            box-shadow: 0 14px 40px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.12);
            margin-bottom: 1.2rem;
            border: 1px solid rgba(0, 191, 255, 0.35);
        }
        .pq-hero-title {
            font-size: 1.85rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin: 0 0 0.38rem 0;
            background: linear-gradient(90deg, #FFFFFF 0%, #7DD3FC 60%, #C4B5FD 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .pq-hero-sub {
            font-size: 0.92rem;
            color: #CBD5E1;
            margin: 0 0 0.95rem 0;
            line-height: 1.48;
        }
        .pq-pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
        }
        .pq-pill {
            background: rgba(0, 191, 255, 0.12);
            border: 1px solid rgba(0, 191, 255, 0.36);
            color: #E0F2FE;
            padding: 0.28rem 0.78rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            box-shadow: 0 2px 10px rgba(0, 191, 255, 0.15);
        }

        /* Floating KPI Cards */
        .pq-kpi-card {
            background: rgba(255, 255, 255, 0.045);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-top: 3px solid rgba(0, 191, 255, 0.85);
            border-radius: 15px;
            padding: 0.95rem 1.05rem;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.38);
            height: 100%;
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        }
        .pq-kpi-card:hover {
            transform: translateY(-5px);
            border-color: rgba(0, 191, 255, 0.75);
            box-shadow: 0 14px 32px rgba(0, 191, 255, 0.24);
        }
        .pq-kpi-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #94A3B8;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .pq-kpi-value {
            font-size: 1.22rem;
            font-weight: 800;
            color: #F8FAFC;
            word-break: break-word;
        }

        /* Floating Glass Partition Boxes */
        .pq-box {
            background: rgba(255, 255, 255, 0.045);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(0, 191, 255, 0.22);
            border-radius: 16px;
            padding: 1.15rem 1.3rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.34);
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        }
        .pq-box:hover {
            transform: translateY(-4px);
            border-color: rgba(0, 191, 255, 0.65);
            box-shadow: 0 14px 34px rgba(0, 191, 255, 0.2);
        }
        .pq-box-accent-blue { border-left: 4px solid #00BFFF; }
        .pq-box-accent-indigo { border-left: 4px solid #818CF8; }
        .pq-box-accent-emerald { border-left: 4px solid #34D399; }
        .pq-box-accent-amber { border-left: 4px solid #FBBF24; }
        .pq-box-title {
            font-size: 1.03rem;
            font-weight: 700;
            color: #F8FAFC;
            margin: 0 0 0.45rem 0;
        }
        .pq-box-body {
            font-size: 0.91rem;
            color: #CBD5E1;
            line-height: 1.56;
            margin: 0;
        }

        /* Bioluminescent Badges */
        .pq-badge {
            display: inline-block;
            padding: 0.2rem 0.62rem;
            border-radius: 999px;
            font-size: 0.73rem;
            font-weight: 700;
            letter-spacing: 0.03em;
        }
        .pq-badge-high {
            background: rgba(239, 68, 68, 0.2);
            color: #FCA5A5;
            border: 1px solid rgba(248, 113, 113, 0.55);
        }
        .pq-badge-moderate {
            background: rgba(245, 158, 11, 0.2);
            color: #FDE68A;
            border: 1px solid rgba(251, 191, 36, 0.55);
        }
        .pq-badge-low {
            background: rgba(16, 185, 129, 0.2);
            color: #6EE7B7;
            border: 1px solid rgba(52, 211, 153, 0.55);
        }
        .pq-badge-patent {
            background: rgba(0, 191, 255, 0.18);
            color: #7DD3FC;
            border: 1px solid rgba(0, 191, 255, 0.55);
        }
        .pq-badge-paper {
            background: rgba(139, 92, 246, 0.22);
            color: #DDD6FE;
            border: 1px solid rgba(167, 139, 250, 0.55);
        }
        .pq-badge-product {
            background: rgba(249, 115, 22, 0.22);
            color: #FED7AA;
            border: 1px solid rgba(251, 146, 60, 0.55);
        }
        .pq-badge-custom {
            background: rgba(148, 163, 184, 0.18);
            color: #E2E8F0;
            border: 1px solid rgba(148, 163, 184, 0.4);
        }

        /* Floating Document & Antigravity Product Cards */
        .pq-doc-card {
            background: rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.11);
            border-radius: 14px;
            padding: 0.95rem 1.1rem;
            margin-bottom: 0.55rem;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.28);
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        }
        .pq-doc-card:hover {
            transform: translateY(-4px);
            border-color: rgba(0, 191, 255, 0.6);
            box-shadow: 0 12px 28px rgba(0, 191, 255, 0.2);
        }
        .pq-doc-header {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.45rem;
            margin-bottom: 0.35rem;
        }
        .pq-doc-title {
            font-size: 0.97rem;
            font-weight: 700;
            color: #F8FAFC;
            margin: 0.15rem 0 0.25rem 0;
        }
        .pq-doc-meta {
            font-size: 0.79rem;
            color: #94A3B8;
            margin-bottom: 0.4rem;
        }
        .pq-doc-meta a {
            color: #38BDF8 !important;
            text-decoration: none;
        }
        .pq-doc-abs {
            font-size: 0.87rem;
            color: #CBD5E1;
            line-height: 1.5;
        }

        .pq-prod-card {
            background: linear-gradient(165deg, rgba(255, 255, 255, 0.065) 0%, rgba(15, 23, 42, 0.78) 100%);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(0, 191, 255, 0.28);
            border-radius: 16px;
            padding: 0.95rem;
            margin-bottom: 0.6rem;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.42);
            transition: transform 0.28s ease, box-shadow 0.28s ease, border-color 0.28s ease;
        }
        .pq-prod-card:hover {
            transform: translateY(-6px);
            border-color: rgba(0, 191, 255, 0.85);
            box-shadow: 0 16px 36px rgba(0, 191, 255, 0.3);
        }
        .pq-prod-img-wrap {
            width: 100%;
            height: 175px;
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid rgba(0, 191, 255, 0.3);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            margin-bottom: 0.7rem;
        }
        .pq-prod-img-wrap img {
            max-width: 100%;
            max-height: 168px;
            object-fit: contain;
        }
        .pq-prod-codename {
            font-size: 0.98rem;
            font-weight: 800;
            color: #7DD3FC;
            margin: 0.25rem 0 0.15rem 0;
            line-height: 1.3;
        }
        .pq-prod-orig {
            font-size: 0.78rem;
            color: #94A3B8;
            margin-bottom: 0.45rem;
        }
        .pq-prod-link {
            display: inline-block;
            margin-top: 0.35rem;
            padding: 0.36rem 0.82rem;
            background: linear-gradient(135deg, #0284C7 0%, #4F46E5 100%);
            color: #FFFFFF !important;
            border: 1px solid rgba(125, 211, 252, 0.55);
            border-radius: 9px;
            font-size: 0.8rem;
            font-weight: 700;
            text-decoration: none;
            box-shadow: 0 4px 14px rgba(0, 191, 255, 0.25);
        }

        /* Agentic Pipeline Flowchart */
        .pq-flow-card {
            background: rgba(255, 255, 255, 0.045);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(0, 191, 255, 0.26);
            border-radius: 16px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 10px 32px rgba(0, 0, 0, 0.38);
            margin-bottom: 1rem;
        }
        .pq-flow-grid {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: center;
            gap: 0.55rem;
            margin-top: 0.75rem;
            margin-bottom: 0.4rem;
        }
        .pq-flow-node {
            padding: 0.55rem 0.9rem;
            border-radius: 12px;
            font-size: 0.82rem;
            font-weight: 700;
            text-align: center;
            min-width: 138px;
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }
        .pq-flow-node:hover {
            transform: translateY(-4px);
        }
        .pq-flow-active-cyan {
            background: rgba(0, 191, 255, 0.16);
            border: 1.5px solid rgba(0, 191, 255, 0.85);
            color: #E0F2FE;
            animation: pqPulseGlow 2.8s infinite ease-in-out;
        }
        .pq-flow-active-green {
            background: rgba(16, 185, 129, 0.16);
            border: 1.5px solid rgba(52, 211, 153, 0.85);
            color: #D1FAE5;
            box-shadow: 0 0 14px rgba(16, 185, 129, 0.32);
        }
        .pq-flow-active-purple {
            background: rgba(139, 92, 246, 0.18);
            border: 1.5px solid rgba(167, 139, 250, 0.85);
            color: #EDE9FE;
            box-shadow: 0 0 14px rgba(139, 92, 246, 0.34);
        }
        .pq-flow-standby {
            background: rgba(148, 163, 184, 0.08);
            border: 1px dashed rgba(148, 163, 184, 0.35);
            color: #94A3B8;
        }
        .pq-flow-arrow {
            color: #38BDF8;
            font-size: 1.1rem;
            font-weight: 800;
        }

        /* Patentability Heatmap Matrix */
        .pq-matrix-wrap {
            background: rgba(255, 255, 255, 0.045);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(0, 191, 255, 0.25);
            border-radius: 16px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.36);
            margin-bottom: 1rem;
        }
        .pq-matrix-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 6px;
            margin-top: 0.45rem;
        }
        .pq-matrix-table th {
            color: #94A3B8;
            font-size: 0.77rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 0.35rem;
            text-align: center;
        }
        .pq-matrix-row-lbl {
            color: #CBD5E1;
            font-size: 0.82rem;
            font-weight: 700;
            text-align: right;
            padding-right: 0.65rem;
            white-space: nowrap;
        }
        .pq-mcell {
            border-radius: 8px;
            padding: 0.48rem 0.4rem;
            text-align: center;
            font-size: 0.8rem;
            font-weight: 700;
        }
        .pq-mcell-green {
            background: rgba(16, 185, 129, 0.32);
            border: 1px solid rgba(52, 211, 153, 0.7);
            color: #A7F3D0;
        }
        .pq-mcell-amber {
            background: rgba(245, 158, 11, 0.28);
            border: 1px solid rgba(251, 191, 36, 0.7);
            color: #FDE68A;
        }
        .pq-mcell-red {
            background: rgba(239, 68, 68, 0.3);
            border: 1px solid rgba(248, 113, 113, 0.7);
            color: #FECACA;
        }
        .pq-mcell-muted {
            background: rgba(148, 163, 184, 0.14);
            border: 1px solid rgba(148, 163, 184, 0.25);
            color: #64748B;
        }

        /* Memorandum Banner */
        .pq-memo-banner {
            background: rgba(0, 191, 255, 0.1);
            border: 1px solid rgba(0, 191, 255, 0.35);
            border-left: 5px solid #00BFFF;
            border-radius: 12px;
            padding: 0.9rem 1.1rem;
            color: #E0F2FE;
            font-size: 0.88rem;
            margin-bottom: 1rem;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.55rem;
            background: rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(10px);
            padding: 0.45rem;
            border-radius: 14px;
            border: 1px solid rgba(0, 191, 255, 0.2);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding: 0.48rem 1rem;
            font-weight: 600;
            color: #CBD5E1;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(0, 191, 255, 0.18) !important;
            color: #FFFFFF !important;
            border: 1px solid rgba(0, 191, 255, 0.55) !important;
            box-shadow: 0 0 14px rgba(0, 191, 255, 0.24);
        }

        /* Expanders */
        .streamlit-expanderHeader {
            background: rgba(255, 255, 255, 0.04) !important;
            border-radius: 10px !important;
            color: #F8FAFC !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _badge_html(label: str, variant: str = "moderate") -> str:
    safe_label = html.escape(label)
    return f'<span class="pq-badge pq-badge-{variant}">{safe_label}</span>'


def _risk_variant(level: str) -> str:
    norm = (level or "").strip().lower()
    if norm in {"high", "strong"}:
        return "high" if norm == "high" else "low"
    if norm in {"low", "none"}:
        return "low"
    return "moderate"


def _rebrand_antigravity_product(title: str) -> str:
    """Apply Antigravity engineering codenames to commercial products while preserving source facts."""
    low = (title or "").lower()
    if any(k in low for k in ["sort", "bin", "waste", "segregat", "trash", "recycl"]):
        return "Grab-ley Gravity-Defying Dry/Wet Segregation System"
    if any(k in low for k in ["cooler", "vaccine", "refrigerat", "thermal", "pcm", "cold", "cryo", "freezer"]):
        return "Antigravity Cryo-Shield™ Zero-Loss Thermal Suspension Unit"
    if any(k in low for k in ["microplastic", "acoustic", "ultrasonic", "water", "filter", "sensor"]):
        return "Antigravity Sonic-Trap™ Suspended Particle Array"
    if any(k in low for k in ["grid", "solar", "inverter", "battery", "microgrid", "edge", "iot"]):
        return "Antigravity Tensor-Pulse™ Autonomous Energy Node"
    if any(k in low for k in ["drone", "uav", "farm", "agri", "rover", "aerial"]):
        return "Antigravity Aero-Glide™ Autonomous Field Platform"
    clean_short = (title or "Commercial Hardware").strip()[:46]
    return f"Antigravity Aero-Core™ {clean_short}"


@st.cache_resource
def get_vector_store() -> ChromaPriorArtStore:
    """Initialize and cache the persistent ChromaDB store."""
    return ChromaPriorArtStore()


@st.cache_data(ttl=300, show_spinner=False)
def _cached_chroma_search(_store: ChromaPriorArtStore, query: str, top_k: int, doc_type_filter: Optional[str]) -> List[PriorArtDocument]:
    """Cache ChromaDB similarity search results for near-instant interactive browsing."""
    return _store.similarity_search(query, top_k=top_k, doc_type_filter=doc_type_filter)


CUSTOM_ENTRY_LABEL = "✍️ Custom / Blank Participant Entry (Write Your Own)"


def _get_all_presets() -> Dict[str, Dict[str, str]]:
    merged: Dict[str, Dict[str, str]] = {}
    merged.update(SAMPLE_INVENTIONS)
    merged.update(load_custom_presets())
    return merged


def _init_session_state() -> None:
    if "provider_key" not in st.session_state:
        auto_prov, auto_model, auto_key = pick_default_provider()
    else:
        auto_prov = st.session_state["provider_key"]
        auto_model = st.session_state.get("model_name", LLM_PROVIDERS[auto_prov].default_model)
        auto_key = st.session_state.get("api_key", "")

    defaults = {
        "idea_title": "",
        "idea_domain": "General Engineering",
        "idea_problem": "",
        "idea_solution": "",
        "idea_components": "",
        "idea_keywords": "",
        "sample_choice": list(SAMPLE_INVENTIONS.keys())[0],
        "provider_key": auto_prov,
        "model_name": auto_model,
        "api_key": auto_key,
        "base_url": LLM_PROVIDERS[auto_prov].default_base_url,
        "patentsview_key": os.getenv("PATENTSVIEW_API_KEY", ""),
        "prompt_technique": "CoT",
        "fast_mode": True,
        "selected_sources": [
            "EuropePMC Patents",
            "Google Patents",
            "arXiv",
            "OpenAlex",
            "Crossref",
            "Online Shopping Products",
        ],
        "top_k": 6,
        "max_per_source": 3,
        "mode_override": "Auto-Detect",
        "followup_query": "",
        "last_result": None,
        "last_elapsed_s": None,
        "preset_feedback": None,
        "llm_status": None,
        "chroma_search_results": None,
        "kb_feedback": None,
        "pinned_memo_items": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _is_pinned(item_id: str) -> bool:
    return any(p.get("id") == item_id for p in st.session_state.get("pinned_memo_items", []))


def _toggle_pin_item(item_id: str, category: str, title: str, detail: str) -> None:
    pinned: List[Dict[str, str]] = list(st.session_state.get("pinned_memo_items", []))
    if any(p.get("id") == item_id for p in pinned):
        pinned = [p for p in pinned if p.get("id") != item_id]
    else:
        pinned.append({"id": item_id, "category": category, "title": title, "detail": detail})
    st.session_state["pinned_memo_items"] = pinned


def _render_pin_button(item_id: str, category: str, title: str, detail: str, key_suffix: str) -> None:
    already = _is_pinned(item_id)
    lbl = "✅ Pinned to Memo" if already else "📌 Pin to Memo"
    st.button(
        lbl,
        key=f"pin_{item_id}_{key_suffix}",
        on_click=_toggle_pin_item,
        args=(item_id, category, title, detail),
    )


def _build_custom_pinned_markdown(result: Optional[ScreeningResult]) -> str:
    pinned: List[Dict[str, str]] = st.session_state.get("pinned_memo_items", [])
    title = st.session_state.get("idea_title") or "Untitled Invention"
    lines = [
        f"# Custom Modular Patentability & FTO Memorandum — {title}",
        f"*Assembled via PatentQ Antigravity Pinning Workbench · {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        f"> {DISCLAIMER_TEXT}",
        "",
    ]
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for item in pinned:
        grouped.setdefault(item["category"], []).append(item)

    for cat, entries in grouped.items():
        lines.append(f"## {cat}")
        for entry in entries:
            lines.append(f"### {entry['title']}")
            lines.append(entry["detail"])
            lines.append("")
    return "\n".join(lines)


def _on_provider_change() -> None:
    prov = st.session_state["provider_key"]
    spec = LLM_PROVIDERS[prov]
    st.session_state["model_name"] = spec.default_model
    st.session_state["base_url"] = spec.default_base_url
    st.session_state["api_key"] = os.getenv(spec.env_var, "") if spec.env_var else ""
    st.session_state["llm_status"] = None


def _on_load_sample() -> None:
    choice = st.session_state["sample_choice"]
    if choice == CUSTOM_ENTRY_LABEL:
        _on_clear_form()
        return
    sample = _get_all_presets().get(choice, {})
    if sample:
        st.session_state["idea_title"] = sample.get("title", "")
        st.session_state["idea_domain"] = sample.get("domain", "General Engineering")
        st.session_state["idea_problem"] = sample.get("problem", "")
        st.session_state["idea_solution"] = sample.get("solution", "")
        st.session_state["idea_components"] = sample.get("components", "")
        st.session_state["idea_keywords"] = sample.get("keywords", "")


def _on_clear_form() -> None:
    st.session_state["idea_title"] = ""
    st.session_state["idea_domain"] = "General Engineering"
    st.session_state["idea_problem"] = ""
    st.session_state["idea_solution"] = ""
    st.session_state["idea_components"] = ""
    st.session_state["idea_keywords"] = ""


def _on_save_preset() -> None:
    title = st.session_state["idea_title"].strip()
    problem = st.session_state["idea_problem"].strip()
    solution = st.session_state["idea_solution"].strip()
    if len(title) < 3 or len(problem) < 10 or len(solution) < 10:
        st.session_state["preset_feedback"] = ("warn", "Enter Title, Problem Statement, and Proposed Solution before saving as a preset.")
        return
    preset_label = f"⭐ {title[:70]}"
    save_custom_preset(
        preset_label,
        {
            "title": title,
            "domain": st.session_state["idea_domain"].strip() or "General Engineering",
            "problem": problem,
            "solution": solution,
            "components": st.session_state["idea_components"].strip(),
            "keywords": st.session_state["idea_keywords"].strip(),
        },
    )
    st.session_state["sample_choice"] = preset_label
    st.session_state["preset_feedback"] = ("ok", f"Saved '{preset_label}' to participant presets!")


def _validate_idea_fields(require_full: bool = True) -> tuple[Optional[InventionIdea], List[str]]:
    errors: List[str] = []
    title = st.session_state["idea_title"].strip()
    problem = st.session_state["idea_problem"].strip()
    solution = st.session_state["idea_solution"].strip()
    components = st.session_state["idea_components"].strip()
    keywords = st.session_state["idea_keywords"].strip()
    domain = st.session_state["idea_domain"].strip() or "General Engineering"

    if len(title) < 3:
        errors.append("**Invention Title*** is required (minimum 3 characters).")
    if require_full:
        if len(problem) < 10:
            errors.append("**Problem Statement*** is required (minimum 10 characters).")
        if len(solution) < 10:
            errors.append("**Proposed Technical Solution / Mechanism*** is required (minimum 10 characters).")

    if errors:
        return None, errors

    return (
        InventionIdea(
            title=title,
            problem=problem if len(problem) >= 10 else title + " — prior art screening.",
            solution=solution if len(solution) >= 10 else title + " — prior art screening.",
            components=components,
            keywords=keywords,
            domain=domain,
        ),
        [],
    )


def _validate_llm_config() -> Optional[str]:
    prov = st.session_state["provider_key"]
    spec = LLM_PROVIDERS[prov]
    entered_key = st.session_state["api_key"].strip()
    env_key = os.getenv(spec.env_var, "").strip() if spec.env_var else ""
    base_url = st.session_state["base_url"].strip()
    if spec.requires_api_key and not entered_key and not env_key:
        if prov == "openai" and base_url:
            return None
        return (
            f"**API Key*** is required for **{spec.label}** "
            f"(enter it in the sidebar or set `{spec.env_var}` in `.env`, or switch to Ollama / Offline Demo LLM)."
        )
    if not st.session_state["model_name"].strip():
        return "**Model Name*** cannot be empty."
    return None


def _execute_pipeline(
    store: ChromaPriorArtStore,
    user_request: str,
    idea: Optional[InventionIdea],
    mode_override: str,
) -> Optional[ScreeningResult]:
    config_err = _validate_llm_config()
    if config_err:
        st.error(config_err)
        return None

    if not st.session_state["selected_sources"]:
        st.warning("No external prior-art sources selected in the sidebar; using existing ChromaDB index only.")

    try:
        llm = create_chat_model(
            provider=st.session_state["provider_key"],
            model_name=st.session_state["model_name"],
            api_key=st.session_state["api_key"],
            base_url=st.session_state["base_url"],
        )
        previous_report = ""
        if st.session_state["last_result"] and st.session_state["last_result"].report_markdown:
            previous_report = st.session_state["last_result"].report_markdown

        progress_bar = st.progress(0.05, text="Initializing PatentQ LCEL Router & Agentic RAG...")
        t_start = time.perf_counter()

        def _on_progress(pct: float, msg: str) -> None:
            progress_bar.progress(min(1.0, max(0.0, pct)), text=msg)

        orchestrator = PatentQOrchestrator(
            llm=llm,
            vector_store=store,
            prompt_technique=st.session_state["prompt_technique"],
            sources=st.session_state["selected_sources"],
            top_k=st.session_state["top_k"],
            max_per_source=st.session_state["max_per_source"],
            patentsview_api_key=st.session_state["patentsview_key"],
            fast_mode=st.session_state["fast_mode"],
            progress_cb=_on_progress,
        )
        result = orchestrator.run(
            user_request=user_request,
            idea=idea,
            mode_override=mode_override,
            report_context_excerpt=previous_report,
        )
        progress_bar.empty()
        st.session_state["last_elapsed_s"] = round(time.perf_counter() - t_start, 2)
        st.session_state["last_result"] = result
        return result
    except Exception as exc:
        friendly = format_llm_error(
            exc,
            st.session_state["provider_key"],
            st.session_state["model_name"],
        )
        st.error(f"Pipeline execution error: {friendly}")
        return None


def _render_sidebar(store: ChromaPriorArtStore) -> None:
    with st.sidebar:
        st.markdown(f"## ⚖️ {APP_NAME}")
        st.caption("Antigravity Prior-Art, §102/§103 & FTO Intelligence")
        st.divider()

        st.subheader("1. LLM Configuration")
        provider_keys = list(LLM_PROVIDERS.keys())
        st.selectbox(
            "LLM Provider *",
            options=provider_keys,
            format_func=lambda k: LLM_PROVIDERS[k].label,
            key="provider_key",
            on_change=_on_provider_change,
        )
        spec = LLM_PROVIDERS[st.session_state["provider_key"]]
        st.caption(spec.help_text)

        if st.session_state["provider_key"] == "ollama":
            reachable, installed_models, ollama_msg = list_ollama_models(st.session_state["base_url"])
            if reachable and installed_models:
                if st.session_state["model_name"] not in installed_models:
                    st.session_state["model_name"] = installed_models[0]
                st.caption(f"✅ {ollama_msg} Installed: `{', '.join(installed_models)}`")
            elif reachable:
                st.warning(
                    "Ollama is running, but **0 local models** are downloaded yet. "
                    "Run `ollama pull llama3.2:1b` (1.3 GB) or `ollama pull llama3.2` in PowerShell, "
                    "or use **Google Gemini / OpenAI / Groq / Offline Demo LLM**."
                )
            else:
                st.warning(
                    f"⚠️ {ollama_msg} Start Ollama or switch **LLM Provider** above to an API provider or **Offline Demo LLM**."
                )

        st.text_input(
            "Model Name *",
            key="model_name",
            help=f"Suggested: {', '.join(spec.suggested_models)}",
        )

        if spec.requires_api_key:
            env_present = bool(spec.env_var and os.getenv(spec.env_var, "").strip())
            key_label = f"API Key ({spec.env_var}) {'(loaded from .env)' if env_present else '*'}"
            st.text_input(
                key_label,
                key="api_key",
                type="password",
                placeholder="Paste your API key...",
            )

        if spec.supports_base_url:
            st.text_input(
                "Base URL",
                key="base_url",
                placeholder="http://localhost:11434",
            )

        if st.button("Test LLM Connection", width="stretch"):
            cfg_err = _validate_llm_config()
            if cfg_err:
                st.session_state["llm_status"] = (False, cfg_err)
            else:
                with st.spinner("Testing LLM connection via LCEL..."):
                    ok, msg = verify_llm_connection(
                        provider=st.session_state["provider_key"],
                        model_name=st.session_state["model_name"],
                        api_key=st.session_state["api_key"],
                        base_url=st.session_state["base_url"],
                    )
                    st.session_state["llm_status"] = (ok, msg)

        if st.session_state["llm_status"] is not None:
            ok, msg = st.session_state["llm_status"]
            if ok:
                st.success(msg)
            else:
                st.error(msg)

        st.divider()
        st.subheader("2. Prompt Engineering & RAG")
        st.selectbox(
            "Prompting Strategy *",
            options=list(PROMPT_TECHNIQUES.keys()),
            format_func=lambda k: f"{k} — {PROMPT_TECHNIQUES[k].split('—')[0].strip()}",
            key="prompt_technique",
        )
        st.caption(PROMPT_TECHNIQUES[st.session_state["prompt_technique"]])

        st.multiselect(
            "Prior-Art, Patent & Product Sources",
            options=[
                "EuropePMC Patents",
                "Google Patents",
                "arXiv",
                "OpenAlex",
                "Crossref",
                "Online Shopping Products",
                "PatentsView",
            ],
            key="selected_sources",
        )
        if "PatentsView" in st.session_state["selected_sources"]:
            st.text_input(
                "PatentsView API Key (Optional)",
                key="patentsview_key",
                type="password",
                help="Required only if querying PatentsView Search v1.",
            )

        st.toggle(
            "⚡ Fast Screening Mode (Parallel Execution)",
            key="fast_mode",
            help="Runs external API searches and Comparison/Gap agents concurrently + uses 10-min query cache to minimize waiting time.",
        )

        st.slider("ChromaDB Top-K Retrieval", min_value=2, max_value=12, key="top_k")
        st.slider("Max Results per Source", min_value=2, max_value=8, key="max_per_source")

        st.divider()
        st.subheader("3. ChromaDB Vector Store")
        col_a, col_b = st.columns(2)
        col_a.metric("Indexed Docs", store.count())
        col_b.metric("Pinned Memo", len(st.session_state.get("pinned_memo_items", [])))
        st.caption(f"Embeddings: `{store.embedding_backend_name}`")

        if st.button("Reset ChromaDB Index", width="stretch"):
            store.reset()
            st.session_state["chroma_search_results"] = None
            st.success("ChromaDB collection cleared.")
            st.rerun()


def _render_agent_flowchart(result: ScreeningResult) -> None:
    """Render the glowing Antigravity Agentic Pipeline Flowchart."""
    agents_ran = {t.agent for t in result.traces}
    has_router = True
    has_search = any("Search" in a or "RAG" in a for a in agents_ran)
    has_comp = result.comparison is not None or any("Comparison" in a for a in agents_ran)
    has_gap = result.gap_analysis is not None or any("Gap" in a for a in agents_ran)
    has_rep = bool(result.report_markdown) or any("Report" in a for a in agents_ran)

    c_router = "pq-flow-active-cyan" if has_router else "pq-flow-standby"
    c_search = "pq-flow-active-cyan" if has_search else "pq-flow-standby"
    c_comp = "pq-flow-active-green" if has_comp else "pq-flow-standby"
    c_gap = "pq-flow-active-purple" if has_gap else "pq-flow-standby"
    c_rep = "pq-flow-active-cyan" if has_rep else "pq-flow-standby"

    st.markdown(
        f"""
        <div class="pq-flow-card">
            <div class="pq-box-title">🕸️ Agentic Pipeline Flowchart</div>
            <div class="pq-box-body" style="font-size:0.82rem;color:#94A3B8;">
                Intent: <code>{html.escape(result.intent_result.intent)}</code> ·
                Pipeline: <code>{html.escape(result.intent_result.routed_pipeline)}</code>
            </div>
            <div class="pq-flow-grid">
                <div class="pq-flow-node {c_router}">🧭 LCEL Router<br><span style="font-size:0.7rem;opacity:0.8;">Conf {result.intent_result.confidence:.2f}</span></div>
                <div class="pq-flow-arrow">➔</div>
                <div class="pq-flow-node {c_search}">🔎 Search Agent<br><span style="font-size:0.7rem;opacity:0.8;">{len(result.newly_indexed_docs)} indexed</span></div>
                <div class="pq-flow-arrow">➔</div>
                <div class="pq-flow-node {c_comp}">⚖️ NoveltyComparisonAgent<br><span style="font-size:0.7rem;opacity:0.8;">§102 / §103 / FTO</span></div>
                <div class="pq-flow-arrow">➔</div>
                <div class="pq-flow-node {c_gap}">💡 Gap Agent<br><span style="font-size:0.7rem;opacity:0.8;">White Space</span></div>
                <div class="pq-flow-arrow">➔</div>
                <div class="pq-flow-node {c_rep}">📄 Report Agent<br><span style="font-size:0.7rem;opacity:0.8;">Memo Draft</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_patentability_matrix(result: ScreeningResult) -> None:
    """Render the Patentability & FTO Heatmap Matrix matching the Antigravity visual paradigm."""
    overlap = (result.comparison.overall_overlap_risk if result.comparison else "N/A").lower()
    fto = (result.comparison.fto_risk_level if result.comparison else "N/A").lower()
    has_gaps = bool(result.gap_analysis and result.gap_analysis.novel_gaps)
    top_gap = (result.gap_analysis.novel_gaps[0].novelty_signal.lower() if has_gaps else "n/a")

    def _cell(active: bool, tone: str, txt: str) -> str:
        cls = f"pq-mcell-{tone}" if active else "pq-mcell-muted"
        return f'<td class="pq-mcell {cls}">{txt}</td>'

    st.markdown(
        f"""
        <div class="pq-matrix-wrap">
            <div class="pq-box-title">🧬 Patentability &amp; FTO Matrices</div>
            <table class="pq-matrix-table">
                <thead>
                    <tr>
                        <th></th>
                        <th>§102 Novelty</th>
                        <th>§103 Obviousness</th>
                        <th>FTO Clearance</th>
                        <th>White-Space</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="pq-matrix-row-lbl">High Risk</td>
                        {_cell(overlap == "high", "red", "HIGH" if overlap == "high" else "—")}
                        {_cell(overlap == "high", "amber", "COMB" if overlap == "high" else "—")}
                        {_cell(fto == "high", "red", "BLOCK" if fto == "high" else "—")}
                        {_cell(top_gap == "weak", "red", "NARROW" if top_gap == "weak" else "—")}
                    </tr>
                    <tr>
                        <td class="pq-matrix-row-lbl">Moderate</td>
                        {_cell(overlap == "moderate", "amber", "PARTIAL" if overlap == "moderate" else "—")}
                        {_cell(overlap in {"moderate", "low"} and result.comparison is not None, "amber", "CHECK" if result.comparison else "—")}
                        {_cell(fto == "moderate", "amber", "DESIGN" if fto == "moderate" else "—")}
                        {_cell(top_gap == "moderate", "amber", "MOD" if top_gap == "moderate" else "—")}
                    </tr>
                    <tr>
                        <td class="pq-matrix-row-lbl">Clear / Strong</td>
                        {_cell(overlap == "low", "green", "NOVEL" if overlap == "low" else "—")}
                        {_cell(overlap == "low", "green", "NON-OBV" if overlap == "low" else "—")}
                        {_cell(fto == "low", "green", "CLEAR" if fto == "low" else "—")}
                        {_cell(top_gap == "strong", "green", "STRONG" if top_gap == "strong" else "—")}
                    </tr>
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_result_boxes(result: ScreeningResult) -> None:
    """Render richly partitioned Antigravity floating neo-glass cards & modular pinning dashboard."""
    st.markdown("---")
    elapsed = st.session_state.get("last_elapsed_s")
    timing_txt = f" • Completed in **{elapsed:.2f}s**" if elapsed is not None else ""
    st.markdown(f"### 📊 Screening Intelligence Dashboard")
    st.caption(f"MODE: `{'Fast Parallel screening' if st.session_state.get('fast_mode') else 'Deep Multi-Query'}`{timing_txt}")

    for warn in result.warnings:
        st.warning(warn)

    overlap_label = result.comparison.overall_overlap_risk if result.comparison else "N/A"
    fto_label = result.comparison.fto_risk_level if result.comparison else "N/A"

    # Floating KPI Strip
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f'<div class="pq-kpi-card">'
            f'<div class="pq-kpi-label">Detected Intent</div>'
            f'<div class="pq-kpi-value">{html.escape(result.intent_result.intent)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f'<div class="pq-kpi-card" style="border-top-color:#34D399;">'
            f'<div class="pq-kpi-label">Indexed / Retrieved</div>'
            f'<div class="pq-kpi-value">{len(result.newly_indexed_docs)} new docs · {len(result.retrieved_docs)} hits</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with k3:
        badge_o = _badge_html(overlap_label, _risk_variant(overlap_label)) if overlap_label != "N/A" else "N/A"
        st.markdown(
            f'<div class="pq-kpi-card" style="border-top-color:#FBBF24;">'
            f'<div class="pq-kpi-label">Patentability Overlap Risk</div>'
            f'<div class="pq-kpi-value">{badge_o}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with k4:
        badge_f = _badge_html(fto_label, _risk_variant(fto_label)) if fto_label != "N/A" else "N/A"
        st.markdown(
            f'<div class="pq-kpi-card" style="border-top-color:#A78BFA;">'
            f'<div class="pq-kpi-label">Registered Patent FTO Risk</div>'
            f'<div class="pq-kpi-value">{badge_f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.write("")

    # Split Visual Dashboard: Flowchart & Heatmap on Left | Intent Routing & Antigravity Product Showcase on Right
    left_dash, right_dash = st.columns([1.15, 1.0])
    with left_dash:
        _render_agent_flowchart(result)
        _render_patentability_matrix(result)

    with right_dash:
        st.markdown(
            f'<div class="pq-box pq-box-accent-blue">'
            f'<div class="pq-box-title">🧭 LCEL Intent Detection & Agentic Routing</div>'
            f'<p class="pq-box-body"><strong>Routed Pipeline:</strong> <code>{html.escape(result.intent_result.routed_pipeline)}</code> '
            f'&nbsp;|&nbsp; <strong>Confidence:</strong> <code>{result.intent_result.confidence:.2f}</code><br>'
            f'{html.escape(result.intent_result.rationale)}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Highlight top commercial products in the upper-right Antigravity showcase
        top_products = [
            d for d in (list(result.newly_indexed_docs) + list(result.retrieved_docs)) if d.doc_type == "product"
        ]
        if top_products:
            featured = top_products[0]
            ag_title = _rebrand_antigravity_product(featured.title)
            img_tag = (
                f'<img src="{html.escape(featured.image_url)}" alt="{html.escape(ag_title)}" />'
                if featured.image_url
                else '<span style="color:#64748B;font-size:0.82rem;">Antigravity Blueprint Preview</span>'
            )
            link_tag = (
                f'<a class="pq-prod-link" href="{html.escape(featured.url)}" target="_blank">View Product ↗</a>'
                if featured.url
                else ""
            )
            st.markdown(
                f'<div class="pq-prod-card">'
                f'<div class="pq-doc-header">{_badge_html("ANTIGRAVITY FEATURED", "patent")} '
                f'{_badge_html(featured.source_api, "product")} '
                f'{_badge_html(featured.price or "Listed Online", "moderate")}</div>'
                f'<div class="pq-prod-img-wrap">{img_tag}</div>'
                f'<div class="pq-prod-codename">{html.escape(ag_title)}</div>'
                f'<div class="pq-prod-orig">Commercial Match: {html.escape(featured.title[:88])}</div>'
                f'{link_tag}'
                f'</div>',
                unsafe_allow_html=True,
            )

    if result.qa_answer:
        st.markdown(
            f'<div class="pq-box pq-box-accent-indigo">'
            f'<div class="pq-box-title">💬 Agentic RAG Follow-Up Answer</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(result.qa_answer)

    # Box 2: §102 / §103 Patentability Partition Boxes
    if result.comparison:
        st.markdown(
            f'<div class="pq-box pq-box-accent-indigo">'
            f'<div class="pq-box-title">⚖️ Patentability Synthesis</div>'
            f'<p class="pq-box-body">{html.escape(result.comparison.summary)}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        c102, c103 = st.columns(2)
        with c102:
            txt_102 = result.comparison.novelty_102_assessment or "Single-reference novelty screening complete."
            st.markdown(
                f'<div class="pq-box pq-box-accent-emerald">'
                f'<div class="pq-box-title">📘 35 U.S.C. §102 — Single-Reference Novelty</div>'
                f'<p class="pq-box-body">{html.escape(txt_102)}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with c103:
            txt_103 = result.comparison.obviousness_103_assessment or "Multi-reference obviousness screening complete."
            st.markdown(
                f'<div class="pq-box pq-box-accent-amber">'
                f'<div class="pq-box-title">📙 35 U.S.C. §103 — Non-Obviousness Combination</div>'
                f'<p class="pq-box-body">{html.escape(txt_103)}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if result.comparison.overlap_items:
            with st.expander("📋 Feature-by-Feature Prior-Art Overlap Matrix (Pin Claims to Build Memo)", expanded=True):
                for idx, item in enumerate(result.comparison.overlap_items):
                    b_html = _badge_html(f"{item.overlap_level} Overlap", _risk_variant(item.overlap_level))
                    card_col, pin_col = st.columns([5, 1])
                    with card_col:
                        st.markdown(
                            f'<div class="pq-doc-card">'
                            f'<div class="pq-doc-header">{b_html} '
                            f'<strong>{html.escape(item.feature)}</strong> '
                            f'<span class="pq-doc-meta">vs. [{html.escape(item.prior_art_id)}]</span></div>'
                            f'<div class="pq-doc-abs">{html.escape(item.explanation)}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    with pin_col:
                        _render_pin_button(
                            item_id=f"ov_{idx}_{item.prior_art_id}",
                            category="Feature Overlap Findings (§102/§103)",
                            title=f"{item.feature} ({item.overlap_level} vs. {item.prior_art_id})",
                            detail=item.explanation,
                            key_suffix=str(idx),
                        )

        # Box 3: Freedom-to-Operate (FTO) Registered Patent Screening
        with st.expander("🛡️ Freedom-to-Operate (FTO) Registered-Patent Clearance Matrix", expanded=True):
            if result.comparison.fto_items:
                for idx, fto in enumerate(result.comparison.fto_items):
                    f_badge = _badge_html(f"{fto.fto_risk} FTO Risk", _risk_variant(fto.fto_risk))
                    card_col, pin_col = st.columns([5, 1])
                    with card_col:
                        st.markdown(
                            f'<div class="pq-doc-card">'
                            f'<div class="pq-doc-header">{f_badge} '
                            f'{_badge_html(fto.patent_id, "patent")} '
                            f'<strong>{html.escape(fto.title or fto.patent_id)}</strong></div>'
                            f'<div class="pq-doc-meta">Assignee: {html.escape(fto.assignee)} · '
                            f'Status/Jurisdiction: {html.escape(fto.status_or_jurisdiction)}</div>'
                            f'<div class="pq-doc-abs"><strong>Blocking Claim Focus:</strong> {html.escape(fto.blocking_claim_focus)}<br>'
                            f'<strong>Design-Around / Clearance Path:</strong> {html.escape(fto.design_around_note)}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    with pin_col:
                        _render_pin_button(
                            item_id=f"fto_{fto.patent_id}",
                            category="Freedom-to-Operate (FTO) Clearances",
                            title=f"[{fto.patent_id}] {fto.title or fto.patent_id} ({fto.fto_risk} Risk)",
                            detail=f"Blocking Claim Focus: {fto.blocking_claim_focus}\nDesign-Around: {fto.design_around_note}",
                            key_suffix=str(idx),
                        )
            else:
                st.info(
                    "No registered `PAT-*` patents matched directly in the top retrieved ChromaDB slice; "
                    "add a PatentsView API key or index specific patent excerpts in Tab 2 for claim-level FTO checks."
                )

    # Box 4: Novelty Gaps & Patentable White Space
    if result.gap_analysis:
        st.markdown(
            f'<div class="pq-box pq-box-accent-emerald">'
            f'<div class="pq-box-title">💡 Novelty Gaps &amp; Patentable White-Space Opportunities</div>'
            f'<p class="pq-box-body">{html.escape(result.gap_analysis.summary)}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        g_cols = st.columns(max(1, min(2, len(result.gap_analysis.novel_gaps))))
        for idx, gap in enumerate(result.gap_analysis.novel_gaps):
            col = g_cols[idx % len(g_cols)]
            sig_badge = _badge_html(f"{gap.novelty_signal} Signal", _risk_variant(gap.novelty_signal))
            with col:
                st.markdown(
                    f'<div class="pq-doc-card">'
                    f'<div class="pq-doc-header">{sig_badge} <strong>{html.escape(gap.aspect)}</strong></div>'
                    f'<div class="pq-doc-abs"><strong>Evidence Basis:</strong> {html.escape(gap.evidence_basis)}<br>'
                    f'<strong>Suggested Claim / Validation:</strong> {html.escape(gap.suggested_claim_or_experiment)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                _render_pin_button(
                    item_id=f"gap_{idx}_{gap.aspect[:20]}",
                    category="Novelty Gaps & Suggested Claims",
                    title=f"{gap.aspect} ({gap.novelty_signal} Signal)",
                    detail=f"Evidence: {gap.evidence_basis}\nSuggested Claim: {gap.suggested_claim_or_experiment}",
                    key_suffix=str(idx),
                )

    # Box 5: Commercial Products Found Online (Antigravity Reimagined Gallery + Pictures & Links)
    product_map = {}
    for d in list(result.newly_indexed_docs) + list(result.retrieved_docs):
        if d.doc_type == "product" and d.doc_id not in product_map:
            product_map[d.doc_id] = d
    product_docs = list(product_map.values())

    if product_docs:
        with st.expander(
            f"🛒 Antigravity Commercial Product Matrix — Pictures & Links ({len(product_docs)})",
            expanded=True,
        ):
            p_cols = st.columns(min(3, len(product_docs)))
            for idx, prod in enumerate(product_docs):
                col = p_cols[idx % len(p_cols)]
                with col:
                    ag_name = _rebrand_antigravity_product(prod.title)
                    img_html = (
                        f'<img src="{html.escape(prod.image_url)}" alt="{html.escape(ag_name)}" />'
                        if prod.image_url
                        else '<span style="color:#64748B;font-size:0.82rem;">No preview image available</span>'
                    )
                    price_badge = _badge_html(prod.price or "Listed Online", "moderate")
                    plat_badge = _badge_html(prod.source_api, "product")
                    link_btn = (
                        f'<a class="pq-prod-link" href="{html.escape(prod.url)}" target="_blank">View Product ↗</a>'
                        if prod.url
                        else ""
                    )
                    st.markdown(
                        f'<div class="pq-prod-card">'
                        f'<div class="pq-prod-img-wrap">{img_html}</div>'
                        f'<div class="pq-doc-header">{plat_badge} {price_badge}</div>'
                        f'<div class="pq-prod-codename">{html.escape(ag_name)}</div>'
                        f'<div class="pq-prod-orig">{html.escape(prod.title[:90])}</div>'
                        f'{link_btn}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    _render_pin_button(
                        item_id=f"prod_{prod.doc_id}",
                        category="Commercial §102(a)(1) Product References",
                        title=f"[{prod.doc_id}] {ag_name} ({prod.price or 'Listed Online'})",
                        detail=f"Source: {prod.source_api} | Original Listing: {prod.title}\nURL: {prod.url}",
                        key_suffix=f"p{idx}",
                    )

    # Box 6: Direct Patent Portal Links for Executed Queries
    if result.search_queries_used:
        links_html = []
        for q in result.search_queries_used[:3]:
            gp_url = f"https://patents.google.com/?q={urllib.parse.quote_plus(q)}"
            esp_url = f"https://worldwide.espacenet.com/patent/search?q={urllib.parse.quote_plus(q)}"
            links_html.append(
                f'<li><code>{html.escape(q)}</code> — '
                f'<a href="{gp_url}" target="_blank" style="color:#38BDF8;">Google Patents ↗</a> · '
                f'<a href="{esp_url}" target="_blank" style="color:#38BDF8;">Espacenet ↗</a></li>'
            )
        st.markdown(
            f'<div class="pq-box pq-box-accent-blue">'
            f'<div class="pq-box-title">🔗 Live Patent Portal Search Links</div>'
            f'<ul style="margin:0.2rem 0 0 1.2rem;font-size:0.88rem;color:#CBD5E1;">{"".join(links_html)}</ul>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Box 7: Retrieved Prior-Art Cards
    if result.retrieved_docs:
        with st.expander(f"📚 Top Retrieved Prior Art from ChromaDB ({len(result.retrieved_docs)})", expanded=True):
            for idx, d in enumerate(result.retrieved_docs):
                t_var = d.doc_type if d.doc_type in {"patent", "paper", "product", "custom"} else "custom"
                sim_txt = f"Sim {d.similarity_score:.2f}" if d.similarity_score is not None else "Indexed"
                link_html = (
                    f'<a href="{html.escape(d.url)}" target="_blank">Open Source ↗</a>'
                    if d.url
                    else "No URL"
                )
                card_col, pin_col = st.columns([5, 1])
                with card_col:
                    st.markdown(
                        f'<div class="pq-doc-card">'
                        f'<div class="pq-doc-header">{_badge_html(d.doc_type.upper(), t_var)} '
                        f'{_badge_html(d.doc_id, "custom")} '
                        f'{_badge_html(sim_txt, "low")}</div>'
                        f'<div class="pq-doc-title">{html.escape(d.title)}</div>'
                        f'<div class="pq-doc-meta">{html.escape(d.authors_or_assignee)} · {html.escape(d.year)} · '
                        f'{html.escape(d.source_api)} · {link_html}</div>'
                        f'<div class="pq-doc-abs">{html.escape(d.abstract[:360])}{"..." if len(d.abstract) > 360 else ""}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with pin_col:
                    _render_pin_button(
                        item_id=f"doc_{d.doc_id}",
                        category="Pinned Prior-Art Citations",
                        title=f"[{d.doc_id}] {d.title} ({d.year})",
                        detail=f"Source: {d.source_api} | {d.url}\n{d.abstract[:350]}",
                        key_suffix=f"d{idx}",
                    )


def _render_screening_tab(store: ChromaPriorArtStore) -> None:
    with st.container(border=True):
        st.subheader("🔬 Invention Disclosure & Screening Workspace")
        st.caption(
            "Fields marked with **`*`** are required. Choose a competition preset, select **Custom / Blank Participant Entry**, "
            "or save your own ideas as reusable presets."
        )

        all_presets = _get_all_presets()
        preset_options = [CUSTOM_ENTRY_LABEL] + list(all_presets.keys())
        if st.session_state["sample_choice"] not in preset_options:
            st.session_state["sample_choice"] = preset_options[1] if len(preset_options) > 1 else CUSTOM_ENTRY_LABEL

        preset_col, btn_load_col, btn_save_col, btn_clear_col = st.columns([3, 1, 1, 1])
        with preset_col:
            st.selectbox(
                "Sample & Participant Presets",
                options=preset_options,
                key="sample_choice",
                label_visibility="collapsed",
            )
        with btn_load_col:
            st.button("Load Sample", on_click=_on_load_sample, width="stretch")
        with btn_save_col:
            st.button("💾 Save Preset", on_click=_on_save_preset, width="stretch")
        with btn_clear_col:
            st.button("Clear Form", on_click=_on_clear_form, width="stretch")

        if st.session_state.get("preset_feedback"):
            kind, msg = st.session_state["preset_feedback"]
            if kind == "ok":
                st.success(msg)
            else:
                st.warning(msg)
            st.session_state["preset_feedback"] = None

        c1, c2 = st.columns([2, 1])
        with c1:
            st.text_input(
                "Invention Title *",
                key="idea_title",
                placeholder="e.g., Solar-Powered Cold-Chain Vaccine Box with PCM Thermal Buffer",
            )
        with c2:
            st.text_input(
                "Technical Domain (Optional)",
                key="idea_domain",
                placeholder="e.g., Biomedical Devices / Energy",
            )

        st.text_area(
            "Problem Statement *",
            key="idea_problem",
            height=88,
            placeholder="Describe the technical limitation or gap in existing systems...",
        )
        st.text_area(
            "Proposed Technical Solution / Mechanism *",
            key="idea_solution",
            height=105,
            placeholder="Explain how your invention works, its architecture, materials, or algorithm...",
        )

        c3, c4 = st.columns(2)
        with c3:
            st.text_area(
                "Key Components / Draft Claim Elements (Optional)",
                key="idea_components",
                height=90,
                placeholder="1. Component A; 2. Control algorithm B; 3. Packaging C...",
            )
        with c4:
            st.text_input(
                "Technical Keywords (Optional)",
                key="idea_keywords",
                placeholder="comma-separated keywords for search query expansion",
            )
            st.selectbox(
                "Intent Routing Mode",
                options=[
                    "Auto-Detect",
                    "Full Screening Report",
                    "Prior-Art Search Only",
                    "Novelty Comparison",
                    "Gap Analysis",
                ],
                key="mode_override",
                help="Use Auto-Detect to let the LCEL Intent Router classify the task automatically.",
            )

        with st.expander("📌 Attach / Upload a Specific Patent or Reference to Check Against (Optional)", expanded=False):
            st.caption(
                "Participants can paste any patent number & claim/abstract—or upload a `.txt` / `.md` / `.json` file—to "
                "pin it directly into ChromaDB before running screening."
            )
            with st.form("quick_pin_patent_form", clear_on_submit=True):
                qp1, qp2, qp3 = st.columns([1.4, 1, 1])
                q_pat_id = qp1.text_input("Patent / Reference ID *", placeholder="e.g., PAT-US11223344")
                q_pat_type = qp2.selectbox("Type *", options=["patent", "paper", "product", "custom"])
                q_pat_year = qp3.text_input("Year (Optional)", value="2024")
                q_pat_title = st.text_input("Patent / Paper Title *", placeholder="Title of the patent or prior work")
                q_pat_url = st.text_input("URL (Optional)", placeholder="https://patents.google.com/patent/...")
                q_pat_text = st.text_area(
                    "Claims / Abstract Text (or upload a file below)",
                    height=80,
                    placeholder="Paste abstract or independent claim text...",
                )
                q_upload = st.file_uploader(
                    "Or Upload Reference File (.txt, .md, .json)",
                    type=["txt", "md", "json"],
                )
                pin_submit = st.form_submit_button("Pin Reference to ChromaDB")

                if pin_submit:
                    body_txt = q_pat_text.strip()
                    if q_upload is not None and not body_txt:
                        try:
                            body_txt = q_upload.read().decode("utf-8", errors="ignore").strip()
                        except Exception:
                            body_txt = ""
                    if not q_pat_id.strip() or not q_pat_title.strip() or len(body_txt) < 10:
                        st.warning("Provide **Patent / Reference ID***, **Title***, and at least 10 characters of text (or upload a file).")
                    else:
                        clean_pid = q_pat_id.strip().replace(" ", "-")
                        pinned_doc = PriorArtDocument(
                            doc_id=clean_pid,
                            doc_type=q_pat_type,  # type: ignore[arg-type]
                            title=q_pat_title.strip(),
                            abstract=body_txt[:1500],
                            authors_or_assignee="Participant Pinned Reference",
                            year=q_pat_year.strip() or "N/A",
                            url=q_pat_url.strip(),
                            source_api="Participant Input",
                        )
                        store.upsert_documents([pinned_doc])
                        st.success(f"Pinned `{clean_pid}` into ChromaDB ({store.count()} total indexed docs)!")

        b1, b2, b3, b4 = st.columns(4)
        run_full = b1.button("🚀 Run Screening Pipeline", type="primary", width="stretch")
        run_search = b2.button("🔎 Search Prior Art Only", width="stretch")
        run_compare = b3.button("⚖️ Compare Novelty", width="stretch")
        run_gaps = b4.button("💡 Find Novelty Gaps", width="stretch")

    if run_full:
        idea, errs = _validate_idea_fields(require_full=True)
        if errs:
            for e in errs:
                st.warning(e)
        else:
            mode = st.session_state["mode_override"]
            req = (
                f"Run a full prior-art, §102/§103 patentability, and FTO screening report for: {idea.title}"
                if mode == "Auto-Detect"
                else f"{mode} for invention: {idea.title}"
            )
            _execute_pipeline(store, req, idea, mode)

    elif run_search:
        idea, errs = _validate_idea_fields(require_full=False)
        if errs:
            for e in errs:
                st.warning(e)
        else:
            _execute_pipeline(store, f"Search patents and papers for: {idea.title}", idea, "Prior-Art Search Only")

    elif run_compare:
        idea, errs = _validate_idea_fields(require_full=True)
        if errs:
            for e in errs:
                st.warning(e)
        else:
            _execute_pipeline(store, f"Compare novelty overlap and FTO for: {idea.title}", idea, "Novelty Comparison")

    elif run_gaps:
        idea, errs = _validate_idea_fields(require_full=True)
        if errs:
            for e in errs:
                st.warning(e)
        else:
            _execute_pipeline(store, f"Analyze novelty gaps for: {idea.title}", idea, "Gap Analysis")

    st.write("")
    with st.container(border=True):
        st.subheader("💬 RAG Agent Follow-up Assistant")
        st.caption(
            "Ask about retrieved patents/papers, §102/§103 risks, FTO design-arounds, or trigger a narrower search via LCEL routing."
        )
        q_col, q_btn_col = st.columns([4, 1])
        with q_col:
            st.text_input(
                "Ask a question or enter a routing prompt *",
                key="followup_query",
                placeholder="e.g., Which retrieved patent poses the highest FTO risk for our controller?",
                label_visibility="collapsed",
            )
        with q_btn_col:
            send_followup = st.button("Send Request", width="stretch")

    if send_followup:
        req_text = st.session_state["followup_query"].strip()
        if len(req_text) < 4:
            st.warning("Please enter a question or search prompt (at least 4 characters) before clicking **Send Request**.")
        else:
            idea_opt, _ = _validate_idea_fields(require_full=False)
            _execute_pipeline(store, req_text, idea_opt, "Auto-Detect")

    result: Optional[ScreeningResult] = st.session_state["last_result"]
    if result:
        _render_result_boxes(result)


def _render_knowledge_base_tab(store: ChromaPriorArtStore) -> None:
    st.subheader("🗄️ ChromaDB Prior-Art Knowledge Base")
    st.caption(
        f"Collection: `{store.collection_name}` | Persist Directory: `{store.persist_dir}` | "
        f"Embeddings: `{store.embedding_backend_name}`"
    )

    if st.session_state["kb_feedback"]:
        st.success(st.session_state["kb_feedback"])
        st.session_state["kb_feedback"] = None

    with st.container(border=True):
        st.markdown("### 🔍 Semantic Similarity Search (ChromaDB)")
        s1, s2, s3 = st.columns([3, 1, 1])
        with s1:
            kb_query = st.text_input(
                "Semantic Search Query *",
                key="kb_search_query",
                placeholder="Enter technical terms to test ChromaDB similarity retrieval...",
            )
        with s2:
            kb_filter = st.selectbox("Filter Type", options=["All", "patent", "paper", "product", "custom"], key="kb_filter")
        with s3:
            kb_topk = st.number_input("Top-K", min_value=1, max_value=20, value=5, key="kb_topk")

        if st.button("Search ChromaDB", key="btn_kb_search"):
            if not kb_query.strip():
                st.warning("Please enter a search query to test ChromaDB similarity search.")
            else:
                filt = None if kb_filter == "All" else kb_filter
                hits = _cached_chroma_search(store, kb_query.strip(), int(kb_topk), filt)
                st.session_state["chroma_search_results"] = hits

        if st.session_state["chroma_search_results"] is not None:
            hits: List[PriorArtDocument] = st.session_state["chroma_search_results"]
            if not hits:
                st.info("No matching records found in ChromaDB.")
            else:
                for h in hits:
                    score_txt = f"{h.similarity_score:.3f}" if h.similarity_score is not None else "N/A"
                    with st.expander(f"[{h.doc_id}] ({h.doc_type.upper()}, Sim={score_txt}) {h.title}"):
                        st.write(f"**Authors/Assignee:** {h.authors_or_assignee} | **Year:** {h.year} | **Source:** {h.source_api}")
                        if h.url:
                            st.write(f"**URL:** {h.url}")
                        st.write(h.abstract)

    with st.expander("➕ Add Custom Patent or Paper Excerpt to ChromaDB", expanded=False):
        with st.form("custom_doc_form", clear_on_submit=True):
            fc1, fc2, fc3 = st.columns([1.5, 1, 1])
            c_id = fc1.text_input("Document ID *", placeholder="e.g., PAT-US11445566 or CUSTOM-01")
            c_type = fc2.selectbox("Type *", options=["patent", "paper", "custom"])
            c_year = fc3.text_input("Year (Optional)", value="2025")
            c_title = st.text_input("Title *", placeholder="Title of the patent or research paper")
            c_authors = st.text_input("Authors / Assignee (Optional)", placeholder="e.g., MIT / Jane Doe")
            c_url = st.text_input("Reference URL (Optional)", placeholder="https://patents.google.com/patent/...")
            c_abs = st.text_area("Abstract / Claim Excerpt *", height=100, placeholder="Paste key claim or abstract text...")
            submitted = st.form_submit_button("Index Excerpt in ChromaDB")

            if submitted:
                errs = []
                if not c_id.strip():
                    errs.append("**Document ID*** is required.")
                if not c_title.strip():
                    errs.append("**Title*** is required.")
                if len(c_abs.strip()) < 10:
                    errs.append("**Abstract / Claim Excerpt*** is required (minimum 10 characters).")
                if errs:
                    for e in errs:
                        st.warning(e)
                else:
                    clean_id = c_id.strip().replace(" ", "-")
                    doc = PriorArtDocument(
                        doc_id=clean_id,
                        doc_type=c_type,  # type: ignore[arg-type]
                        title=c_title.strip(),
                        abstract=c_abs.strip(),
                        authors_or_assignee=c_authors.strip() or "User Added",
                        year=c_year.strip() or "N/A",
                        url=c_url.strip(),
                        source_api="Manual Entry",
                    )
                    store.upsert_documents([doc])
                    _cached_chroma_search.clear()
                    st.session_state["kb_feedback"] = f"Indexed `{clean_id}` into ChromaDB."
                    st.rerun()

    with st.container(border=True):
        st.markdown("### 📚 All Indexed Records in ChromaDB")
        all_docs = store.list_all_documents(limit=200)
        if not all_docs:
            st.info("ChromaDB is currently empty. Run a search on Tab 1 or add a custom excerpt above.")
        else:
            df = pd.DataFrame(
                [
                    {
                        "ID": d.doc_id,
                        "Type": d.doc_type,
                        "Year": d.year,
                        "Source": d.source_api,
                        "Authors/Assignee": d.authors_or_assignee,
                        "Title": d.title,
                        "URL": d.url,
                    }
                    for d in all_docs
                ]
            )
            st.dataframe(df, width="stretch", hide_index=True)

            del_col, btn_del_col, exp_col = st.columns([2, 1, 1])
            with del_col:
                selected_del_id = st.selectbox(
                    "Select Record to Delete",
                    options=[d.doc_id for d in all_docs],
                    label_visibility="collapsed",
                )
            with btn_del_col:
                if st.button("Delete Record", width="stretch"):
                    store.delete_document(selected_del_id)
                    _cached_chroma_search.clear()
                    st.session_state["kb_feedback"] = f"Deleted `{selected_del_id}` from ChromaDB."
                    st.rerun()
            with exp_col:
                export_json = json.dumps([d.model_dump() for d in all_docs], indent=2)
                st.download_button(
                    "Export Index (.json)",
                    data=export_json,
                    file_name="patentq_chromadb_index.json",
                    mime="application/json",
                    width="stretch",
                )


def _render_trace_and_prompts_tab() -> None:
    st.subheader("🧠 Agent Trace, Prompt Engineering & LCEL Inspector")
    st.caption(
        "Transparently inspect Intent Routing, Agentic RAG decisions, System/User prompts (CoT, Few-Shot, ReAct, Reflection), "
        "and LCEL chain composition."
    )

    result: Optional[ScreeningResult] = st.session_state["last_result"]
    with st.container(border=True):
        st.markdown("### 1. Latest Execution Trace (Intent Router + Agentic RAG + 4-Agent Team)")
        if result:
            st.success(
                f"**Detected Intent:** `{result.intent_result.intent}` | "
                f"**Routed LCEL Branch:** `{result.intent_result.routed_pipeline}`"
            )
            for t in result.traces:
                with st.expander(f"Step {t.step}: {t.agent} — {t.action} [{t.prompt_technique}]", expanded=True):
                    st.write(f"**Decision / Output:** {t.decision_or_output}")
                    if t.retrieved_doc_ids:
                        st.write(f"**Referenced IDs:** `{', '.join(t.retrieved_doc_ids)}`")
                    if t.system_prompt_preview:
                        st.code(t.system_prompt_preview, language="text")
                    if t.user_prompt_preview:
                        st.code(t.user_prompt_preview, language="text")
        else:
            st.info("Run any screening or follow-up query in Tab 1 to view the live agent execution trace here.")

    with st.container(border=True):
        st.markdown("### 2. Active Prompt Engineering Strategy & System/User Prompts")
        active_tech = st.session_state["prompt_technique"]
        st.info(f"**Active Strategy:** `{active_tech}` — {PROMPT_TECHNIQUES[active_tech]}")
        st.code(TECHNIQUE_DIRECTIVES[active_tech], language="text")

        catalog = get_prompt_catalog(active_tech)
        for agent_label, prompts in catalog.items():
            with st.expander(f"Prompt Template: {agent_label}"):
                st.markdown("**System Prompt:**")
                st.code(prompts["system"], language="text")
                st.markdown("**User Prompt Template:**")
                st.code(prompts["user"], language="text")

    with st.container(border=True):
        st.markdown("### 3. LangChain LCEL & Tool Registry")
        st.code(
            "# Intent Detection & LCEL RunnableBranch Routing (src/router.py)\n"
            "intent_chain = build_intent_prompt() | llm | StrOutputParser() | RunnableLambda(parse_intent)\n"
            "router_branch = RunnableBranch(\n"
            "    (lambda x: x['intent_result'].intent == 'prior_art_search', search_handler),\n"
            "    (lambda x: x['intent_result'].intent == 'novelty_comparison', comparison_handler),\n"
            "    (lambda x: x['intent_result'].intent == 'gap_analysis', gap_handler),\n"
            "    (lambda x: x['intent_result'].intent == 'followup_qa', qa_handler),\n"
            "    full_screening_handler,\n"
            ")\n\n"
            "# Agent LCEL Chains (src/agents/*.py)\n"
            "search_query_chain = build_search_query_prompt() | llm | StrOutputParser()\n"
            "agentic_rag_chain  = build_agentic_rag_prompt('ReAct') | llm | StrOutputParser()\n"
            "comparison_chain   = build_comparison_prompt(technique) | llm | StrOutputParser()\n"
            "gap_chain          = build_gap_prompt(technique) | llm | StrOutputParser()\n"
            "report_chain       = build_report_prompt(technique) | llm | StrOutputParser()",
            language="python",
        )
        tools = get_langchain_tools()
        st.markdown("**Registered LangChain Prior-Art Tools:**")
        for tl in tools:
            st.write(f"- `{tl.name}`: {tl.description}")


def _render_report_tab(store: ChromaPriorArtStore) -> None:
    st.subheader("📄 Modular & Formal Patentability & FTO Memorandum")
    st.markdown(f'<div class="pq-memo-banner">{html.escape(DISCLAIMER_TEXT)}</div>', unsafe_allow_html=True)

    result: Optional[ScreeningResult] = st.session_state["last_result"]
    pinned: List[Dict[str, str]] = st.session_state.get("pinned_memo_items", [])

    # Modular Pinned Memorandum Builder section
    with st.container(border=True):
        st.markdown(f"### 📌 Modular Memorandum Builder ({len(pinned)} Pinned Elements)")
        if not pinned:
            st.caption(
                "Click **📌 Pin to Memo** on any overlap card, FTO risk item, novelty gap, product, or prior-art record "
                "in **Screening Workspace** to dynamically assemble a custom executive brief here."
            )
        else:
            custom_md = _build_custom_pinned_markdown(result)
            pm1, pm2 = st.columns([1, 1])
            with pm1:
                st.download_button(
                    "⬇️ Export Custom Pinned Memo (.md)",
                    data=custom_md,
                    file_name=f"PatentQ_Custom_Pinned_Memo_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown",
                    width="stretch",
                )
            with pm2:
                if st.button("🗑️ Clear Pinned Items", width="stretch"):
                    st.session_state["pinned_memo_items"] = []
                    st.rerun()
            with st.expander("Preview Custom Assembled Memorandum", expanded=True):
                st.markdown(custom_md)

    st.write("")
    if not result:
        st.info("No screening results yet. Go to **Screening Workspace** and click **Run Screening Pipeline**.")
        return

    if not result.report_markdown:
        st.info(
            "Your latest action ran a partial route (`"
            f"{result.intent_result.intent}`) without drafting the full Patentability & FTO Memorandum yet."
        )
        if st.button("Draft Full Report Now from Current Idea & ChromaDB", type="primary"):
            idea, errs = _validate_idea_fields(require_full=True)
            if errs:
                for e in errs:
                    st.warning(e)
            else:
                _execute_pipeline(
                    store,
                    f"Draft full §102/§103 patentability and FTO screening report for: {idea.title}",
                    idea,
                    "Full Screening Report",
                )
                st.rerun()
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            "⬇️ Download Note (.md)",
            data=result.report_markdown,
            file_name=f"PatentQ_Screening_Memorandum_{timestamp}.md",
            mime="text/markdown",
            width="stretch",
        )
    with dl2:
        st.download_button(
            "⬇️ Download Note (.txt)",
            data=result.report_markdown,
            file_name=f"PatentQ_Screening_Memorandum_{timestamp}.txt",
            mime="text/plain",
            width="stretch",
        )
    with dl3:
        st.download_button(
            "⬇️ Download Full Bundle (.json)",
            data=result.model_dump_json(indent=2),
            file_name=f"PatentQ_Screening_Bundle_{timestamp}.json",
            mime="application/json",
            width="stretch",
        )

    st.write("")
    with st.container(border=True):
        st.markdown(result.report_markdown)


def _render_hero_header(store: ChromaPriorArtStore) -> None:
    prov_label = LLM_PROVIDERS[st.session_state["provider_key"]].label
    model_name = st.session_state["model_name"]
    strategy = st.session_state["prompt_technique"]
    doc_count = store.count()
    pinned_cnt = len(st.session_state.get("pinned_memo_items", []))
    st.markdown(
        f"""
        <div class="pq-hero">
            <div class="pq-hero-title">⚖️ {html.escape(APP_NAME)} — {html.escape(APP_SUBTITLE)}</div>
            <div class="pq-hero-sub">{html.escape(DISCLAIMER_TEXT)}</div>
            <div class="pq-pill-row">
                <span class="pq-pill">🤖 LLM: {html.escape(prov_label)} ({html.escape(model_name)})</span>
                <span class="pq-pill">🧠 Strategy: {html.escape(strategy)}</span>
                <span class="pq-pill">🗄️ ChromaDB Indexed: {doc_count}</span>
                <span class="pq-pill">📌 Pinned Memo Items: {pinned_cnt}</span>
                <span class="pq-pill">🛡️ §102 / §103 &amp; FTO Registered-Patent Screening</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    _inject_custom_css()
    _init_session_state()
    store = get_vector_store()
    _render_sidebar(store)
    _render_hero_header(store)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔬 Screening Workspace",
            "🗄️ ChromaDB Knowledge Base",
            "🧠 Agent Trace & Prompts",
            "📄 Patentability & FTO Memorandum",
        ]
    )

    with tab1:
        _render_screening_tab(store)
    with tab2:
        _render_knowledge_base_tab(store)
    with tab3:
        _render_trace_and_prompts_tab()
    with tab4:
        _render_report_tab(store)


if __name__ == "__main__":
    main()
