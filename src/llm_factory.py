"""LangChain ChatModel factory, connection tester, and resilient JSON parser for PatentQ."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.config import LLM_PROVIDERS


class PatentQDemoChatModel(BaseChatModel):
    """Offline LangChain BaseChatModel for keyless UI evaluation and testing.

    Generates grounded responses using only the retrieved [DOC_ID]s and invention text
    supplied in the prompt messages, allowing all LCEL chains to execute end-to-end.
    """

    model_name: str = "patentq-deterministic-simulator"

    @property
    def _llm_type(self) -> str:
        return "patentq-demo-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        sys_text = ""
        user_text = ""
        for msg in messages:
            if msg.type == "system":
                sys_text += str(msg.content) + "\n"
            elif msg.type == "human":
                user_text += str(msg.content) + "\n"

        content = self._synthesize_response(sys_text, user_text)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def _extract_doc_ids(self, text: str) -> List[str]:
        found = re.findall(r"\[((?:PAT|ARXIV|OALEX|DOI|CUSTOM)-[A-Za-z0-9._-]+)\]", text)
        return list(dict.fromkeys(found))

    def _extract_field(self, text: str, prefix: str, default: str = "") -> str:
        m = re.search(rf"^{re.escape(prefix)}\s*(.+)$", text, flags=re.MULTILINE)
        return m.group(1).strip() if m else default

    def _synthesize_response(self, sys_text: str, user_text: str) -> str:
        doc_ids = self._extract_doc_ids(user_text)
        title = self._extract_field(user_text, "Invention Title:", "") or self._extract_field(
            user_text, "Title:", "Proposed Invention"
        )

        # 1. Intent Detection
        if "Intent Detection & Routing Controller" in sys_text:
            req = user_text.lower()
            if "only search" in req or "find prior art" in req or "search patents" in req:
                intent = "prior_art_search"
            elif "compare" in req or "overlap" in req:
                intent = "novelty_comparison"
            elif "gap" in req or "novel" in req or "white space" in req:
                intent = "gap_analysis"
            elif "question:" in req or "follow-up" in req or "explain" in req or "?" in req:
                intent = "followup_qa"
            else:
                intent = "full_screening_report"
            return json.dumps(
                {
                    "intent": intent,
                    "confidence": 0.91,
                    "rationale": f"Classified request into '{intent}' based on task structure and query cues.",
                    "extracted_query": title[:100] if title != "Proposed Invention" else user_text[:100].strip(),
                }
            )

        # 2. Search Query Reformulation
        if "Search Agent for PatentQ" in sys_text:
            keywords = self._extract_field(user_text, "Keywords:", "")
            domain = self._extract_field(user_text, "Domain:", "")
            base = " ".join(title.split()[:6]) or "novel technical system"
            q1 = keywords.split(",")[0].strip() + " " + base if keywords else base
            q2 = f"{base} {domain}".strip()
            q3 = " ".join(keywords.split(",")[:3]).strip() or f"{base} architecture method"
            return json.dumps(
                {
                    "queries": [q1.strip(), q2.strip(), q3.strip()],
                    "rationale": "Generated mechanism-focused, domain-level, and component-combination queries.",
                }
            )

        # 3. Agentic RAG Decision Controller
        if "Agentic RAG Decision Controller" in sys_text:
            size_match = re.search(r"ChromaDB Collection Size:\s*(\d+)", user_text)
            col_size = int(size_match.group(1)) if size_match else 0
            if col_size < 3 or len(doc_ids) < 2:
                return json.dumps(
                    {
                        "decision": "CALL_SEARCH_TOOL",
                        "reason": "ChromaDB lacks sufficient relevant prior-art coverage; invoking search tools first.",
                        "search_query": title if title != "Proposed Invention" else "prior art technology",
                    }
                )
            return json.dumps(
                {
                    "decision": "USE_RETRIEVED",
                    "reason": f"ChromaDB returned {len(doc_ids)} relevant citations; proceeding with grounded RAG synthesis.",
                    "search_query": "",
                }
            )

        # 4. Novelty Comparison Agent
        if "Novelty Comparison Agent" in sys_text:
            top_ids = doc_ids[:3] if doc_ids else ["NO-RETRIEVED-DOCS"]
            patent_ids = [d for d in doc_ids if d.startswith("PAT-")][:3]
            components_raw = self._extract_field(user_text, "Components:", "Core mechanism; Control logic; System integration")
            features = [c.strip(" 1234567890.") for c in re.split(r"[;\n]", components_raw) if c.strip()][:3]
            if not features:
                features = ["Core technical architecture", "Control/processing mechanism", "Application integration"]
            items = []
            levels = ["High", "Moderate", "Low"]
            for idx, feat in enumerate(features):
                ref_id = top_ids[idx % len(top_ids)]
                items.append(
                    {
                        "feature": feat[:120],
                        "prior_art_id": ref_id,
                        "prior_art_title": f"Retrieved Reference {ref_id}",
                        "overlap_level": levels[idx % len(levels)],
                        "explanation": (
                            f"Retrieved prior work [{ref_id}] addresses adjacent problem elements, "
                            f"showing {levels[idx % len(levels)].lower()} functional overlap with '{feat[:60]}'."
                        ),
                    }
                )
            fto_items = [
                {
                    "patent_id": pid,
                    "title": f"Registered/Published Patent {pid}",
                    "assignee": "Recorded Patent Assignee",
                    "status_or_jurisdiction": "Published / Registered Patent — Verify Active Claims",
                    "fto_risk": "Moderate",
                    "blocking_claim_focus": "Sub-system hardware or baseline sensing/control method.",
                    "design_around_note": "Differentiate via integrated closed-loop control and specific novel mechanism parameters.",
                }
                for pid in patent_ids
            ]
            cites_str = ", ".join(f"[{d}]" for d in top_ids)
            return json.dumps(
                {
                    "overall_overlap_risk": "Moderate" if doc_ids else "Low",
                    "fto_risk_level": "Moderate" if patent_ids else "Low",
                    "summary": (
                        f"Screened '{title}' against {len(doc_ids)} retrieved ChromaDB references "
                        f"({cites_str}). Adjacent domain concepts exist, while the integrated system-level "
                        "architecture retains potential patentable differentiation."
                    ),
                    "novelty_102_assessment": (
                        f"35 U.S.C. §102 Screening: No single retrieved reference ({cites_str}) identically "
                        "discloses every combined element of the disclosure, though individual sub-components overlap."
                    ),
                    "obviousness_103_assessment": (
                        f"35 U.S.C. §103 Screening: Moderate combination risk if an examiner combines baseline "
                        f"mechanisms from {cites_str}; emphasize unexpected synergy and quantitative gains."
                    ),
                    "overlap_items": items,
                    "fto_items": fto_items,
                    "closest_references": top_ids,
                    "reasoning_trace": (
                        "Step 1: Decomposed invention into claim-style features. Step 2: Screened §102 single-reference "
                        "anticipation and §103 combination overlap. Step 3: Mapped FTO exposure on retrieved registered patents."
                    ),
                }
            )

        # 5. Gap Agent
        if "Novelty Gap Agent" in sys_text:
            ref_str = ", ".join(f"[{d}]" for d in doc_ids[:3]) or "[INDEX-EMPTY]"
            return json.dumps(
                {
                    "summary": (
                        f"Relative to retrieved prior art ({ref_str}), potential white space exists in the "
                        "tight coupling of sensing/control logic and specific domain constraints."
                    ),
                    "novel_gaps": [
                        {
                            "aspect": "Integrated multi-stage architecture & adaptive control loop",
                            "novelty_signal": "Strong",
                            "evidence_basis": f"Retrieved references {ref_str} address isolated sub-problems without disclosing this exact closed-loop coupling.",
                            "suggested_claim_or_experiment": "Document quantitative ablation benchmarks and draft dependent claims on the control threshold parameters.",
                        },
                        {
                            "aspect": "Resource-constrained edge operation / packaging",
                            "novelty_signal": "Moderate",
                            "evidence_basis": f"Cited works {ref_str} focus primarily on conventional laboratory or cloud setups.",
                            "suggested_claim_or_experiment": "Measure power, latency, or thermal hold-over margins under real operating conditions.",
                        },
                    ],
                    "under_explored_combinations": [
                        "Coupling predictive edge control with domain-specific physical buffering",
                        "Cross-layer validation against real-world failure modes",
                    ],
                    "evidence_sufficiency": "Sufficient" if len(doc_ids) >= 2 else "Needs_More_Search",
                    "followup_search_queries": [] if len(doc_ids) >= 2 else [f"{title} prior art patent"],
                    "reasoning_trace": "Self-Reflection check: Confirmed all gaps reference retrieved [DOC_ID] tags without legal patentability claims.",
                }
            )

        # 6. Report Agent
        if "Report Agent for PatentQ" in sys_text:
            problem = self._extract_field(user_text, "Problem:", "Problem statement provided in disclosure.")
            solution = self._extract_field(user_text, "Solution:", "Proposed technical solution.")
            ref_tags = ", ".join(f"[{d}]" for d in doc_ids[:6]) or "None retrieved"
            patent_ids = [d for d in doc_ids if d.startswith("PAT-")]
            fto_row = (
                f"| [{patent_ids[0]}] | Moderate | Baseline hardware/method claims | Differentiate via integrated adaptive control |"
                if patent_ids
                else "| No active PAT-* hits | Low | No direct blocking patent identified in retrieved set | Verify full USPTO/EPO family before commercialization |"
            )
            return (
                f"# Formal Patentability & FTO Screening Memorandum\n\n"
                f"> **Screening Scope Notice:** Generated by **PatentQ** (§102/§103 Novelty + Registered-Patent FTO Screening). "
                f"Review independent patent claims with a registered patent attorney/agent prior to commercial filing.\n\n"
                f"## 1. Executive Patentability & FTO Summary\n"
                f"- **Invention Title:** {title}\n"
                f"- **§102 / §103 Patentability Signal:** Moderate-to-Strong (no single reference identically anticipates the integrated system)\n"
                f"- **Freedom-to-Operate (FTO) Risk:** {'Moderate' if patent_ids else 'Low'}\n\n"
                f"## 2. Invention Disclosure & Claim Scope\n"
                f"- **Problem Addressed:** {problem}\n"
                f"- **Proposed Technical Mechanism:** {solution}\n\n"
                f"## 3. Prior-Art Search & Registered Patent Coverage\n"
                f"Indexed and retrieved **{len(doc_ids)}** prior-art references via ChromaDB semantic similarity search: {ref_tags}.\n\n"
                f"## 4. 35 U.S.C. §102 Novelty & §103 Non-Obviousness Assessment\n"
                f"| Feature | Closest Reference | Overlap Level | §102 / §103 Screening Note |\n"
                f"|---|---|---|---|\n"
                f"| Core Problem Motivation | {f'[{doc_ids[0]}]' if doc_ids else 'N/A'} | High | Related domain problem addressed in prior art. |\n"
                f"| Integrated Architecture & Control | {f'[{doc_ids[1]}]' if len(doc_ids) > 1 else (f'[{doc_ids[0]}]' if doc_ids else 'N/A')} | Moderate | Partial functional overlap; combined mechanism shows §102 novelty margin. |\n\n"
                f"## 5. Freedom-to-Operate (FTO) Risk Matrix (Registered Patents)\n"
                f"| Patent ID | FTO Risk | Blocking Claim Focus | Design-Around / Clearance Path |\n"
                f"|---|---|---|---|\n"
                f"{fto_row}\n\n"
                f"## 6. Novelty Gaps, Claim Strategy & Recommended Next Steps\n"
                f"- **Independent Claim Anchor:** Focus independent claims on the coupled closed-loop control + architectural synergy across {ref_tags}.\n"
                f"- **Validation & Filing Prep:** Benchmark quantitative performance deltas and run a live LLM (Ollama, Gemini, OpenAI, or Groq) for full claim-chart drafting.\n"
            )

        # 7. Follow-up Q&A
        question = self._extract_field(user_text, "USER QUESTION:", user_text[:150])
        cites = ", ".join(f"[{d}]" for d in doc_ids[:4]) or "no indexed documents yet"
        return (
            f"Based on the retrieved ChromaDB prior-art excerpts ({cites}), here is a grounded response to "
            f"your question (*{question}*):\n\n"
            f"- The most relevant indexed items are {cites}.\n"
            f"- Review their abstracts and overlap ratings in the Knowledge Base and Report tabs, or switch to a live "
            f"LLM provider (Ollama / Gemini / OpenAI / Groq) for deeper technical synthesis."
        )


def create_chat_model(
    provider: str,
    model_name: str = "",
    api_key: str = "",
    base_url: str = "",
    temperature: float = 0.2,
) -> BaseChatModel:
    """Instantiate a LangChain BaseChatModel for the chosen provider."""
    spec = LLM_PROVIDERS.get(provider)
    if not spec:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    chosen_model = (model_name or spec.default_model).strip()
    resolved_key = (api_key or (os.getenv(spec.env_var, "") if spec.env_var else "")).strip()
    resolved_url = (base_url or spec.default_base_url).strip()

    if provider == "demo":
        return PatentQDemoChatModel()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=chosen_model,
            base_url=resolved_url or "http://localhost:11434",
            temperature=temperature,
        )

    if provider == "gemini":
        if not resolved_key:
            raise ValueError("Google Gemini requires GOOGLE_API_KEY (enter it in the sidebar or .env).")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=chosen_model,
            google_api_key=resolved_key,
            temperature=temperature,
        )

    if provider == "openai":
        if not resolved_key and not resolved_url:
            raise ValueError("OpenAI requires OPENAI_API_KEY (or a local OpenAI-compatible Base URL).")
        from langchain_openai import ChatOpenAI

        kwargs: Dict[str, Any] = {
            "model": chosen_model,
            "api_key": resolved_key or "local-key",
            "temperature": temperature,
        }
        if resolved_url:
            kwargs["base_url"] = resolved_url
        return ChatOpenAI(**kwargs)

    if provider == "groq":
        if not resolved_key:
            raise ValueError("Groq requires GROQ_API_KEY (enter it in the sidebar or .env).")
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=chosen_model,
            groq_api_key=resolved_key,
            temperature=temperature,
        )

    raise ValueError(f"Unhandled provider: {provider}")


_OLLAMA_CACHE: Dict[str, Tuple[float, Tuple[bool, List[str], str]]] = {}


def list_ollama_models(base_url: str = "http://localhost:11434") -> Tuple[bool, List[str], str]:
    """Query the local Ollama server for installed model names with 0 retries and fast cache."""
    clean_base = (base_url or "http://localhost:11434").rstrip("/")
    now = time.time()
    cached = _OLLAMA_CACHE.get(clean_base)
    if cached and (now - cached[0]) < 30.0:
        return cached[1]

    url = f"{clean_base}/api/tags"
    session = requests.Session()
    from requests.adapters import HTTPAdapter

    session.mount("http://", HTTPAdapter(max_retries=0))
    session.mount("https://", HTTPAdapter(max_retries=0))
    try:
        resp = session.get(url, timeout=0.45)
        resp.raise_for_status()
        models = [str(m.get("name", "")).strip() for m in resp.json().get("models", []) if m.get("name")]
        if models:
            res = (True, models, f"Ollama running ({len(models)} model(s) installed).")
        else:
            res = (True, [], "Ollama is running, but 0 models are installed yet.")
    except Exception as exc:
        res = (False, [], f"Ollama not reachable ({exc}).")

    _OLLAMA_CACHE[clean_base] = (now, res)
    return res


def pick_default_provider() -> Tuple[str, str, str]:
    """Auto-select a ready-to-run provider on startup so all buttons work immediately.

    Priority: (1) configured API key in environment -> (2) local Ollama if responding -> (3) Offline Demo LLM.
    Returns (provider_key, model_name, api_key).
    """
    for key in ("gemini", "openai", "groq"):
        spec = LLM_PROVIDERS[key]
        env_val = os.getenv(spec.env_var, "").strip() if spec.env_var else ""
        if env_val:
            return key, spec.default_model, env_val

    ollama_url = LLM_PROVIDERS["ollama"].default_base_url
    reachable, installed_models, _ = list_ollama_models(ollama_url)
    if reachable and installed_models:
        preferred = LLM_PROVIDERS["ollama"].default_model
        chosen = preferred if preferred in installed_models else installed_models[0]
        return "ollama", chosen, ""

    return "demo", LLM_PROVIDERS["demo"].default_model, ""


def format_llm_error(exc: Exception, provider: str, model_name: str) -> str:
    """Turn raw provider exceptions into actionable user guidance."""
    raw = str(exc)
    if provider == "ollama" and ("404" in raw or "not found" in raw.lower()):
        chosen = model_name or LLM_PROVIDERS["ollama"].default_model
        return (
            f"Ollama model '{chosen}' is not downloaded locally yet (HTTP 404). "
            f"Fix it in one of two ways:\n"
            f"1. Pull the model in PowerShell: `ollama pull {chosen}` (or `ollama pull llama3.2:1b` for a smaller 1.3 GB model), then click **Test LLM Connection**.\n"
            f"2. Or switch **LLM Provider** in the sidebar to **Google Gemini**, **OpenAI**, **Groq** (with an API key), or **Offline Demo LLM**."
        )
    if provider == "ollama" and any(k in raw.lower() for k in ["connection", "refused", "11434"]):
        return (
            "Cannot reach local Ollama at the configured Base URL. Start the Ollama app/service, "
            "or switch **LLM Provider** in the sidebar to an API provider or **Offline Demo LLM**."
        )
    return raw


def verify_llm_connection(
    provider: str,
    model_name: str = "",
    api_key: str = "",
    base_url: str = "",
) -> Tuple[bool, str]:
    """Verify that the configured LangChain LLM responds to a lightweight LCEL ping."""
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        llm = create_chat_model(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
        )
        ping_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a connectivity probe. Reply briefly."),
                ("human", "Respond with: PatentQ LLM Ready."),
            ]
        )
        chain = ping_prompt | llm | StrOutputParser()
        reply = chain.invoke({})
        preview = (reply or "").strip().replace("\n", " ")[:120]
        return True, f"Connected ({provider} / {model_name or LLM_PROVIDERS[provider].default_model}): {preview}"
    except Exception as exc:
        return False, f"Connection failed: {format_llm_error(exc, provider, model_name)}"


def extract_json_from_text(raw_text: str) -> Dict[str, Any]:
    """Parse JSON from LLM text output, stripping optional markdown code fences."""
    cleaned = (raw_text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}
