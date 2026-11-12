# ⚖️ PatentQ — Prior-Art, Patentability (§102/§103) & FTO Registered-Patent Screener

**PatentQ** is a multi-agent, Agentic-RAG invention-screening, patentability (§102/§103), and Freedom-to-Operate (FTO) intelligence dashboard built with **Streamlit**, **LangChain (LCEL)**, and **ChromaDB**, designed with a **Neo-Glassmorphic Antigravity UI**.

> **Screening Scope:** PatentQ combines 35 U.S.C. §102 Single-Reference Novelty, §103 Non-Obviousness Combinations, FTO Registered-Patent Clearance Checks, and §102(a)(1) Commercial Product Screening into an interactive screening memorandum workbench. *Users should consult a registered patent attorney/agent prior to formal USPTO/EPO/WIPO filings or commercial launch.*

---

## ✨ Key Capabilities & Antigravity UI Highlights

- **Floating Neo-Glassmorphic Dashboard:** Deep-space layered glass cards, bioluminescent accents, hover micro-animations, live **Agentic Pipeline Flowchart**, and **Patentability & FTO Heatmap Matrix**.
- **Antigravity Commercial Product Matrix:** Searches live online marketplaces (Made-in-China & Snapdeal) concurrently, displays product images, pricing, clickable retail links, and elevated Antigravity system designations.
- **Modular RAG Memorandum Builder via Pinning:** Pin individual overlap findings, FTO blocking claims, novelty gaps, commercial products, or prior-art references into a custom assembled executive brief.
- **Competition & Participant-Ready:** Includes **8 technical domain presets**, **Custom / Blank Participant Entry**, **Save Preset** (`custom_presets.json`), and one-click **Pin Reference / Upload `.txt/.md/.json`** directly to ChromaDB.
- **Fast Screening Mode:** Concurrent `ThreadPoolExecutor` API fetching + parallel Comparison/Gap execution + 10-minute thread-safe search cache and `@st.cache_resource` / `@st.cache_data` caching.

---

## Multi-Agent Architecture

| Agent | Responsibility | Implementation |
|---|---|---|
| **Intent Router (LCEL)** | Detects request intent (`prior_art_search`, `novelty_comparison`, `gap_analysis`, `full_screening_report`, `followup_qa`) and routes via `RunnableBranch`. | [`src/router.py`](src/router.py) |
| **Search Agent** | Reformulates queries via LCEL, queries EuropePMC Patents (`SRC:PAT`), Google Patents, arXiv, OpenAlex, Crossref, Online Shopping Products (plus optional PatentsView), and indexes excerpts into ChromaDB. | [`src/agents/search_agent.py`](src/agents/search_agent.py), [`src/tools/search_tools.py`](src/tools/search_tools.py) |
| **Novelty Comparison Agent** | Evaluates top-k semantic matches from ChromaDB for §102 Novelty, §103 Non-Obviousness, feature overlaps, and Registered-Patent FTO risks with `[DOC_ID]` citations. | [`src/agents/comparison_agent.py`](src/agents/comparison_agent.py) |
| **Gap Agent** | Identifies novel white-space angles, under-explored combinations, and triggers secondary Agentic RAG searches when gaps need deeper evidence. | [`src/agents/gap_agent.py`](src/agents/gap_agent.py) |
| **Report Agent** | Drafts a cited **Formal Patentability & FTO Screening Memorandum** exportable as Markdown, TXT, or JSON. | [`src/agents/report_agent.py`](src/agents/report_agent.py) |

---

## Mandatory Acceptance Criteria Mapping

1. **LLM Integration (`src/llm_factory.py`)**: Supports local models via **Ollama** (`ChatOllama`) and API models via **Google Gemini** (`ChatGoogleGenerativeAI`), **OpenAI / OpenAI-compatible** (`ChatOpenAI`), **Groq** (`ChatGroq`), plus an **Offline Demo LLM** (`PatentQDemoChatModel`) for zero-config evaluation. Auto-detects downloaded Ollama models via `/api/tags`.
2. **Interactive GUI (`app.py`)**: Neo-glassmorphic Streamlit interface branded **PatentQ** with 4 tabs (*Screening Workspace*, *ChromaDB Knowledge Base*, *Agent Trace & Prompts*, *Patentability & FTO Memorandum*), visual flowchart, risk matrix heatmap, and modular pinning.
3. **Prompt Engineering (`src/prompts.py`)**: Explicitly separates `SYSTEM` and `USER` prompts for every agent and supports **Chain-of-Thought (CoT)**, **Few-Shot** (`FewShotChatMessagePromptTemplate`), **ReAct**, and **Self-Reflection**, all inspectable in the *Agent Trace & Prompts* tab.
4. **LangChain Core Components (`src/prompts.py`, `src/router.py`, `src/agents/*.py`)**: Uses `ChatPromptTemplate`, `FewShotChatMessagePromptTemplate`, `StrOutputParser`, `RunnableLambda`, `RunnableBranch`, and `@tool` composed via **LCEL** (`prompt | llm | parser`).
5. **Agentic RAG (`src/agents/orchestrator.py`)**: Inspects ChromaDB retrieval sufficiency before comparison/Q&A and decides whether to `USE_RETRIEVED` or `CALL_SEARCH_TOOL`, with secondary evidence loops when the Gap Agent flags `Needs_More_Search`.
6. **Intent Detection and Routing (`src/router.py`)**: Classifies requests with an LCEL intent chain and dispatches via `RunnableBranch`.
7. **Vector Database (`src/vector_store.py`)**: Persistent **ChromaDB** (`chromadb.PersistentClient`) storing patent, paper, and product metadata + embeddings (`all-MiniLM-L6-v2` ONNX local embeddings) with cosine similarity search, type filtering, manual ingestion, deletion, and JSON export.

---

## Quickstart

### 1. Clone & Create Virtual Environment
```bash
git clone <your-repo-url>
cd patentQ
python -m venv .venv
```

Activate the virtual environment:
- **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure an LLM (Local or API)
Copy `.env.example` to `.env` (optional—you can also enter credentials directly in the Streamlit sidebar):

- **Option A — Local LLM (Ollama, no API key required):**
  ```bash
  ollama pull llama3.2:1b   # or llama3.2
  ```
- **Option B — API LLM (Gemini / OpenAI / Groq):**
  Set `GOOGLE_API_KEY`, `OPENAI_API_KEY`, or `GROQ_API_KEY` in `.env` or paste into the sidebar.
- **Option C — Offline Demo Mode:**
  Select **Offline Demo LLM** in the sidebar (automatically selected if no Ollama model is pulled or API key configured).

### 4. Run PatentQ
```bash
streamlit run app.py
```

### 5. Run Tests
```bash
pytest -v
```

---

## Project Structure

```text
patentQ/
├── app.py                      # Antigravity Neo-Glassmorphic Streamlit frontend (PatentQ)
├── .streamlit/config.toml        # Dark space theme configuration
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Pytest configuration
├── .env.example                # Environment variable template
├── .gitignore
├── README.md
├── src/
│   ├── config.py               # Provider catalog, 8 sample presets, custom preset persistence
│   ├── models.py               # Pydantic schemas (§102/§103, FTO, commercial products)
│   ├── prompts.py              # System/User prompt templates (CoT, Few-Shot, ReAct, Reflection)
│   ├── llm_factory.py          # LangChain ChatModel factory, Ollama tag check & error guidance
│   ├── vector_store.py         # Persistent ChromaDB wrapper & LCEL retriever
│   ├── router.py               # Intent detection & LCEL RunnableBranch routing
│   ├── tools/
│   │   └── search_tools.py     # Concurrent EuropePMC Patents, Google Patents, arXiv, OpenAlex, Crossref & Shopping tools
│   └── agents/
│       ├── search_agent.py
│       ├── comparison_agent.py
│       ├── gap_agent.py
│       ├── report_agent.py
│       ├── orchestrator.py     # Agentic RAG & parallel 4-agent pipeline coordinator
│       └── utils.py
└── tests/
    └── test_patentq.py         # Unit, integration, and Streamlit AppTest suite
```
