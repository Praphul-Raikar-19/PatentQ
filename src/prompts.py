"""LangChain prompt templates and prompt-engineering strategies for PatentQ.

Explicitly separates SYSTEM and USER prompts and supports:
- Chain-of-Thought (CoT)
- Few-Shot Prompting (via FewShotChatMessagePromptTemplate)
- ReAct (Reason + Act + Observation)
- Self-Reflection (Draft + Evidence Critique)
"""

from __future__ import annotations

from typing import Dict, List

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

# ---------------------------------------------------------------------------
# 1. Technique-Specific Prompt Engineering Directives
# ---------------------------------------------------------------------------

TECHNIQUE_DIRECTIVES: Dict[str, str] = {
    "CoT": (
        "PROMPTING STRATEGY — CHAIN OF THOUGHT (CoT):\n"
        "Reason step by step before producing your final answer:\n"
        "Step 1: Decompose the invention into atomic technical features (inputs, mechanism, architecture, outputs).\n"
        "Step 2: Map each feature against the retrieved prior-art excerpts using explicit citation tags [DOC_ID].\n"
        "Step 3: Weigh exact overlaps vs. partial analogies vs. missing elements.\n"
        "Step 4: Synthesize the final structured conclusion strictly grounded in the retrieved excerpts."
    ),
    "Few-Shot": (
        "PROMPTING STRATEGY — FEW-SHOT CALIBRATION:\n"
        "Follow the exact analytical depth, citation discipline ([DOC_ID]), and calibrated overlap "
        "classifications demonstrated in the reference examples provided in this prompt."
    ),
    "ReAct": (
        "PROMPTING STRATEGY — ReAct (REASON + ACT):\n"
        "Structure your internal trace explicitly using:\n"
        "Thought: What technical question or evidence gap still needs verification?\n"
        "Action: Choose either RETRIEVE_CHROMA(query), SEARCH_EXTERNAL(query), or SYNTHESIZE.\n"
        "Observation: Summarize what the retrieved citations actually state (or fail to cover).\n"
        "Repeat until evidence is sufficient, then produce the final grounded JSON/report."
    ),
    "Reflection": (
        "PROMPTING STRATEGY — SELF-REFLECTION:\n"
        "1. Draft an initial assessment of overlap/novelty from the retrieved context.\n"
        "2. Critique your draft: Did you overclaim novelty? Did every claim cite a specific [DOC_ID]? "
        "Are any key mechanisms unverified in the retrieved snippets?\n"
        "3. Produce a revised, stricter final assessment reflecting the critique."
    ),
}

# Few-shot exemplars for Novelty Comparison calibration
COMPARISON_FEW_SHOT_EXAMPLES: List[Dict[str, str]] = [
    {
        "example_input": (
            "Invention: Wearable ultrasonic patch for continuous blood pressure monitoring using flexible piezoelectric arrays.\n"
            "Prior Art [PAT-US10987001]: Rigid wristwatch photoplethysmography (PPG) sensor estimating blood pressure."
        ),
        "example_output": (
            '{"overall_overlap_risk": "Moderate", '
            '"summary": "Prior art addresses non-invasive blood pressure estimation [PAT-US10987001], '
            'but relies on optical PPG in a rigid wrist form factor rather than conformal ultrasonic arrays.", '
            '"overlap_items": [{"feature": "Continuous cuffless blood pressure estimation", "prior_art_id": "PAT-US10987001", '
            '"prior_art_title": "Wearable PPG BP Monitor", "overlap_level": "High", '
            '"explanation": "Both target continuous cuffless blood pressure tracking [PAT-US10987001]."}, '
            '{"feature": "Flexible piezoelectric ultrasonic arterial-wall transducer", "prior_art_id": "PAT-US10987001", '
            '"prior_art_title": "Wearable PPG BP Monitor", "overlap_level": "None", '
            '"explanation": "Reference uses optical PPG LEDs and does not disclose ultrasonic arterial diameter sensing."}], '
            '"closest_references": ["PAT-US10987001"], '
            '"reasoning_trace": "Optical PPG overlaps on clinical objective, while acoustic mechanism remains distinct."}'
        ),
    }
]

# ---------------------------------------------------------------------------
# 2. Explicitly Separated SYSTEM & USER Prompts for Each Agent
# ---------------------------------------------------------------------------

INTENT_SYSTEM_PROMPT = """You are the Intent Detection & Routing Controller for PatentQ, a prior-art and patentability screening assistant.
Classify the user's request into EXACTLY ONE intent:
- "prior_art_search": Searching or collecting related patents and academic papers only.
- "novelty_comparison": Comparing an invention idea against existing/retrieved prior art to assess overlap.
- "gap_analysis": Identifying novel aspects, white spaces, or under-explored directions.
- "full_screening_report": Running a complete screening pipeline (search + comparison + gap analysis + cited preliminary note).
- "followup_qa": Asking a specific follow-up question about already retrieved prior art or the report.

Return ONLY valid JSON with keys:
{{"intent": "<one of the 5 intents>", "confidence": <0.0-1.0>, "rationale": "<1-2 sentences>", "extracted_query": "<concise search query>"}}"""

INTENT_USER_PROMPT = """User Request / Action Context:
{user_request}

Has Structured Invention Idea: {has_idea}
Indexed Prior-Art Count in ChromaDB: {indexed_count}
User Requested Mode Override: {mode_override}"""


SEARCH_SYSTEM_PROMPT = """You are the Search Agent for PatentQ.
Your job is to translate a technical invention disclosure into 3 diverse, high-recall search queries suitable for patent databases (Google Patents / PatentsView / USPTO) and research paper APIs (arXiv / OpenAlex / Crossref).
Query 1: Core technical mechanism + functional keywords.
Query 2: Broader domain / problem-oriented terminology and synonyms.
Query 3: Specific hardware/algorithmic component combination.

Return ONLY valid JSON:
{{"queries": ["query 1", "query 2", "query 3"], "rationale": "brief explanation"}}"""

SEARCH_USER_PROMPT = """Invention Title: {title}
Domain: {domain}
Problem Statement: {problem}
Proposed Solution: {solution}
Key Components: {components}
Keywords: {keywords}"""


AGENTIC_RAG_SYSTEM_PROMPT = """You are the Agentic RAG Decision Controller for PatentQ.
Inspect the user's query/invention and the preview of chunks currently retrieved from ChromaDB.
Decide whether the retrieved ChromaDB evidence is sufficient to answer accurately with citations, OR whether external search tools should be invoked first to gather more patents/papers into ChromaDB.

{technique_directive}

Return ONLY valid JSON:
{{"decision": "USE_RETRIEVED" or "CALL_SEARCH_TOOL", "reason": "<why>", "search_query": "<refined query if CALL_SEARCH_TOOL else empty>"}}"""

AGENTIC_RAG_USER_PROMPT = """Task / Query: {task_description}
ChromaDB Collection Size: {collection_size}
Top Retrieved Chunks Preview:
{retrieved_preview}"""


COMPARISON_SYSTEM_PROMPT = """You are the Novelty Comparison Agent for PatentQ.
Compare the user's invention against ONLY the retrieved prior-art excerpts provided below.
Be objective, precise, and never invent patent numbers or paper IDs not present in the context.
Cite references inline using their exact bracketed ID, e.g., [ARXIV-2401.0001] or [PAT-US11223344].

{technique_directive}

Return ONLY valid JSON matching this schema:
{{
  "overall_overlap_risk": "High" | "Moderate" | "Low",
  "fto_risk_level": "High" | "Moderate" | "Low",
  "summary": "<2-4 sentence grounded synthesis>",
  "novelty_102_assessment": "<35 U.S.C. 102 single-reference anticipation screening citing [DOC_ID]>",
  "obviousness_103_assessment": "<35 U.S.C. 103 obviousness / combination risk screening citing [DOC_ID]>",
  "overlap_items": [
    {{
      "feature": "<invention feature or claim element>",
      "prior_art_id": "<exact doc_id from context>",
      "prior_art_title": "<title>",
      "overlap_level": "High" | "Moderate" | "Low" | "None",
      "explanation": "<grounded comparison citing [DOC_ID]>"
    }}
  ],
  "fto_items": [
    {{
      "patent_id": "<patent doc_id from context or Closest Reference>",
      "title": "<patent title>",
      "assignee": "<assignee or owner>",
      "status_or_jurisdiction": "<e.g., Published/Granted Patent — Active Status Check>",
      "fto_risk": "High" | "Moderate" | "Low",
      "blocking_claim_focus": "<feature or claim scope with potential FTO overlap>",
      "design_around_note": "<concrete technical differentiation or design-around path>"
    }}
  ],
  "closest_references": ["<doc_id_1>", "<doc_id_2>"],
  "reasoning_trace": "<concise summary of your reasoning steps>"
}}"""

COMPARISON_USER_PROMPT = """INVENTION TO SCREEN:
Title: {title}
Domain: {domain}
Problem: {problem}
Solution: {solution}
Components: {components}

RETRIEVED PRIOR-ART CONTEXT (FROM CHROMADB):
{context}"""


GAP_SYSTEM_PROMPT = """You are the Novelty Gap Agent for PatentQ.
Your role is to identify technical aspects of the invention that appear novel, differentiated, or under-explored relative to the retrieved prior art and the §102/§103/FTO assessment.
Also evaluate whether current evidence is sufficient or if additional targeted searches are warranted.

{technique_directive}

Return ONLY valid JSON matching this schema:
{{
  "summary": "<2-3 sentence summary of potential novelty white space>",
  "novel_gaps": [
    {{
      "aspect": "<distinctive technical element or combination>",
      "novelty_signal": "Strong" | "Moderate" | "Exploratory",
      "evidence_basis": "<why retrieved docs [DOC_ID] do not fully cover this>",
      "suggested_claim_or_experiment": "<concrete next technical validation or claim focus>"
    }}
  ],
  "under_explored_combinations": ["<combination 1>", "<combination 2>"],
  "evidence_sufficiency": "Sufficient" | "Needs_More_Search",
  "followup_search_queries": ["<optional extra query if needed>"],
  "reasoning_trace": "<concise reasoning or self-reflection note>"
}}"""

GAP_USER_PROMPT = """INVENTION:
Title: {title}
Solution: {solution}
Components: {components}

NOVELTY COMPARISON SUMMARY:
{comparison_summary}

RETRIEVED PRIOR-ART CONTEXT:
{context}"""


REPORT_SYSTEM_PROMPT = """You are the Report Agent for PatentQ.
Draft a formal, structured **Patentability & Freedom-to-Operate (FTO) Screening Memorandum** in Markdown.
Every factual statement about prior work or registered patents MUST cite the provided reference tags [DOC_ID].
Do NOT fabricate patent numbers or references. Include a concise Screening Scope Notice at the top and bottom.

{technique_directive}

Required Markdown Sections:
1. **Formal Patentability & FTO Screening Memorandum** (Title + Executive Risk Summary + Screening Scope Notice)
2. **Invention Disclosure & Technical Claim Scope**
3. **Prior-Art Search & Registered Patent Coverage**
4. **35 U.S.C. §102 Novelty & §103 Non-Obviousness Assessment** (include a structured Markdown overlap table)
5. **Freedom-to-Operate (FTO) Risk Matrix (Registered Patents)** (include a Markdown FTO table with patent IDs, risk level, blocking focus, and design-around path)
6. **Novelty Gaps, Claim Strategy & Recommended Next Steps**
7. **Cited References** (bulleted list with `[DOC_ID]`, title, type, year, authors/assignee, and URL)"""

REPORT_USER_PROMPT = """INVENTION DISCLOSURE:
Title: {title}
Domain: {domain}
Problem: {problem}
Solution: {solution}
Components: {components}

SEARCH QUERIES EXECUTED:
{search_queries}

COMPARISON ANALYSIS:
{comparison_json}

GAP ANALYSIS:
{gap_json}

RETRIEVED REFERENCES:
{references_block}"""


QA_SYSTEM_PROMPT = """You are the PatentQ Agentic RAG Research Assistant.
Answer the user's follow-up question strictly using the retrieved ChromaDB prior-art excerpts and current screening context.
Always cite specific references using [DOC_ID]. If the retrieved context lacks the answer, explicitly state what is missing and suggest a targeted search query.

{technique_directive}"""

QA_USER_PROMPT = """USER QUESTION:
{question}

ACTIVE INVENTION TITLE: {title}

RETRIEVED CHROMADB EXCERPTS:
{context}

LATEST SCREENING SUMMARY (IF AVAILABLE):
{report_excerpt}"""


# ---------------------------------------------------------------------------
# 3. LangChain ChatPromptTemplate Builders
# ---------------------------------------------------------------------------


def build_intent_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", INTENT_SYSTEM_PROMPT), ("human", INTENT_USER_PROMPT)]
    )


def build_search_query_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", SEARCH_SYSTEM_PROMPT), ("human", SEARCH_USER_PROMPT)]
    )


def build_agentic_rag_prompt(technique: str = "ReAct") -> ChatPromptTemplate:
    directive = TECHNIQUE_DIRECTIVES.get(technique, TECHNIQUE_DIRECTIVES["ReAct"])
    return ChatPromptTemplate.from_messages(
        [
            ("system", AGENTIC_RAG_SYSTEM_PROMPT),
            ("human", AGENTIC_RAG_USER_PROMPT),
        ]
    ).partial(technique_directive=directive)


def build_comparison_prompt(technique: str = "CoT") -> ChatPromptTemplate:
    directive = TECHNIQUE_DIRECTIVES.get(technique, TECHNIQUE_DIRECTIVES["CoT"])
    messages = [("system", COMPARISON_SYSTEM_PROMPT)]
    if technique == "Few-Shot":
        example_prompt = ChatPromptTemplate.from_messages(
            [("human", "{example_input}"), ("ai", "{example_output}")]
        )
        few_shot = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=COMPARISON_FEW_SHOT_EXAMPLES,
        )
        messages.append(few_shot)
    messages.append(("human", COMPARISON_USER_PROMPT))
    return ChatPromptTemplate.from_messages(messages).partial(technique_directive=directive)


def build_gap_prompt(technique: str = "Reflection") -> ChatPromptTemplate:
    directive = TECHNIQUE_DIRECTIVES.get(technique, TECHNIQUE_DIRECTIVES["Reflection"])
    return ChatPromptTemplate.from_messages(
        [("system", GAP_SYSTEM_PROMPT), ("human", GAP_USER_PROMPT)]
    ).partial(technique_directive=directive)


def build_report_prompt(technique: str = "CoT") -> ChatPromptTemplate:
    directive = TECHNIQUE_DIRECTIVES.get(technique, TECHNIQUE_DIRECTIVES["CoT"])
    return ChatPromptTemplate.from_messages(
        [("system", REPORT_SYSTEM_PROMPT), ("human", REPORT_USER_PROMPT)]
    ).partial(technique_directive=directive)


def build_qa_prompt(technique: str = "ReAct") -> ChatPromptTemplate:
    directive = TECHNIQUE_DIRECTIVES.get(technique, TECHNIQUE_DIRECTIVES["ReAct"])
    return ChatPromptTemplate.from_messages(
        [("system", QA_SYSTEM_PROMPT), ("human", QA_USER_PROMPT)]
    ).partial(technique_directive=directive)


def get_prompt_catalog(active_technique: str = "CoT") -> Dict[str, Dict[str, str]]:
    """Return human-readable System & User prompts for display in the Prompt Inspector UI."""
    directive = TECHNIQUE_DIRECTIVES.get(active_technique, TECHNIQUE_DIRECTIVES["CoT"])
    return {
        "Intent Detection & Router": {
            "system": INTENT_SYSTEM_PROMPT,
            "user": INTENT_USER_PROMPT,
        },
        "Search Agent (Query Reformulation)": {
            "system": SEARCH_SYSTEM_PROMPT,
            "user": SEARCH_USER_PROMPT,
        },
        "Agentic RAG Decision Controller": {
            "system": AGENTIC_RAG_SYSTEM_PROMPT.format(technique_directive=directive),
            "user": AGENTIC_RAG_USER_PROMPT,
        },
        "Novelty Comparison Agent": {
            "system": COMPARISON_SYSTEM_PROMPT.format(technique_directive=directive),
            "user": COMPARISON_USER_PROMPT,
        },
        "Gap Agent": {
            "system": GAP_SYSTEM_PROMPT.format(technique_directive=directive),
            "user": GAP_USER_PROMPT,
        },
        "Report Agent": {
            "system": REPORT_SYSTEM_PROMPT.format(technique_directive=directive),
            "user": REPORT_USER_PROMPT,
        },
        "Follow-up RAG Q&A": {
            "system": QA_SYSTEM_PROMPT.format(technique_directive=directive),
            "user": QA_USER_PROMPT,
        },
    }

