"""Configuration constants, LLM provider catalog, and sample invention presets for PatentQ."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

APP_NAME = "PatentQ"
APP_SUBTITLE = "Prior-Art, Patentability (§102/§103) & FTO Registered-Patent Screener"
DISCLAIMER_TEXT = (
    "Screening Scope Notice: PatentQ generates structured Patentability (§102 Novelty / §103 Obviousness) "
    "and Freedom-to-Operate (FTO) risk screening memoranda against registered patents and academic literature "
    "via Agentic RAG. Automated outputs support IP evaluation and filing preparation and should be reviewed "
    "with a registered patent attorney/agent before binding commercial or legal filings."
)

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db")
CHROMA_COLLECTION_NAME = "patentq_prior_art"


@dataclass(frozen=True)
class ProviderSpec:
    """Metadata and defaults for a supported LLM provider."""

    key: str
    label: str
    default_model: str
    suggested_models: List[str]
    requires_api_key: bool
    env_var: str
    supports_base_url: bool
    default_base_url: str = ""
    help_text: str = ""


LLM_PROVIDERS: Dict[str, ProviderSpec] = {
    "ollama": ProviderSpec(
        key="ollama",
        label="Ollama (Local LLM)",
        default_model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        suggested_models=["llama3.2", "llama3.1:8b", "qwen2.5:7b", "mistral", "gemma2:9b"],
        requires_api_key=False,
        env_var="",
        supports_base_url=True,
        default_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        help_text="Runs completely locally via Ollama. Start Ollama and pull a model (e.g., `ollama pull llama3.2`).",
    ),
    "gemini": ProviderSpec(
        key="gemini",
        label="Google Gemini (API)",
        default_model="gemini-2.5-flash",
        suggested_models=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        requires_api_key=True,
        env_var="GOOGLE_API_KEY",
        supports_base_url=False,
        help_text="Uses Google AI Studio / Gemini API (`GOOGLE_API_KEY`).",
    ),
    "openai": ProviderSpec(
        key="openai",
        label="OpenAI / Compatible (API)",
        default_model="gpt-4o-mini",
        suggested_models=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        requires_api_key=True,
        env_var="OPENAI_API_KEY",
        supports_base_url=True,
        default_base_url=os.getenv("OPENAI_BASE_URL", ""),
        help_text="Uses OpenAI or any OpenAI-compatible endpoint (LM Studio, vLLM, OpenRouter, Together).",
    ),
    "groq": ProviderSpec(
        key="groq",
        label="Groq (API)",
        default_model="llama-3.3-70b-versatile",
        suggested_models=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        requires_api_key=True,
        env_var="GROQ_API_KEY",
        supports_base_url=False,
        help_text="Fast hosted inference via Groq API (`GROQ_API_KEY`).",
    ),
    "demo": ProviderSpec(
        key="demo",
        label="Offline Demo LLM (No Key / Evaluation Fallback)",
        default_model="patentq-deterministic-simulator",
        suggested_models=["patentq-deterministic-simulator"],
        requires_api_key=False,
        env_var="",
        supports_base_url=False,
        help_text=(
            "Uses a LangChain Fake/Runnable LLM fallback for offline UI verification when no API key or "
            "local Ollama model is active. Switch to Ollama, Gemini, OpenAI, or Groq for live LLM reasoning."
        ),
    ),
}

PROMPT_TECHNIQUES: Dict[str, str] = {
    "CoT": "Chain-of-Thought (CoT) — Step-by-step claim decomposition and feature overlap reasoning.",
    "Few-Shot": "Few-Shot Prompting — Calibrated examples of prior-art comparison and gap framing.",
    "ReAct": "ReAct (Reason + Act) — Explicit Thought -> Action (Search/Retrieve) -> Observation loop.",
    "Reflection": "Self-Reflection — Initial draft followed by critique against retrieved evidence.",
}

INTENT_DESCRIPTIONS: Dict[str, str] = {
    "prior_art_search": "User wants to search or discover related patents and research papers only.",
    "novelty_comparison": "User wants to compare an invention idea against indexed/retrieved prior art.",
    "gap_analysis": "User wants to identify novel white-space angles, differentiators, and under-explored areas.",
    "full_screening_report": "User wants an end-to-end screening: search, comparison, gap analysis, and preliminary note.",
    "followup_qa": "User is asking a targeted question about retrieved prior art, citations, or the generated report.",
}

SAMPLE_INVENTIONS: Dict[str, Dict[str, str]] = {
    "Solar-Powered Cold-Chain Vaccine Box with Phase-Change Thermal Buffer": {
        "title": "Solar-Powered Cold-Chain Vaccine Carrier with Adaptive Phase-Change Thermal Buffer",
        "problem": (
            "Last-mile vaccine transport in off-grid rural clinics suffers from temperature excursions "
            "when battery-powered portable coolers lose charge during cloudy weather or prolonged transit."
        ),
        "solution": (
            "A portable medical cooler combining a flexible perovskite-silicon solar lid, a solid-state "
            "thermoelectric Peltier module, and a dual-layer paraffin phase-change material (PCM) jacket "
            "controlled by a tinyML predictive microcontroller that pre-cools the PCM during peak solar irradiance."
        ),
        "components": (
            "1. Flexible solar canopy lid; 2. Dual-stage PCM thermal reservoir (2-8 C); "
            "3. Closed-loop tinyML irradiance & ambient forecast controller; 4. Passive vacuum-insulated liner."
        ),
        "keywords": "solar vaccine cooler, phase change material, cold chain, thermoelectric Peltier, tinyML",
        "domain": "Biomedical Devices / Sustainable Energy",
    },
    "Federated Edge Anomaly Detection for Smart Microgrids": {
        "title": "Privacy-Preserving Federated Graph Neural Network for Microgrid Fault & Theft Detection",
        "problem": (
            "Utility smart meters expose household occupancy patterns when raw time-series load data "
            "is centralized for electricity theft and line-fault detection."
        ),
        "solution": (
            "A hierarchical federated learning framework running lightweight quantized temporal graph "
            "neural networks directly on substation concentrators, exchanging differentially private "
            "topology gradients rather than raw consumer consumption curves."
        ),
        "components": (
            "1. Edge smart-meter quantized GNN encoder; 2. Differential privacy gradient clipper; "
            "3. Substation hierarchical aggregator; 4. Cross-phase impedance fault classifier."
        ),
        "keywords": "federated learning, smart grid, graph neural network, differential privacy, electricity theft",
        "domain": "Energy Informatics / Cybersecurity",
    },
    "Acoustic Microplastic Sensor for Tap Water Monitoring": {
        "title": "Inline Ultrasonic Standing-Wave & Optical Scattering Microplastic Sensor for Household Taps",
        "problem": (
            "Existing microplastic detection requires laboratory Raman spectroscopy or Nile Red staining, "
            "making continuous household water-quality monitoring infeasible."
        ),
        "solution": (
            "A compact faucet-mounted microfluidic channel that uses piezoelectric acoustic radiation forces "
            "to focus suspended polymer particles into a single streamline passing a dual-wavelength "
            "polarized laser photodiode array to distinguish microplastics from mineral silt."
        ),
        "components": (
            "1. Piezoelectric acoustic standing-wave focuser; 2. Microfluidic quartz flow cell; "
            "3. Dual-wavelength polarized scatter detector; 4. On-board particle-size classifier."
        ),
        "keywords": "microplastic sensor, ultrasonic focusing, acoustic microfluidic, optical scattering, water quality",
        "domain": "Environmental Sensing / MEMS",
    },
    "Autonomous Precision Weed-Zapping Agricultural Rover": {
        "title": "Solar-Assisted Vision-Guided Laser Weeding Rover with Real-Time Crop Root Boundary Mapping",
        "problem": (
            "Chemical herbicides contaminate soil and groundwater, while mechanical tillers damage shallow "
            "crop roots and trigger soil erosion on small farms."
        ),
        "solution": (
            "A lightweight four-wheel agricultural rover using stereo RGB-NIR cameras and a pruned edge "
            "segmentation model to distinguish crop seedlings from invasive weeds, aiming a galvo-steered "
            "diode laser burst exclusively at weed meristems without touching soil."
        ),
        "components": (
            "1. Stereo RGB-NIR vision rig; 2. Edge meristem segmentation accelerator; "
            "3. Dual-axis galvo mirror diode laser emitter; 4. Articulated suspension rover chassis."
        ),
        "keywords": "laser weeding robot, precision agriculture, computer vision, autonomous rover, herbicide free",
        "domain": "AgriTech / Robotics",
    },
    "Immersion-Cooled EV Battery Pack with Acoustic Thermal Runaway Early Warning": {
        "title": "Dielectric Immersion-Cooled Lithium-Ion Module with Piezoelectric Gas-Vent Acoustic Sensing",
        "problem": (
            "Conventional EV battery thermal sensors react seconds after internal cell venting begins, "
            "leaving insufficient time to arrest cascading thermal runaway."
        ),
        "solution": (
            "A modular battery pack submerged in biodegradable ester dielectric fluid with embedded "
            "ultrasonic piezoelectric transducers along busbars that detect sub-millisecond micro-bubble "
            "nucleation and electrolyte off-gassing before cell temperatures spike."
        ),
        "components": (
            "1. Biodegradable dielectric immersion manifold; 2. Busbar-mounted ultrasonic piezo array; "
            "3. High-frequency acoustic anomaly DSP; 4. Rapid directional coolant purge valve."
        ),
        "keywords": "battery thermal runaway, immersion cooling, acoustic emission sensor, electric vehicle, BMS",
        "domain": "EV Powertrain / Battery Safety",
    },
    "Haptic Braille & Sign-Language Translation Smart Glove": {
        "title": "Bidirectional Piezo-Textured Smart Glove for Deafblind Tactile Braille & Gesture Translation",
        "problem": (
            "Assistive communication devices rarely support real-time two-way tactile conversation for "
            "deafblind users without bulky external refreshable Braille displays."
        ),
        "solution": (
            "A washable textile glove integrating stretchable strain sensors on finger joints to translate "
            "sign gestures to speech, paired with micro-actuated fingertip piezo-braille pins that render "
            "incoming speech-to-text directly onto the wearer's phalanges."
        ),
        "components": (
            "1. Stretchable carbon-elastomer flex sensors; 2. Fingertip 6-dot micro-piezo actuator cells; "
            "3. BLE low-latency phoneme-to-Braille microcontroller; 4. Washable conductive thread harness."
        ),
        "keywords": "haptic braille glove, sign language translator, assistive wearable, piezo actuator, deafblind",
        "domain": "Assistive Technology / Wearables",
    },
    "Biodegradable Mycelium-Chitosan Active Food Packaging Film": {
        "title": "Colorimetric Anthocyanin-Doped Mycelium-Chitosan Packaging Film for Real-Time Spoilage Sensing",
        "problem": (
            "Static printed expiry dates cause massive food waste while petroleum plastics persist in landfills."
        ),
        "solution": (
            "A fully compostable bio-composite film grown from agricultural husk mycelium cross-linked with "
            "chitosan nanoparticles and natural anthocyanin pH indicators that visibly shift from violet to "
            "green when volatile amine spoilage gases accumulate inside sealed packages."
        ),
        "components": (
            "1. Mycelium-chitosan biopolymer matrix; 2. Anthocyanin colorimetric indicator layer; "
            "3. Beeswax moisture-barrier coating; 4. Smartphone QR color-calibration patch."
        ),
        "keywords": "smart food packaging, biodegradable film, spoilage sensor, chitosan, mycelium",
        "domain": "Sustainable Materials / Food Tech",
    },
    "Zero-Trust Hardware PUF Authentication for Medical IoT Implants": {
        "title": "Ultra-Low-Power SRAM-PUF & Ultrasonic Wake-Up Authentication for Implantable Medical Devices",
        "problem": (
            "Wireless pacemakers and insulin pumps face battery drain and replay/spoofing attacks when using "
            "always-on RF radios and stored cryptographic keys."
        ),
        "solution": (
            "An implantable security co-processor that remains in nanowatt sleep until woken by a coded "
            "near-field acoustic pulse from a clinician programmer, deriving ephemeral session keys on-the-fly "
            "from intrinsic silicon SRAM Physical Unclonable Function (PUF) bit variations."
        ),
        "components": (
            "1. Nanowatt ultrasonic wake-up receiver; 2. Intrinsic SRAM-PUF key generator; "
            "3. Fuzzy extractor error-correction logic; 4. Ephemeral lightweight AEAD cipher."
        ),
        "keywords": "implantable medical device security, physical unclonable function, SRAM PUF, ultrasonic wakeup",
        "domain": "Cybersecurity / Biomedical IoT",
    },
}

CUSTOM_PRESETS_FILE = BASE_DIR / "custom_presets.json"


def load_custom_presets() -> Dict[str, Dict[str, str]]:
    """Load user-saved competition/participant invention presets from disk."""
    import json

    if not CUSTOM_PRESETS_FILE.exists():
        return {}
    try:
        data = json.loads(CUSTOM_PRESETS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): v for k, v in data.items() if isinstance(v, dict)}
    except Exception:
        pass
    return {}


def save_custom_preset(name: str, payload: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """Persist a custom participant invention preset to disk and return updated map."""
    import json

    existing = load_custom_presets()
    existing[name.strip()] = payload
    CUSTOM_PRESETS_FILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return existing

