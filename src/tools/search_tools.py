"""External patent, research-paper, and commercial-product search tools for PatentQ.

Supports free, no-key-required public sources:
- EuropePMC Patents (SRC:PAT — real US/EP/WO/CN/KR patent records)
- Google Patents (public XHR with automatic EuropePMC fallback on rate-limit)
- arXiv (Atom XML API)
- OpenAlex (Scholarly works REST API)
- Crossref (REST API)
- Online Shopping Platforms (Made-in-China & Snapdeal public catalog search with product links & images)
And optional PatentsView API v1 (when PATENTSVIEW_API_KEY is supplied).
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from src.models import PriorArtDocument

DEFAULT_HEADERS = {"User-Agent": "PatentQ-Screener/1.0 (academic-prior-art-research)"}
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    )
}
REQUEST_TIMEOUT = 5
_CACHE_TTL_SECONDS = 600
_CACHE_LOCK = threading.Lock()
_SEARCH_CACHE: Dict[str, Tuple[float, List[PriorArtDocument], List[str]]] = {}

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into",
    "is", "of", "on", "or", "the", "to", "with", "using", "based", "via", "system",
    "method", "apparatus", "device", "solutions", "adaptive", "off", "grid",
    "powered", "smart", "portable", "novel", "integrated", "efficient",
}


def _clean_text(raw: Optional[str]) -> str:
    """Strip HTML tags/entities and collapse whitespace."""
    if not raw:
        return ""
    unescaped = html.unescape(raw)
    text = BeautifulSoup(unescaped, "html.parser").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def _extract_core_terms(query: str, max_terms: int = 5) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]{3,}", (query or "").lower())
    filtered = [t for t in tokens if t not in _STOPWORDS]
    deduped = list(dict.fromkeys(filtered or tokens))
    return deduped[:max_terms]


def _reconstruct_openalex_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """Rebuild abstract plain text from OpenAlex abstract_inverted_index."""
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    max_pos = -1
    for positions in inverted_index.values():
        if positions:
            max_pos = max(max_pos, max(positions))
    if max_pos < 0:
        return ""
    words = [""] * (max_pos + 1)
    for token, positions in inverted_index.items():
        for pos in positions:
            if 0 <= pos <= max_pos:
                words[pos] = token
    return " ".join(w for w in words if w).strip()


def search_europepmc_patents(query: str, max_results: int = 4) -> Tuple[List[PriorArtDocument], Optional[str]]:
    """Query EuropePMC patent corpus (SRC:PAT) for real US/EP/CN/WO/KR patents without API keys."""
    docs: List[PriorArtDocument] = []
    terms = _extract_core_terms(query, max_terms=5)
    if not terms:
        return docs, None

    # Build a high-recall boolean query: require at least top 2 terms + OR cluster of remaining terms
    if len(terms) >= 3:
        epmc_query = f"SRC:PAT AND ({terms[0]} OR {terms[1]}) AND ({' OR '.join(terms[2:])})"
    else:
        epmc_query = f"SRC:PAT AND ({' OR '.join(terms)})"

    try:
        resp = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={
                "query": epmc_query,
                "format": "json",
                "pageSize": max_results,
                "resultType": "core",
            },
            headers=DEFAULT_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("resultList", {}).get("result", [])

        # Broaden once if strict intersection returned 0 hits
        if not results and len(terms) >= 2:
            broad_q = f"SRC:PAT AND ({' OR '.join(terms[:3])})"
            resp = requests.get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params={"query": broad_q, "format": "json", "pageSize": max_results, "resultType": "core"},
                headers=DEFAULT_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            results = resp.json().get("resultList", {}).get("result", [])

        for item in results[:max_results]:
            pid = str(item.get("id") or "").strip()
            title = _clean_text(item.get("title"))
            if not pid or not title:
                continue
            abstract = _clean_text(item.get("abstractText")) or f"Registered patent document {pid}: {title}."
            assignee = _clean_text(item.get("authorString")) or "Registered Patent Assignee"
            year = str(item.get("pubYear") or "N/A")
            docs.append(
                PriorArtDocument(
                    doc_id=f"PAT-{pid}",
                    doc_type="patent",
                    title=title,
                    abstract=abstract,
                    authors_or_assignee=assignee,
                    year=year,
                    url=f"https://patents.google.com/patent/{pid}/en",
                    source_api="EuropePMC Patents",
                )
            )
        return docs, None
    except Exception as exc:
        return docs, f"EuropePMC Patents lookup note: {exc}"


def search_google_patents(query: str, max_results: int = 4) -> Tuple[List[PriorArtDocument], Optional[str]]:
    """Query Google Patents public XHR endpoint, automatically falling back to EuropePMC Patents on rate-limit."""
    docs: List[PriorArtDocument] = []
    if not query.strip():
        return docs, None
    try:
        inner = urllib.parse.urlencode({"q": query, "oq": query})
        url = f"https://patents.google.com/xhr/query?url={urllib.parse.quote(inner, safe='')}"
        resp = requests.get(url, headers=BROWSER_HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code in {429, 503}:
            # Seamless silent fallback to EuropePMC Patents so UI stays clean and still retrieves patents
            fallback_docs, _ = search_europepmc_patents(query, max_results=max_results)
            return fallback_docs, None
        resp.raise_for_status()
        payload = resp.json()
        clusters = payload.get("results", {}).get("cluster", [])
        for cluster in clusters:
            for item in cluster.get("result", []):
                patent = item.get("patent", {})
                pub_num = (patent.get("publication_number") or "").strip()
                title = _clean_text(patent.get("title"))
                snippet = _clean_text(patent.get("snippet"))
                if not pub_num or not title:
                    continue
                pub_date = (patent.get("publication_date") or "").strip()
                year = pub_date[:4] if len(pub_date) >= 4 else "N/A"
                assignee = _clean_text(patent.get("assignee")) or _clean_text(patent.get("inventor")) or "Unknown Assignee"
                docs.append(
                    PriorArtDocument(
                        doc_id=f"PAT-{pub_num}",
                        doc_type="patent",
                        title=title,
                        abstract=snippet or f"Patent publication {pub_num}: {title}.",
                        authors_or_assignee=assignee,
                        year=year,
                        url=f"https://patents.google.com/patent/{pub_num}/en",
                        source_api="Google Patents",
                    )
                )
                if len(docs) >= max_results:
                    return docs, None
        if not docs:
            return search_europepmc_patents(query, max_results=max_results)
        return docs, None
    except Exception:
        return search_europepmc_patents(query, max_results=max_results)


def search_patentsview(
    query: str,
    max_results: int = 5,
    api_key: Optional[str] = None,
) -> Tuple[List[PriorArtDocument], Optional[str]]:
    """Query USPTO PatentsView API when an API key is provided."""
    docs: List[PriorArtDocument] = []
    key = (api_key or os.getenv("PATENTSVIEW_API_KEY", "")).strip()
    if not key:
        return docs, None
    try:
        endpoint = "https://search.patentsview.org/api/v1/patent/"
        params = {
            "q": json.dumps({"_text_any": {"patent_abstract": query}}),
            "f": json.dumps(["patent_id", "patent_title", "patent_abstract", "patent_date"]),
            "o": json.dumps({"size": max_results}),
        }
        resp = requests.get(
            endpoint,
            params=params,
            headers={**DEFAULT_HEADERS, "X-Api-Key": key},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        for pat in data.get("patents", [])[:max_results]:
            pid = str(pat.get("patent_id", "")).strip()
            title = _clean_text(pat.get("patent_title"))
            abstract = _clean_text(pat.get("patent_abstract"))
            if not pid or not title:
                continue
            pdate = str(pat.get("patent_date") or "")
            docs.append(
                PriorArtDocument(
                    doc_id=f"PAT-US{pid}",
                    doc_type="patent",
                    title=title,
                    abstract=abstract or title,
                    authors_or_assignee="USPTO / PatentsView",
                    year=pdate[:4] if len(pdate) >= 4 else "N/A",
                    url=f"https://patents.google.com/patent/US{pid}/en",
                    source_api="PatentsView",
                )
            )
        return docs, None
    except Exception as exc:
        return docs, f"PatentsView lookup warning: {exc}"


def search_arxiv_papers(query: str, max_results: int = 5) -> Tuple[List[PriorArtDocument], Optional[str]]:
    """Query arXiv Atom API for preprints and research papers."""
    docs: List[PriorArtDocument] = []
    if not query.strip():
        return docs, None
    try:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
        }
        resp = requests.get(
            "http://export.arxiv.org/api/query",
            params=params,
            headers=DEFAULT_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("atom:entry", ns):
            raw_id = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
            arxiv_id = raw_id.rsplit("/", 1)[-1] if raw_id else ""
            title = _clean_text(entry.findtext("atom:title", default="", namespaces=ns))
            abstract = _clean_text(entry.findtext("atom:summary", default="", namespaces=ns))
            published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
            authors = [
                _clean_text(a.findtext("atom:name", default="", namespaces=ns))
                for a in entry.findall("atom:author", ns)
            ]
            authors_str = ", ".join([a for a in authors if a][:3])
            if len(authors) > 3:
                authors_str += " et al."
            if not arxiv_id or not title:
                continue
            docs.append(
                PriorArtDocument(
                    doc_id=f"ARXIV-{arxiv_id}",
                    doc_type="paper",
                    title=title,
                    abstract=abstract or title,
                    authors_or_assignee=authors_str or "arXiv Author(s)",
                    year=published[:4] if len(published) >= 4 else "N/A",
                    url=raw_id or f"https://arxiv.org/abs/{arxiv_id}",
                    source_api="arXiv",
                )
            )
        return docs, None
    except Exception as exc:
        return docs, f"arXiv lookup warning: {exc}"


def search_openalex_papers(query: str, max_results: int = 5) -> Tuple[List[PriorArtDocument], Optional[str]]:
    """Query OpenAlex scholarly works API."""
    docs: List[PriorArtDocument] = []
    if not query.strip():
        return docs, None
    try:
        resp = requests.get(
            "https://api.openalex.org/works",
            params={"search": query, "per-page": max_results},
            headers=DEFAULT_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        for work in data.get("results", [])[:max_results]:
            oa_id = (work.get("id") or "").rsplit("/", 1)[-1]
            title = _clean_text(work.get("display_name") or work.get("title"))
            if not oa_id or not title:
                continue
            abstract = _reconstruct_openalex_abstract(work.get("abstract_inverted_index"))
            year = str(work.get("publication_year") or "N/A")
            authorships = work.get("authorships", [])
            names = [
                _clean_text((a.get("author") or {}).get("display_name"))
                for a in authorships[:3]
            ]
            authors_str = ", ".join(n for n in names if n) or "OpenAlex Author(s)"
            doi_url = work.get("doi") or work.get("id") or ""
            docs.append(
                PriorArtDocument(
                    doc_id=f"OALEX-{oa_id}",
                    doc_type="paper",
                    title=title,
                    abstract=abstract or f"Scholarly work indexed in OpenAlex ({year}): {title}.",
                    authors_or_assignee=authors_str,
                    year=year,
                    url=doi_url,
                    source_api="OpenAlex",
                )
            )
        return docs, None
    except Exception as exc:
        return docs, f"OpenAlex lookup warning: {exc}"


def search_crossref_papers(query: str, max_results: int = 5) -> Tuple[List[PriorArtDocument], Optional[str]]:
    """Query Crossref REST API for published papers and DOIs."""
    docs: List[PriorArtDocument] = []
    if not query.strip():
        return docs, None
    try:
        resp = requests.get(
            "https://api.crossref.org/works",
            params={"query": query, "rows": max_results},
            headers=DEFAULT_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        items = resp.json().get("message", {}).get("items", [])
        for idx, item in enumerate(items[:max_results]):
            doi = (item.get("DOI") or f"cr-{idx}").strip()
            titles = item.get("title") or []
            title = _clean_text(titles[0]) if titles else ""
            if not title:
                continue
            abstract = _clean_text(item.get("abstract"))
            year_parts = (item.get("published") or {}).get("date-parts") or [[]]
            year = str(year_parts[0][0]) if year_parts and year_parts[0] else "N/A"
            authors_list = item.get("author") or []
            names = [
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in authors_list[:3]
            ]
            authors_str = ", ".join(n for n in names if n) or "Crossref Author(s)"
            safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", doi)[:32]
            docs.append(
                PriorArtDocument(
                    doc_id=f"DOI-{safe_id}",
                    doc_type="paper",
                    title=title,
                    abstract=abstract or f"Published scholarly record ({doi}): {title}.",
                    authors_or_assignee=authors_str,
                    year=year,
                    url=item.get("URL") or f"https://doi.org/{doi}",
                    source_api="Crossref",
                )
            )
        return docs, None
    except Exception as exc:
        return docs, f"Crossref lookup warning: {exc}"


def _search_made_in_china_products(query: str, max_results: int = 3) -> List[PriorArtDocument]:
    """Fetch commercial product listings with links and product images from Made-in-China."""
    docs: List[PriorArtDocument] = []
    terms = _extract_core_terms(query, max_terms=3)
    if not terms:
        return docs
    candidate_slugs = ["_".join(t.capitalize() for t in terms)]
    if len(terms) >= 3:
        candidate_slugs.append(f"{terms[0].capitalize()}_{terms[-1].capitalize()}")
    for slug in candidate_slugs:
        url = f"https://www.made-in-china.com/products-search/hot-china-products/{urllib.parse.quote(slug)}.html"
        try:
            resp = requests.get(url, headers=BROWSER_HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            for h2 in soup.select("h2.product-name"):
                a = h2.select_one("a")
                if not a or not a.get("href"):
                    continue
                title = _clean_text(a.get_text(" ", strip=True))
                prod_url = a["href"].strip()
                if prod_url.startswith("//"):
                    prod_url = "https:" + prod_url
                img_url = ""
                price_txt = ""
                node = h2
                for _ in range(6):
                    if node.parent is None:
                        break
                    node = node.parent
                    imgs = [im.get("data-original") or im.get("src") or "" for im in node.find_all("img")]
                    prod_imgs = [u for u in imgs if "image.made-in-china.com" in u]
                    if prod_imgs:
                        img_url = prod_imgs[0]
                        p_el = node.select_one(".price")
                        price_txt = _clean_text(p_el.get_text(" ", strip=True)) if p_el else ""
                        break
                if img_url.startswith("//"):
                    img_url = "https:" + img_url
                if not title or not prod_url:
                    continue
                pid = hashlib.md5(prod_url.encode("utf-8")).hexdigest()[:8].upper()
                docs.append(
                    PriorArtDocument(
                        doc_id=f"PROD-MIC-{pid}",
                        doc_type="product",
                        title=title,
                        abstract=(
                            f"Commercial product listed on Made-in-China ({price_txt or 'Price on request'}). "
                            f"Relevant for §102(a)(1) public-use / on-sale commercial prior-art screening."
                        ),
                        authors_or_assignee="Made-in-China Marketplace",
                        year="Active Listing",
                        url=prod_url,
                        image_url=img_url,
                        price=price_txt or "Price on request",
                        source_api="Made-in-China",
                    )
                )
                if len(docs) >= max_results:
                    return docs
            if docs:
                return docs
        except Exception:
            continue
    return docs


def _search_snapdeal_products(query: str, max_results: int = 3) -> List[PriorArtDocument]:
    """Fetch commercial product listings with links and product images from Snapdeal."""
    docs: List[PriorArtDocument] = []
    terms = _extract_core_terms(query, max_terms=4)
    if not terms:
        return docs
    kw = " ".join(terms)
    try:
        resp = requests.get(
            "https://www.snapdeal.com/search",
            params={"keyword": kw, "noOfResults": max_results},
            headers=BROWSER_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            return docs
        soup = BeautifulSoup(resp.text, "html.parser")
        for card in soup.select("div.product-tuple-listing"):
            t_el = card.select_one("p.product-title")
            a_el = card.select_one("a.dp-widget-link")
            p_el = card.select_one("span.product-price")
            img_el = card.select_one("img")
            title = _clean_text(t_el.get_text()) if t_el else ""
            prod_url = (a_el.get("href") or "").strip() if a_el else ""
            price_txt = _clean_text(p_el.get_text()) if p_el else ""
            img_url = ((img_el.get("src") or img_el.get("data-src") or "").strip()) if img_el else ""
            if not title or not prod_url:
                continue
            pid = hashlib.md5(prod_url.encode("utf-8")).hexdigest()[:8].upper()
            docs.append(
                PriorArtDocument(
                    doc_id=f"PROD-SD-{pid}",
                    doc_type="product",
                    title=title,
                    abstract=(
                        f"Commercial retail product listed on Snapdeal ({price_txt or 'Available'}). "
                        f"Relevant for §102(a)(1) on-sale commercial prior-art screening."
                    ),
                    authors_or_assignee="Snapdeal Marketplace",
                    year="Active Listing",
                    url=prod_url,
                    image_url=img_url,
                    price=price_txt or "Listed Online",
                    source_api="Snapdeal",
                )
            )
            if len(docs) >= max_results:
                break
    except Exception:
        pass
    return docs


def search_commercial_products(query: str, max_results: int = 4) -> Tuple[List[PriorArtDocument], Optional[str]]:
    """Query public online shopping platforms concurrently for commercial products with links & images."""
    per_site = max(2, (max_results + 1) // 2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_mic = pool.submit(_search_made_in_china_products, query, per_site)
        f_sd = pool.submit(_search_snapdeal_products, query, per_site)
        mic = f_mic.result()
        sd = f_sd.result()
    combined = (mic + sd)[:max_results]
    return combined, None


def run_multi_source_search(
    queries: List[str],
    sources: Optional[List[str]] = None,
    max_per_source: int = 4,
    patentsview_api_key: Optional[str] = None,
) -> Tuple[List[PriorArtDocument], List[str]]:
    """Execute queries concurrently across selected patent, paper, and online product sources."""
    selected = sources or [
        "EuropePMC Patents",
        "Google Patents",
        "arXiv",
        "OpenAlex",
        "Crossref",
        "Online Shopping Products",
    ]
    clean_queries = [q.strip() for q in queries[:3] if q and q.strip()]
    if not clean_queries:
        return [], []

    cache_key = json.dumps(
        {"q": clean_queries, "s": sorted(selected), "m": max_per_source, "pv": bool(patentsview_api_key)},
        sort_keys=True,
    )
    now = time.time()
    with _CACHE_LOCK:
        cached = _SEARCH_CACHE.get(cache_key)
        if cached and (now - cached[0]) < _CACHE_TTL_SECONDS:
            return [d.model_copy() for d in cached[1]], list(cached[2])

    tasks = []
    google_patents_queued = False
    shopping_queued = False

    with ThreadPoolExecutor(max_workers=8) as pool:
        for q in clean_queries:
            if "EuropePMC Patents" in selected:
                tasks.append(pool.submit(search_europepmc_patents, q, max_per_source))

            if "Google Patents" in selected and not google_patents_queued:
                google_patents_queued = True
                tasks.append(pool.submit(search_google_patents, q, max_per_source))

            if "PatentsView" in selected:
                tasks.append(pool.submit(search_patentsview, q, max_per_source, patentsview_api_key))

            if "arXiv" in selected:
                tasks.append(pool.submit(search_arxiv_papers, q, max_per_source))

            if "OpenAlex" in selected:
                tasks.append(pool.submit(search_openalex_papers, q, max_per_source))

            if "Crossref" in selected:
                tasks.append(pool.submit(search_crossref_papers, q, max_per_source))

            if "Online Shopping Products" in selected and not shopping_queued:
                shopping_queued = True
                tasks.append(pool.submit(search_commercial_products, q, max(4, max_per_source)))

        all_docs: List[PriorArtDocument] = []
        warnings: List[str] = []
        seen_keys = set()

        for fut in as_completed(tasks):
            try:
                batch, warn = fut.result()
            except Exception:
                continue
            if warn:
                warnings.append(warn)
            for doc in batch:
                norm_key = (doc.doc_id.lower(), doc.title.lower()[:80])
                if norm_key not in seen_keys:
                    seen_keys.add(norm_key)
                    all_docs.append(doc)

    unique_warnings = list(dict.fromkeys(warnings))
    with _CACHE_LOCK:
        _SEARCH_CACHE[cache_key] = (time.time(), [d.model_copy() for d in all_docs], list(unique_warnings))
    return all_docs, unique_warnings


# ---------------------------------------------------------------------------
# LangChain @tool wrappers for Agentic / ReAct inspection
# ---------------------------------------------------------------------------


@tool
def patent_search_tool(query: str) -> str:
    """Search EuropePMC Patents and Google Patents for granted patents and patent applications."""
    docs, _ = search_europepmc_patents(query, max_results=4)
    if not docs:
        docs, _ = search_google_patents(query, max_results=4)
    if not docs:
        return f"No patents found for query: {query}"
    return "\n".join(f"[{d.doc_id}] ({d.year}) {d.title} — {d.abstract[:220]}" for d in docs)


@tool
def paper_search_tool(query: str) -> str:
    """Search arXiv and OpenAlex research papers for prior academic publications."""
    d1, w1 = search_arxiv_papers(query, max_results=3)
    d2, w2 = search_openalex_papers(query, max_results=3)
    docs = d1 + d2
    if not docs:
        return "; ".join(w for w in [w1, w2] if w) or f"No papers found for query: {query}"
    return "\n".join(f"[{d.doc_id}] ({d.year}) {d.title} — {d.abstract[:220]}" for d in docs)


@tool
def commercial_product_search_tool(query: str) -> str:
    """Search online shopping platforms for commercial products matching an invention (§102 on-sale prior art)."""
    docs, _ = search_commercial_products(query, max_results=4)
    if not docs:
        return f"No commercial products found for query: {query}"
    return "\n".join(
        f"[{d.doc_id}] ({d.source_api}, {d.price}) {d.title} — {d.url}" for d in docs
    )


def get_langchain_tools() -> List:
    """Return LangChain tool objects exposed to the PatentQ agent layer."""
    return [patent_search_tool, paper_search_tool, commercial_product_search_tool]
