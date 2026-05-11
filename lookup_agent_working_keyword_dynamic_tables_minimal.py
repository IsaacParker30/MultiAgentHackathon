import argparse
import html
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import autogen
import requests


# =========================
# DEFAULTS
# =========================

DEFAULT_QUESTION = "What is the adsorption energy of water on hBN?"
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_OUTPUT_DIR = "outputs"


# =========================
# SMALL UTILITIES
# =========================

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_for_match(text):
    text = clean_text(text).lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9+()\-./\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_xml(text, tag):
    match = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", text, flags=re.DOTALL)
    return clean_text(match.group(1)) if match else ""


def extract_arxiv_id(entry):
    match = re.search(r"<id>http://arxiv.org/abs/(.*?)</id>", entry)
    return clean_text(match.group(1)) if match else None


def safe_json_from_text(text):
    """Extract first JSON object from an LLM response."""
    text = (text or "").replace("TERMINATE", "").strip()
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return None


def compact_unique(items):
    out = []
    seen = set()
    for item in items or []:
        if item is None:
            continue
        item = clean_text(str(item))
        if not item:
            continue
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def terms_from_profile(profile):
    """Flatten the dynamic relevance profile into searchable terms."""
    if not profile:
        return []

    fields = [
        "core_entities",
        "entity_aliases",
        "target_property",
        "property_aliases",
        "required_context",
        "useful_methods",
        "comparable_systems",
        "extra_search_terms",
    ]

    terms = []
    for field in fields:
        value = profile.get(field)
        if isinstance(value, list):
            terms.extend(value)
        elif isinstance(value, dict):
            for key, aliases in value.items():
                terms.append(key)
                if isinstance(aliases, list):
                    terms.extend(aliases)
                elif isinstance(aliases, str):
                    terms.append(aliases)
        elif isinstance(value, str):
            terms.append(value)

    return compact_unique(terms)


def simple_terms_from_queries(queries):
    """Emergency non-hardcoded terms from the LLM keyword queries, not from hand-written chemistry rules."""
    terms = []
    for q in queries:
        qn = normalize_for_match(q)
        # Preserve short formula-like tokens if present, but skip purely generic words.
        for tok in re.findall(r"[a-z0-9+()\-./]+", qn):
            if len(tok) >= 3:
                terms.append(tok)
    return compact_unique(terms)


# =========================
# SEARCH LAYER
# =========================

def search_semantic_scholar(query, limit=5, timeout=20):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,year,abstract,externalIds,url,venue,citationCount",
    }

    try:
        response = requests.get(url, params=params, timeout=timeout)
    except Exception as e:
        return {"source": "semantic_scholar", "query": query, "status": "error", "error": str(e), "papers": []}

    if response.status_code == 429:
        return {
            "source": "semantic_scholar",
            "query": query,
            "status": "rate_limited",
            "error": "Semantic Scholar rate limit reached. Skipping retries.",
            "papers": [],
        }

    if not response.ok:
        return {"source": "semantic_scholar", "query": query, "status": "error", "error": f"HTTP {response.status_code}", "papers": []}

    data = response.json()
    papers = []
    for p in data.get("data", []):
        external_ids = p.get("externalIds") or {}
        authors = ", ".join(a.get("name", "") for a in p.get("authors", [])[:8])
        papers.append({
            "source": "Semantic Scholar",
            "title": clean_text(p.get("title")),
            "authors": clean_text(authors),
            "year": p.get("year"),
            "venue": clean_text(p.get("venue")),
            "doi": external_ids.get("DOI"),
            "arxiv": external_ids.get("ArXiv"),
            "url": p.get("url"),
            "citation_count": p.get("citationCount"),
            "abstract": clean_text(p.get("abstract")),
        })

    return {"source": "semantic_scholar", "query": query, "status": "ok", "error": None, "papers": papers}


def search_arxiv(query, limit=5, timeout=20):
    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": limit,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    try:
        response = requests.get(url, params=params, timeout=timeout)
    except Exception as e:
        return {"source": "arxiv", "query": query, "status": "error", "error": str(e), "papers": []}

    if not response.ok:
        return {"source": "arxiv", "query": query, "status": "error", "error": f"HTTP {response.status_code}", "papers": []}

    entries = response.text.split("<entry>")[1:]
    papers = []
    for entry in entries:
        arxiv_id = extract_arxiv_id(entry)
        papers.append({
            "source": "arXiv",
            "title": extract_xml(entry, "title"),
            "authors": "arXiv authors not parsed",
            "year": (extract_xml(entry, "published") or "")[:4] or None,
            "venue": "arXiv",
            "doi": None,
            "arxiv": arxiv_id,
            "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
            "citation_count": None,
            "abstract": extract_xml(entry, "summary"),
        })

    return {"source": "arxiv", "query": query, "status": "ok", "error": None, "papers": papers}


# =========================
# KEYWORD QUERY CLEANING
# =========================

def clean_search_query(q):
    """Make sure planned queries are keyword-style, not full English questions."""
    q = clean_text(q)
    q = q.replace("TERMINATE", "").strip()
    q = re.sub(r"^[-*\d.)\s]+", "", q).strip()
    q = re.sub(r"\?$", "", q).strip()

    # Light cleanup only. This is not a hard-coded science filter.
    q = re.sub(r"\b(what|which|where|why|how)\b", "", q, flags=re.I)
    q = re.sub(r"\b(is|are|does|do|did|exist|exists|there|reported|including|include)\b", "", q, flags=re.I)
    q = re.sub(r"\b(e\.g\.|for example)\b", "", q, flags=re.I)
    q = re.sub(r"[?:;,]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def parse_queries(text, max_queries=15):
    queries = []
    for line in text.splitlines():
        q = clean_search_query(line)
        if q and len(q) > 3 and q not in queries:
            queries.append(q)
    return queries[:max_queries]


# =========================
# DYNAMIC RELEVANCE FILTER
# =========================

def paper_text_fields(paper):
    title = normalize_for_match(paper.get("title") or "")
    abstract = normalize_for_match(paper.get("abstract") or "")
    venue = normalize_for_match(paper.get("venue") or "")
    return title, abstract, venue, f"{title} {abstract} {venue}"


def score_term_match(term, title, abstract, venue):
    term = normalize_for_match(term)
    if not term:
        return 0

    # Exact phrase match. Short formula terms like h-bn still work.
    is_phrase = " " in term or "-" in term or "/" in term

    if term in title:
        return 7 if is_phrase else 5
    if term in abstract:
        return 4 if is_phrase else 2
    if term in venue:
        return 1
    return 0


def relevance_score(paper, profile_terms, planned_queries):
    """
    Loose dynamic score.
    No fixed hBN/water aliases, no fixed junk list.
    Uses only LLM-generated profile terms + LLM-generated search query terms.
    """
    title, abstract, venue, text = paper_text_fields(paper)

    score = 0
    matched = []

    all_terms = compact_unique(list(profile_terms) + simple_terms_from_queries(planned_queries))

    for term in all_terms:
        s = score_term_match(term, title, abstract, venue)
        if s:
            score += s
            matched.append(term)

    # Small bonus when the source query itself overlaps with the paper.
    source_query = normalize_for_match(paper.get("source_query") or "")
    for tok in re.findall(r"[a-z0-9+()\-./]+", source_query):
        if len(tok) >= 4 and tok in text:
            score += 1

    # Citation count is not relevance, but helps sort ties.
    paper["matched_terms"] = compact_unique(matched)[:25]
    return score


def filter_relevant_papers(results, profile_terms, planned_queries, min_score=1, keep_top=30, rescue_top=10):
    candidates = []
    for result in results:
        for p in result.get("papers", []):
            p = dict(p)
            p["source_query"] = result.get("query")
            p["search_source"] = result.get("source")
            p["relevance_score"] = relevance_score(p, profile_terms, planned_queries)
            candidates.append(p)

    # Deduplicate
    seen = set()
    unique = []
    for p in candidates:
        key = (p.get("doi") or p.get("arxiv") or p.get("title") or "").lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(p)

    unique.sort(key=lambda x: (x.get("relevance_score", 0), x.get("citation_count") or 0), reverse=True)

    retained = [p for p in unique if p.get("relevance_score", 0) >= min_score][:keep_top]

    # Important: do not let the whole pipeline die silently. If all scores are weak,
    # keep the top raw results as "low-confidence candidates" for the agents to inspect.
    if not retained and unique:
        retained = unique[:rescue_top]
        for p in retained:
            p["low_confidence_rescue"] = True

    return retained, unique[:50]


def papers_to_markdown(papers):
    if not papers:
        return "No relevant papers found."

    blocks = []
    for i, p in enumerate(papers, 1):
        rescue = "Yes" if p.get("low_confidence_rescue") else "No"
        blocks.append(
            f"### Paper {i}\n"
            f"Title: {p.get('title') or 'Unknown'}\n"
            f"Authors: {p.get('authors') or 'Unknown'}\n"
            f"Year: {p.get('year') or 'Unknown'}\n"
            f"Venue: {p.get('venue') or 'Unknown'}\n"
            f"DOI: {p.get('doi') or 'None'}\n"
            f"arXiv: {p.get('arxiv') or 'None'}\n"
            f"URL: {p.get('url') or 'None'}\n"
            f"Relevance score: {p.get('relevance_score')}\n"
            f"Low-confidence rescue: {rescue}\n"
            f"Matched terms: {', '.join(p.get('matched_terms') or []) or 'None'}\n"
            f"Source query: {p.get('source_query')}\n"
            f"Abstract: {p.get('abstract') or 'No abstract available.'}\n"
        )
    return "\n\n---\n\n".join(blocks)


def profile_to_markdown(profile, profile_terms):
    lines = ["# Dynamic Relevance Profile", ""]
    if profile:
        for key, value in profile.items():
            lines.append(f"## {key}")
            if isinstance(value, dict):
                for k, v in value.items():
                    lines.append(f"- **{k}**: {v}")
            elif isinstance(value, list):
                for item in value:
                    lines.append(f"- {item}")
            else:
                lines.append(str(value))
            lines.append("")
    lines.append("## Flattened terms used for scoring")
    for t in profile_terms:
        lines.append(f"- {t}")
    return "\n".join(lines)


# =========================
# AUTOGEN HELPERS
# =========================

def make_config(model):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY='your_key'")
    return {"api_type": "openai", "model": model, "api_key": api_key}


def extract_last_assistant_message(chat_result):
    messages = getattr(chat_result, "chat_history", []) or []
    for msg in reversed(messages):
        content = msg.get("content")
        if content and str(content).strip() and str(content).strip() != "TERMINATE":
            return str(content).replace("TERMINATE", "").strip()
    return "No answer captured."


def ask_agent(agent, message):
    user = autogen.UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        code_execution_config=False,
        max_consecutive_auto_reply=1,
        is_termination_msg=lambda msg: "TERMINATE" in str(msg.get("content", "")),
    )
    result = user.initiate_chat(agent, message=message)
    return extract_last_assistant_message(result)


# =========================
# AGENTS
# =========================

def build_agents(config):
    relevance_profile_agent = autogen.AssistantAgent(
        name="RelevanceProfileAgent",
        system_message="""
You are a scientific relevance-profiling assistant.

Given a scientific research question, produce a JSON profile for literature retrieval.
The goal is broad recall, not strict filtering.

Return ONLY valid JSON with these keys:
core_entities: list of the main molecules/materials/systems
entity_aliases: object mapping each core entity to spelling variants and common aliases
target_property: list of the main property/quantity being asked for
property_aliases: list of synonyms or equivalent terms used in papers
required_context: list of context words likely to appear in relevant papers
useful_methods: list of experimental or computational methods likely to appear
comparable_systems: list of related systems useful for rough comparison
extra_search_terms: list of extra short keywords useful for broad search

Rules:
- Do not hard-code any system. Only use what follows from the question.
- Include abbreviations, chemical formulae, hyphenation variants, and long names where likely.
- Prefer terms that appear in paper titles and abstracts.
- Do not include negative/exclusion terms.
- Do not explain.
End with TERMINATE.
""",
        llm_config={"config_list": [config]},
        code_execution_config=False,
    )

    search_planner_agent = autogen.AssistantAgent(
        name="SearchPlannerAgent",
        system_message="""
You are a scientific search planning assistant.

Given a scientific research question and optionally a relevance profile, propose exactly 15 keyword-based literature search queries.
They must be compact keyword queries suitable for Semantic Scholar/arXiv.

Rules:
- Do NOT write full English questions.
- Do NOT start with what/how/which/why.
- Use paper-title/abstract terminology.
- Include aliases and spelling variants from the question/profile.
- Cover direct query, experimental query, theory/simulation query, DFT/AIMD query, high-level benchmark query, and comparable-system query.
- If you don't find experimental values, or values at the level of theory needed, go to higher or lower levels of theory, at the end you should try to find some kind of value, some value at some theory (or experimental level is better than none).
- Each query must contain at most 6 whitespace-separated terms.
- Do not explain.
End with TERMINATE.
""",
        llm_config={"config_list": [config]},
        code_execution_config=False,
    )

    dft_agent = autogen.AssistantAgent(
        name="DFTParamsAgent",
        system_message="""
You are a DFT/AIMD/computational-setup assistant.

Use only the supplied question and retrieved papers/search results, plus clearly marked best-practice inference.
Do not call tools.
Do not describe the search process.
Do not ask follow-up questions.

Adapt your answer to the actual property in the question.
For example:
- diffusion question -> include MSD/Einstein relation setup
- adsorption/binding energy question -> include adsorption-energy formula and geometry setup
- spectroscopy question -> include appropriate observable and simulation/analysis route

Return:
# DFT / Simulation Setup Recommendations
## Target system
## Target property
## Recommended method
## Functional / model / force field
## Dispersion or long-range treatment
## Basis / cutoff / numerical settings
## Cell / slab / boundary conditions
## Sampling / k-points / trajectory settings
## How to compute the target property
## DFT starting-point table
Use a markdown table with columns:
Parameter | Suggested value/range | Units | Direct-system evidence | Comparable-system evidence | Reference | Confidence
Rules for this table:
- Put any concrete numerical DFT/simulation settings here, for example cutoff, basis, slab size, vacuum, k-points, timestep, trajectory length, dispersion setup, cell size, convergence thresholds, adsorption geometry, or equivalent quantities relevant to the question.
- In the Reference column, cite where the number came from using the retrieved paper number, e.g. [Paper 3](#paper-3).
- If the value is not directly reported and is only your best-practice inference, write Best-practice inference in the Reference column.
- Do not give a precise numerical value without either a paper reference or an explicit Best-practice inference label.
- Ensure in the DFT starting-point table to have the references, from what paper you got what number? 
## Parameters supported by retrieved papers
## Best-practice inferred parameters
## Missing information

End with TERMINATE.
""",
        llm_config={"config_list": [config]},
        code_execution_config=False,
    )

    data_agent = autogen.AssistantAgent(
        name="DataAgent",
        system_message="""
You are a scientific literature-data extraction assistant.

Use only the supplied question and retrieved papers/search results, plus clearly marked best-practice inference.
Do not call tools.
Do not describe the search process.
Do not ask follow-up questions.

Extract reference values and literature data relevant to the actual question.
If no direct value is found, say that clearly.

Return:
# Reference Values & Literature Data
## Direct answer
## Direct data for the requested system
## Experimental data
## Simulation / theoretical data
## Comparable systems
## Numerical values table
Use a markdown table with columns:
Quantity | Value | Units | System | Method | Conditions | Reference | Confidence
Rules for this table:
- Put all direct numerical literature values here.
- In the Reference column, cite where the number came from using the retrieved paper number, e.g. [Paper 2](#paper-2).
- If a value is only inferred from comparable systems, clearly mark it as comparable/inferred in the Conditions or Confidence column.
- Do not give a precise numerical value without a paper reference.
## Important papers found
## Gaps
## Confidence

End with TERMINATE.
""",
        llm_config={"config_list": [config]},
        code_execution_config=False,
    )

    report_agent = autogen.AssistantAgent(
        name="ReportWriterAgent",
        system_message="""
You are a scientific report writer.

Combine the supplied DFT/simulation answer and literature-data answer into a clean final report.
Do not include process logs.
Do not say "I searched".
Do not ask follow-up questions.

Return:
# Final Report
## Question
## Short Answer
## Literature Data
## DFT / Simulation Setup Recommendation
## Numerical Values
Preserve source-backed numerical values from the literature and DFT tables. Keep references as [Paper N](#paper-N) links where provided.
## Most Relevant References
## Uncertainties
## Next Best Actions

End with TERMINATE.
""",
        llm_config={"config_list": [config]},
        code_execution_config=False,
    )

    return relevance_profile_agent, search_planner_agent, dft_agent, data_agent, report_agent


def get_relevance_profile(agent, question):
    raw = ask_agent(agent, f"Question:\n{question}\n")
    profile = safe_json_from_text(raw)
    if profile is None:
        profile = {
            "core_entities": [],
            "entity_aliases": {},
            "target_property": [],
            "property_aliases": [],
            "required_context": [],
            "useful_methods": [],
            "comparable_systems": [],
            "extra_search_terms": [],
            "profile_parse_warning": "Could not parse JSON profile from LLM response.",
            "raw_response": raw,
        }
    return profile


# =========================
# HTML REPORTING
# =========================

def render_inline_markdown(text):
    """Escape text, then render a very small subset of inline markdown.
    This is only for report display; it does not affect search/filter logic.
    """
    safe = html.escape(text or "")
    safe = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", safe)
    safe = re.sub(
        r"\[([^\]]+)\]\((#[A-Za-z0-9_\-]+|https?://[^)]+)\)",
        r'<a href="\2">\1</a>',
        safe,
    )
    return safe


def simple_markdown_to_html(md):
    if not md:
        return '<p class="no-data">No answer captured.</p>'

    lines = md.splitlines()
    out = []
    in_ul = False
    in_table = False

    def close_ul():
        nonlocal in_ul
        if in_ul:
            out.append("</ul>")
            in_ul = False

    def close_table():
        nonlocal in_table
        if in_table:
            out.append("</tbody></table>")
            in_table = False

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            close_ul(); close_table()
            i += 1
            continue

        if "|" in line and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{3,}:?", lines[i + 1]):
            close_ul(); close_table()
            headers = [render_inline_markdown(c.strip()) for c in line.strip("|").split("|")]
            out.append("<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead><tbody>")
            in_table = True
            i += 2
            while i < len(lines) and "|" in lines[i]:
                cells = [render_inline_markdown(c.strip()) for c in lines[i].strip().strip("|").split("|")]
                out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
                i += 1
            continue

        if line.startswith("### "):
            close_ul(); close_table()
            out.append(f"<h3>{render_inline_markdown(line[4:])}</h3>")
        elif line.startswith("## "):
            close_ul(); close_table()
            out.append(f"<h2>{render_inline_markdown(line[3:])}</h2>")
        elif line.startswith("# "):
            close_ul(); close_table()
            out.append(f"<h2>{render_inline_markdown(line[2:])}</h2>")
        elif line.startswith(('- ', '* ')):
            close_table()
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{render_inline_markdown(line[2:])}</li>")
        else:
            close_ul(); close_table()
            safe = render_inline_markdown(line)
            out.append(f"<p>{safe}</p>")

        i += 1

    close_ul(); close_table()
    return "\n".join(out)


def paper_cards_html(papers):
    if not papers:
        return '<p class="no-data">No relevant papers retained after filtering.</p>'

    cards = []
    for i, p in enumerate(papers, 1):
        title = html.escape(p.get("title") or "Untitled")
        authors = html.escape(p.get("authors") or "Unknown authors")
        year = html.escape(str(p.get("year") or "Unknown year"))
        abstract = html.escape((p.get("abstract") or "No abstract available.")[:900])
        url = p.get("url")
        doi = p.get("doi")
        arxiv = p.get("arxiv")
        score = html.escape(str(p.get("relevance_score", "")))
        matched = html.escape(", ".join(p.get("matched_terms") or []))
        rescue = " · low-confidence rescue" if p.get("low_confidence_rescue") else ""

        link_bits = []
        if doi:
            link_bits.append(f'<a href="https://doi.org/{html.escape(doi)}" target="_blank">DOI:{html.escape(doi)}</a>')
        if arxiv:
            link_bits.append(f'<a href="https://arxiv.org/abs/{html.escape(arxiv)}" target="_blank">arXiv:{html.escape(arxiv)}</a>')
        if url:
            link_bits.append(f'<a href="{html.escape(url)}" target="_blank">source</a>')
        links = " &middot; ".join(link_bits) if link_bits else "no link"

        cards.append(f"""
        <div class="paper-card" id="paper-{i}">
            <div class="paper-title">Paper {i}: {title}</div>
            <div class="paper-meta">{authors} &middot; {year} &middot; relevance {score}{rescue} &middot; {links}</div>
            <div class="paper-meta">matched: {matched or 'none'}</div>
            <div class="paper-abstract">{abstract}</div>
        </div>
        """)
    return "\n".join(cards)


def build_html_report(question, model, profile_md, dft_answer, data_answer, final_report, papers, output_path):
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    q = html.escape(question)
    model_safe = html.escape(model)

    profile_html = simple_markdown_to_html(profile_md)
    dft_html = simple_markdown_to_html(dft_answer)
    data_html = simple_markdown_to_html(data_answer)
    final_html = simple_markdown_to_html(final_report)
    cards_html = paper_cards_html(papers)

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Scientific Report — {q}</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:#0d0f14; --surface:#13161d; --surface2:#1a1e28;
    --border:#2a2f3d; --accent:#00d4a0; --accent2:#4d8bff;
    --accent3:#ff6b6b; --text:#e2e8f0; --muted:#6b7a99;
    --mono:'IBM Plex Mono',monospace; --sans:'IBM Plex Sans',sans-serif;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:var(--sans);font-weight:300;line-height:1.7;min-height:100vh}}
  .header{{border-bottom:1px solid var(--border);padding:2.5rem 3rem;background:var(--surface);position:relative;overflow:hidden}}
  .header::before{{content:'';position:absolute;top:-60px;right:-60px;width:300px;height:300px;background:radial-gradient(circle,rgba(0,212,160,.08) 0%,transparent 70%);pointer-events:none}}
  .tag{{font-family:var(--mono);font-size:.7rem;color:var(--accent);letter-spacing:.2em;text-transform:uppercase;margin-bottom:.75rem}}
  .header h1{{font-size:1.4rem;font-weight:600;max-width:900px;line-height:1.4}}
  .meta{{margin-top:1rem;font-family:var(--mono);font-size:.72rem;color:var(--muted);display:flex;gap:2rem;flex-wrap:wrap}}
  .meta span{{color:var(--accent2)}}
  .container{{max-width:1100px;margin:0 auto;padding:2.5rem 3rem}}
  .section{{margin-bottom:3rem}}
  .sec-header{{display:flex;align-items:center;gap:1rem;margin-bottom:1.5rem;padding-bottom:.75rem;border-bottom:1px solid var(--border)}}
  .badge{{font-family:var(--mono);font-size:.65rem;padding:.25rem .6rem;border-radius:3px;letter-spacing:.1em;text-transform:uppercase;font-weight:600}}
  .b-dft{{background:rgba(77,139,255,.15);color:var(--accent2);border:1px solid rgba(77,139,255,.3)}}
  .b-lit{{background:rgba(0,212,160,.12);color:var(--accent);border:1px solid rgba(0,212,160,.3)}}
  .b-pap{{background:rgba(255,107,107,.12);color:var(--accent3);border:1px solid rgba(255,107,107,.3)}}
  .b-prof{{background:rgba(180,130,255,.12);color:#b482ff;border:1px solid rgba(180,130,255,.3)}}
  .sec-title{{font-size:1.05rem;font-weight:600}}
  .param-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:1rem;margin-bottom:1.5rem}}
  .param-card{{background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:1.1rem 1.3rem}}
  .param-label{{font-family:var(--mono);font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;margin-bottom:.4rem}}
  .param-value{{font-size:.9rem}}.param-value .hi{{font-family:var(--mono);color:var(--accent2);font-weight:600}}
  .box{{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:2rem;overflow-x:auto}}
  .box p{{margin-bottom:.6rem;font-size:.92rem}}
  .box h2,.box h3{{color:var(--accent2);margin:1.2rem 0 .4rem;font-size:.95rem}}
  .box strong{{color:var(--accent);font-weight:600}}
  .box li{{margin-left:1.2rem;font-size:.92rem;margin-bottom:.3rem}}
  table{{width:100%;border-collapse:collapse;margin:1rem 0;font-size:.82rem}}
  th,td{{border:1px solid var(--border);padding:.55rem;text-align:left;vertical-align:top}}
  th{{background:var(--surface2);color:var(--accent)}}
  .papers-grid{{display:flex;flex-direction:column;gap:1rem}}
  .paper-card{{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--accent3);border-radius:6px;padding:1.2rem 1.5rem;transition:border-color .2s}}
  .paper-card:hover{{border-left-color:var(--accent)}}
  .paper-title{{font-size:.92rem;font-weight:600;margin-bottom:.4rem;line-height:1.4}}
  .paper-meta{{font-family:var(--mono);font-size:.7rem;color:var(--muted);margin-bottom:.6rem}}
  .paper-meta a{{color:var(--accent2);text-decoration:none}}
  .paper-meta a:hover{{text-decoration:underline}}
  .paper-abstract{{font-size:.83rem;color:var(--muted);line-height:1.6}}
  .no-data{{font-family:var(--mono);font-size:.8rem;color:var(--accent3);padding:1rem;background:rgba(255,107,107,.05);border:1px dashed rgba(255,107,107,.3);border-radius:4px}}
  .footer{{border-top:1px solid var(--border);padding:1.5rem 3rem;font-family:var(--mono);font-size:.68rem;color:var(--muted);display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap}}
</style>
</head>
<body>
<div class="header">
  <div class="tag">Multi-Agent Scientific Workflow &mdash; Keyword Search + Dynamic Relevance</div>
  <h1>{q}</h1>
  <div class="meta">
    <div>Generated <span>{generated}</span></div>
    <div>Agents <span>Profile · SearchPlanner · DFTParams · Data · ReportWriter</span></div>
    <div>Model <span>{model_safe}</span></div>
  </div>
</div>

<div class="container">
  <div class="section">
    <div class="sec-header"><span class="badge b-lit">Final</span><span class="sec-title">Combined Answer</span></div>
    <div class="box">{final_html}</div>
  </div>

  <div class="section">
    <div class="sec-header"><span class="badge b-prof">Profile</span><span class="sec-title">Dynamic Keywords Used for Search/Scoring</span></div>
    <div class="box">{profile_html}</div>
  </div>

  <div class="section">
    <div class="sec-header"><span class="badge b-dft">DFT Parameters</span><span class="sec-title">Recommended Simulation Setup</span></div>
    <div class="param-grid">
      <div class="param-card"><div class="param-label">Query Style</div><div class="param-value"><span class="hi">Keyword</span> — no full-question API search</div></div>
      <div class="param-card"><div class="param-label">Relevance</div><div class="param-value"><span class="hi">Dynamic</span> — no hard-coded system aliases</div></div>
      <div class="param-card"><div class="param-label">Evidence</div><div class="param-value"><span class="hi">{len(papers)}</span> retained papers</div></div>
      <div class="param-card"><div class="param-label">Filtering</div><div class="param-value"><span class="hi">Loose</span> — top papers rescued if scores are weak</div></div>
    </div>
    <div class="box">{dft_html}</div>
  </div>

  <div class="section">
    <div class="sec-header"><span class="badge b-lit">Literature Data</span><span class="sec-title">Reference Values &amp; Literature Data</span></div>
    <div class="box">{data_html}</div>
  </div>

  <div class="section">
    <div class="sec-header"><span class="badge b-pap">Papers Found</span><span class="sec-title">Retrieved from arXiv / Semantic Scholar</span></div>
    <div class="papers-grid">{cards_html}</div>
  </div>
</div>

<div class="footer">
  <div>Generated by lookup_agent_working_keyword_dynamic.py</div>
  <div>Question-general pipeline · keyword search · dynamic relevance profile · HTML report</div>
</div>
</body>
</html>
"""
    Path(output_path).write_text(doc, encoding="utf-8")


# =========================
# MAIN PIPELINE
# =========================

def main():
    parser = argparse.ArgumentParser(description="Generic multi-agent scientific lookup with keyword search and HTML report output.")
    parser.add_argument("--question", "-q", default=DEFAULT_QUESTION, help="Scientific question to answer.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for outputs.")
    parser.add_argument("--per-query-limit", type=int, default=5, help="Papers to retrieve per query per source.")
    parser.add_argument("--max-queries", type=int, default=15, help="Maximum planned queries to run.")
    parser.add_argument("--min-score", type=int, default=1, help="Minimum loose relevance score.")
    parser.add_argument("--keep-top", type=int, default=30, help="Maximum retained papers after filtering.")
    args = parser.parse_args()

    question = args.question.strip()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = make_config(args.model)
    profile_agent, planner, dft_agent, data_agent, report_agent = build_agents(config)

    print("\n=== QUESTION ===")
    print(question)

    print("\n=== BUILDING DYNAMIC RELEVANCE PROFILE ===")
    profile = get_relevance_profile(profile_agent, question)
    profile_terms = terms_from_profile(profile)
    print(json.dumps(profile, indent=2, ensure_ascii=False))
    print("\nFlattened profile terms:")
    for t in profile_terms:
        print("-", t)

    profile_md = profile_to_markdown(profile, profile_terms)

    print("\n=== PLANNING KEYWORD SEARCH QUERIES ===")
    planner_msg = f"Research question:\n{question}\n\nDynamic relevance profile:\n{json.dumps(profile, indent=2, ensure_ascii=False)}\n"
    planned_text = ask_agent(planner, planner_msg)
    queries = parse_queries(planned_text, max_queries=args.max_queries)

    if not queries:
        raise RuntimeError("Search planner produced no usable keyword queries.")

    for q in queries:
        print("-", q)

    print("\n=== RUNNING BOUNDED SEARCHES ===")
    all_results = []
    for q in queries:
        print(f"Semantic Scholar: {q}")
        all_results.append(search_semantic_scholar(q, limit=args.per_query_limit))
        time.sleep(1)

    for q in queries[:3]:
        print(f"arXiv: {q}")
        all_results.append(search_arxiv(q, limit=args.per_query_limit))
        time.sleep(1)

    relevant, debug_candidates = filter_relevant_papers(
        all_results,
        profile_terms=profile_terms,
        planned_queries=queries,
        min_score=args.min_score,
        keep_top=args.keep_top,
    )
    evidence_md = papers_to_markdown(relevant)

    (output_dir / "raw_search_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    (output_dir / "debug_candidate_scores.json").write_text(json.dumps(debug_candidates, indent=2), encoding="utf-8")
    (output_dir / "filtered_papers.json").write_text(json.dumps(relevant, indent=2), encoding="utf-8")
    (output_dir / "retrieved_evidence.md").write_text(evidence_md, encoding="utf-8")
    (output_dir / "planned_queries.txt").write_text("\n".join(queries), encoding="utf-8")
    (output_dir / "relevance_profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "relevance_profile.md").write_text(profile_md, encoding="utf-8")

    print(f"\nRelevant papers retained: {len(relevant)}")
    print("\nTop candidate scores:")
    for p in debug_candidates[:10]:
        print(f"- {p.get('relevance_score'):>3} | {p.get('title')} | matched: {', '.join(p.get('matched_terms') or [])}")

    shared_msg = f"""
Question:
{question}

Dynamic relevance profile:
{json.dumps(profile, indent=2, ensure_ascii=False)}

Retrieved evidence:
{evidence_md}

Rules:
- Use only retrieved evidence plus clearly marked best-practice inference.
- Do not invent exact numerical values.
- If no direct value is found, say so clearly.
- If a paper is only a comparable-system paper, mark it as comparable rather than direct evidence.
"""

    print("\n=== DFT / SIMULATION AGENT ===")
    dft_answer = ask_agent(dft_agent, shared_msg)

    print("\n=== DATA / LITERATURE AGENT ===")
    data_answer = ask_agent(data_agent, shared_msg)

    report_msg = f"""
Question:
{question}

DFT/simulation answer:
{dft_answer}

Literature/data answer:
{data_answer}
"""

    print("\n=== REPORT WRITER AGENT ===")
    final_report = ask_agent(report_agent, report_msg)

    (output_dir / "dft_params_answer.md").write_text(dft_answer, encoding="utf-8")
    (output_dir / "data_literature_answer.md").write_text(data_answer, encoding="utf-8")
    (output_dir / "final_report.md").write_text(final_report, encoding="utf-8")

    html_path = output_dir / "report.html"
    build_html_report(
        question=question,
        model=args.model,
        profile_md=profile_md,
        dft_answer=dft_answer,
        data_answer=data_answer,
        final_report=final_report,
        papers=relevant,
        output_path=html_path,
    )

    print("\n=== DONE ===")
    print(f"HTML report: {html_path}")
    print(f"Markdown report: {output_dir / 'final_report.md'}")
    print(f"Evidence: {output_dir / 'retrieved_evidence.md'}")
    print(f"Profile: {output_dir / 'relevance_profile.md'}")
    print(f"Debug candidate scores: {output_dir / 'debug_candidate_scores.json'}")
    print(f"Raw search JSON: {output_dir / 'raw_search_results.json'}")


if __name__ == "__main__":
    main()
