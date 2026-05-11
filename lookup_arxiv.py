import autogen
import os
import requests
import time
import json
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime

config_list = [{"model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")}]

# ─────────────────────────────────────────
# SEARCH CACHE (avoid repeat API hits)
# ─────────────────────────────────────────
_cache = {}

# ─────────────────────────────────────────
# TOOL: arXiv (primary — no rate limits)
# ─────────────────────────────────────────

def search_arxiv(query: str) -> str:
    """Search arXiv for scientific papers. No rate limits."""
    if query in _cache:
        print(f"  [Cache hit: {query[:50]}]")
        return _cache[query]

    time.sleep(3)  # be polite
    encoded = urllib.parse.quote(query)
    url = f"http://export.arxiv.org/api/query?search_query=all:{encoded}&max_results=6&sortBy=relevance"

    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)

        if not entries:
            result = "No papers found on arXiv for this query."
            _cache[query] = result
            return result

        result = ""
        for e in entries:
            title = (e.findtext("atom:title", "", ns) or "").replace("\n", " ").strip()
            authors = ", ".join(
                a.findtext("atom:name", "", ns)
                for a in e.findall("atom:author", ns)[:3]
            )
            summary = (e.findtext("atom:summary", "", ns) or "").replace("\n", " ").strip()[:400]
            arxiv_id = (e.findtext("atom:id", "", ns) or "").split("/abs/")[-1]
            published = (e.findtext("atom:published", "", ns) or "")[:4]

            result += f"\n---\n"
            result += f"Title: {title}\n"
            result += f"Authors: {authors}\n"
            result += f"Year: {published}\n"
            result += f"arXiv ID: {arxiv_id}\n"
            result += f"Abstract: {summary}\n"

        _cache[query] = result
        return result

    except Exception as e:
        return f"arXiv search failed: {str(e)}"


# ─────────────────────────────────────────
# TOOL: Semantic Scholar (fallback)
# ─────────────────────────────────────────

def search_semantic_scholar(query: str) -> str:
    """Fallback search on Semantic Scholar."""
    if query in _cache:
        return _cache[query]

    time.sleep(5)
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": query, "limit": 5,
              "fields": "title,authors,year,abstract,externalIds"}
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 429:
            return "Semantic Scholar rate limited. Use arXiv instead."
        r.raise_for_status()
        papers = r.json().get("data", [])
        if not papers:
            return "No papers found."
        result = ""
        for p in papers:
            authors = ", ".join(a["name"] for a in p.get("authors", [])[:3])
            abstract = (p.get("abstract") or "")[:400]
            doi = p.get("externalIds", {}).get("DOI", "No DOI")
            result += f"\n---\nTitle: {p['title']}\n"
            result += f"Authors: {authors}\nYear: {p.get('year','N/A')}\n"
            result += f"DOI: {doi}\nAbstract: {abstract}\n"
        _cache[query] = result
        return result
    except Exception as e:
        return f"Search failed: {str(e)}"


# ─────────────────────────────────────────
# TOOL SCHEMAS
# ─────────────────────────────────────────

arxiv_schema = {
    "type": "function",
    "function": {
        "name": "search_arxiv",
        "description": (
            "Search arXiv for scientific papers. Use this as your PRIMARY search tool. "
            "Call ONE query at a time. Good for physics, chemistry, materials science."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "Short search query, 3-6 words, e.g. 'water diffusion hBN interface'"}
            },
            "required": ["query"]
        }
    }
}

ss_schema = {
    "type": "function",
    "function": {
        "name": "search_semantic_scholar",
        "description": "Fallback search tool. Use only if arXiv returns irrelevant results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    }
}

# ─────────────────────────────────────────
# AGENTS
# ─────────────────────────────────────────

orchestrator = autogen.UserProxyAgent(
    name="Orchestrator",
    human_input_mode="NEVER",
    code_execution_config=False,
    max_consecutive_auto_reply=25,
    function_map={
        "search_arxiv": search_arxiv,
        "search_semantic_scholar": search_semantic_scholar,
    },
)

SEARCH_STRATEGY = """
SEARCH STRATEGY — follow this reasoning each time:

Step 1: Start BROAD — search the general phenomenon first
  e.g. "interfacial water dynamics surface"

Step 2: Add the specific material — try name variants
  hBN = "hexagonal boron nitride" = "h-BN" = "boron nitride"
  e.g. "water boron nitride molecular dynamics"

Step 3: Search comparable systems if hBN-specific papers are scarce
  graphene, MoS2, mica are structurally similar
  e.g. "water diffusion graphene surface AIMD"

Step 4: Search by method or known authors
  e.g. "Michaelides water 2D material"
  e.g. "AIMD water surface diffusion coefficient"

Step 5: If results are off-topic, change ONE word and retry
  "diffusion" → "dynamics" or "transport"
  "interface" → "surface" or "adsorption"
  "molecular dynamics" → "AIMD" or "ab initio MD"

RULES:
- Use search_arxiv as primary, search_semantic_scholar only as fallback
- ONE query at a time, never parallel
- Run at least 4 searches before writing your answer
- Never report a value not found in search results
- If the specific system has no data, report comparable systems and explain
"""

dft_agent = autogen.AssistantAgent(
    name="DFTParamsAgent",
    system_message=f"""
You are a DFT/DFT-MD parameter extraction assistant for the hBN/water interface.
You have access to search_arxiv (primary) and search_semantic_scholar (fallback).

{SEARCH_STRATEGY}

After searching, return your findings structured as:
1. Target system
2. DFT code type — identify explicitly:
   - Plane-wave + pseudopotential (e.g. VASP, Quantum ESPRESSO, CP2K plane-wave)
   - All-electron numeric (e.g. FHI-aims, WIEN2k)
   - Gaussian/atom-centred basis set (e.g. Gaussian, CP2K Gaussian, CRYSTAL)
   Report which code(s) the literature uses, and which is recommended for this system and why.
3. Recommended DFT functional(s) — cite paper if found, else flag as general practice
4. Dispersion correction
5. Pseudopotentials / basis set — specify type AND which atoms (e.g. PAW for B, N, O, H in VASP)
6. Plane-wave cutoff (eV) — or basis set quality if all-electron
7. k-point sampling
8. Supercell/slab setup (layers, vacuum gap)
9. Number of water molecules
10. Temperature and thermostat
11. MD timestep and trajectory length
12. How to compute the quantity asked (e.g. binding energy = E_total - E_slab - E_water)
13. Parameters still uncertain — flag clearly
14. Papers found (title, authors, year, arXiv ID or DOI) — real ones only

Reply TERMINATE when done.
""",
    llm_config={
        "config_list": config_list,
        "temperature": 0.2,
        "tools": [arxiv_schema, ss_schema],
    },
)

data_agent = autogen.AssistantAgent(
    name="DataAgent",
    system_message=f"""
You are a scientific literature assistant. Goal: find real data on the
diffusion coefficient of interfacial water on hBN at 300K.
You have access to search_arxiv (primary) and search_semantic_scholar (fallback).

{SEARCH_STRATEGY}

After searching, return:
1. Best estimated value/range with units — cite source, or state "not found in search"
2. Experimental data if found (real citations only)
3. DFT-MD or MD simulation data if found (real citations only)
4. Comparable systems with their D values: graphene, mica, MoS2
5. Conditions: temperature, method, water model for each data point
6. Confidence level: high / medium / low — explain why
7. All real papers found (title, authors, year, arXiv ID or DOI)
8. Gaps: what is genuinely missing from the literature

Never invent author names, titles, or IDs.
Only report what the search tools actually returned.
Reply TERMINATE when done.
""",
    llm_config={
        "config_list": config_list,
        "temperature": 0.2,
        "tools": [arxiv_schema, ss_schema],
    },
)

for agent in [dft_agent, data_agent]:
    autogen.register_function(
        search_arxiv,
        caller=agent, executor=orchestrator,
        name="search_arxiv",
        description="Search arXiv for scientific papers.",
    )
    autogen.register_function(
        search_semantic_scholar,
        caller=agent, executor=orchestrator,
        name="search_semantic_scholar",
        description="Fallback: search Semantic Scholar.",
    )

# ─────────────────────────────────────────
# HTML REPORT
# ─────────────────────────────────────────

def collect_papers(history):
    papers = []
    seen = set()
    for msg in history:
        content = msg.get("content", "") or ""
        if "Title:" not in content:
            continue
        for block in content.split("---"):
            if "Title:" not in block:
                continue
            lines = {}
            for line in block.strip().splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    lines[k.strip()] = v.strip()
            title = lines.get("Title", "")
            if title and title not in seen:
                seen.add(title)
                doi = lines.get("DOI", "") or lines.get("arXiv ID", "")
                papers.append({
                    "title": title,
                    "authors": lines.get("Authors", "N/A"),
                    "year": lines.get("Year", "N/A"),
                    "id": doi,
                    "abstract": lines.get("Abstract", "")
                })
    return papers

def get_final_answer(history):
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            content = (msg.get("content") or "").strip()
            if content and "TERMINATE" not in content and len(content) > 100:
                return content
    return ""

def make_paper_link(paper):
    pid = paper["id"]
    if not pid or pid == "No DOI":
        return pid
    if pid.startswith("10."):
        return f'<a href="https://doi.org/{pid}" target="_blank">{pid}</a>'
    return f'<a href="https://arxiv.org/abs/{pid}" target="_blank">arXiv:{pid}</a>'

import re
def md_to_html(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    lines = text.split("\n")
    out = []
    for line in lines:
        s = line.strip()
        if s.startswith("### "): out.append(f"<h3>{s[4:]}</h3>")
        elif s.startswith("## "): out.append(f"<h2>{s[3:]}</h2>")
        elif s.startswith("# "): out.append(f"<h1>{s[2:]}</h1>")
        elif s.startswith("- ") or s.startswith("* "):
            out.append(f"<li>{s[2:]}</li>")
        elif re.match(r'^\d+\.', s):
            out.append(f"<li>{s}</li>")
        elif s == "":
            out.append("<br>")
        else:
            out.append(f"<p>{line}</p>")
    return "\n".join(out)

def generate_html_report(question, dft_history, data_history, filename="report.html"):
    dft_answer = get_final_answer(dft_history)
    data_answer = get_final_answer(data_history)
    dft_papers = collect_papers(dft_history)
    data_papers = collect_papers(data_history)
    all_papers = list({p["title"]: p for p in dft_papers + data_papers}.values())

    papers_html = ""
    for p in all_papers:
        link = make_paper_link(p)
        papers_html += f"""
        <div class="paper-card">
            <div class="paper-title">{p['title']}</div>
            <div class="paper-meta">{p['authors']} &middot; {p['year']} &middot; {link}</div>
            <div class="paper-abstract">{p['abstract'][:350]}{'...' if len(p['abstract']) > 350 else ''}</div>
        </div>"""

    if not papers_html:
        papers_html = '<p class="no-data">No papers retrieved. Re-run when API is not rate-limited.</p>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Scientific Report — {question}</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:#0d0f14; --surface:#13161d; --surface2:#1a1e28;
    --border:#2a2f3d; --accent:#00d4a0; --accent2:#4d8bff;
    --accent3:#ff6b6b; --text:#e2e8f0; --muted:#6b7a99;
    --mono:'IBM Plex Mono',monospace; --sans:'IBM Plex Sans',sans-serif;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:var(--sans);
        font-weight:300;line-height:1.7;min-height:100vh}}

  .header{{border-bottom:1px solid var(--border);padding:2.5rem 3rem;
           background:var(--surface);position:relative;overflow:hidden}}
  .header::before{{content:'';position:absolute;top:-60px;right:-60px;
    width:300px;height:300px;
    background:radial-gradient(circle,rgba(0,212,160,.08) 0%,transparent 70%);
    pointer-events:none}}
  .tag{{font-family:var(--mono);font-size:.7rem;color:var(--accent);
        letter-spacing:.2em;text-transform:uppercase;margin-bottom:.75rem}}
  .header h1{{font-size:1.4rem;font-weight:600;max-width:800px;line-height:1.4}}
  .meta{{margin-top:1rem;font-family:var(--mono);font-size:.72rem;
         color:var(--muted);display:flex;gap:2rem}}
  .meta span{{color:var(--accent2)}}

  .container{{max-width:1100px;margin:0 auto;padding:2.5rem 3rem}}
  .section{{margin-bottom:3rem}}
  .sec-header{{display:flex;align-items:center;gap:1rem;margin-bottom:1.5rem;
               padding-bottom:.75rem;border-bottom:1px solid var(--border)}}
  .badge{{font-family:var(--mono);font-size:.65rem;padding:.25rem .6rem;
          border-radius:3px;letter-spacing:.1em;text-transform:uppercase;font-weight:600}}
  .b-dft{{background:rgba(77,139,255,.15);color:var(--accent2);border:1px solid rgba(77,139,255,.3)}}
  .b-lit{{background:rgba(0,212,160,.12);color:var(--accent);border:1px solid rgba(0,212,160,.3)}}
  .b-pap{{background:rgba(255,107,107,.12);color:var(--accent3);border:1px solid rgba(255,107,107,.3)}}
  .sec-title{{font-size:1.05rem;font-weight:600}}

  .param-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:1rem;margin-bottom:1.5rem}}
  .param-card{{background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:1.1rem 1.3rem}}
  .param-label{{font-family:var(--mono);font-size:.65rem;color:var(--muted);
                text-transform:uppercase;letter-spacing:.12em;margin-bottom:.4rem}}
  .param-value{{font-size:.9rem}}.param-value .hi{{font-family:var(--mono);color:var(--accent2);font-weight:600}}

  .box{{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:2rem}}
  .box p{{margin-bottom:.6rem;font-size:.92rem}}
  .box h2,.box h3{{color:var(--accent2);margin:1.2rem 0 .4rem;font-size:.95rem}}
  .box strong{{color:var(--accent);font-weight:600}}
  .box li{{margin-left:1.2rem;font-size:.92rem;margin-bottom:.3rem}}

  .papers-grid{{display:flex;flex-direction:column;gap:1rem}}
  .paper-card{{background:var(--surface);border:1px solid var(--border);
               border-left:3px solid var(--accent3);border-radius:6px;padding:1.2rem 1.5rem;
               transition:border-color .2s}}
  .paper-card:hover{{border-left-color:var(--accent)}}
  .paper-title{{font-size:.92rem;font-weight:600;margin-bottom:.4rem;line-height:1.4}}
  .paper-meta{{font-family:var(--mono);font-size:.7rem;color:var(--muted);margin-bottom:.6rem}}
  .paper-meta a{{color:var(--accent2);text-decoration:none}}
  .paper-meta a:hover{{text-decoration:underline}}
  .paper-abstract{{font-size:.83rem;color:var(--muted);line-height:1.6}}
  .no-data{{font-family:var(--mono);font-size:.8rem;color:var(--accent3);padding:1rem;
            background:rgba(255,107,107,.05);border:1px dashed rgba(255,107,107,.3);border-radius:4px}}

  .footer{{border-top:1px solid var(--border);padding:1.5rem 3rem;
           font-family:var(--mono);font-size:.68rem;color:var(--muted);
           display:flex;justify-content:space-between}}
</style>
</head>
<body>

<div class="header">
  <div class="tag">Multi-Agent Scientific Workflow &mdash; arXiv + Semantic Scholar</div>
  <h1>{question}</h1>
  <div class="meta">
    <div>Generated <span>{datetime.now().strftime('%Y-%m-%d %H:%M')}</span></div>
    <div>Agents <span>DFTParamsAgent · DataAgent</span></div>
    <div>Model <span>GPT-4o-mini</span></div>
  </div>
</div>

<div class="container">

  <div class="section">
    <div class="sec-header">
      <span class="badge b-dft">DFT Parameters</span>
      <span class="sec-title">Recommended Simulation Setup</span>
    </div>
    <div class="param-grid">
      <div class="param-card"><div class="param-label">DFT Code Type</div><div class="param-value"><span class="hi">Plane-wave</span> + pseudopotential (VASP / QE)</div></div>
      <div class="param-card"><div class="param-label">Functional</div><div class="param-value"><span class="hi">PBE</span> (GGA)</div></div>
      <div class="param-card"><div class="param-label">Dispersion</div><div class="param-value"><span class="hi">D3(BJ)</span> Grimme</div></div>
      <div class="param-card"><div class="param-label">Pseudopotentials</div><div class="param-value"><span class="hi">PAW</span> — B, N, O, H</div></div>
      <div class="param-card"><div class="param-label">Plane-wave Cutoff</div><div class="param-value"><span class="hi">500–600 eV</span></div></div>
      <div class="param-card"><div class="param-label">k-point Sampling</div><div class="param-value"><span class="hi">3×3×1</span> Γ-centered</div></div>
      <div class="param-card"><div class="param-label">Vacuum Layer</div><div class="param-value"><span class="hi">≥15 Å</span></div></div>
      <div class="param-card"><div class="param-label">Thermostat</div><div class="param-value"><span class="hi">Nosé–Hoover</span> @ 300 K</div></div>
      <div class="param-card"><div class="param-label">MD Timestep</div><div class="param-value"><span class="hi">0.5–1.0 fs</span></div></div>
      <div class="param-card"><div class="param-label">Trajectory</div><div class="param-value"><span class="hi">≥50 ps</span> production</div></div>
      <div class="param-card"><div class="param-label">Observable</div><div class="param-value"><span class="hi">From agent</span> see analysis below</div></div>
    </div>
    <div class="box">{md_to_html(dft_answer) if dft_answer else '<p class="no-data">No answer captured.</p>'}</div>
  </div>

  <div class="section">
    <div class="sec-header">
      <span class="badge b-lit">Literature Data</span>
      <span class="sec-title">Diffusion Coefficients &amp; Reference Values</span>
    </div>
    <div class="box">{md_to_html(data_answer) if data_answer else '<p class="no-data">No answer captured.</p>'}</div>
  </div>

  <div class="section">
    <div class="sec-header">
      <span class="badge b-pap">Papers Found</span>
      <span class="sec-title">Retrieved from arXiv / Semantic Scholar</span>
    </div>
    <div class="papers-grid">{papers_html}</div>
  </div>

</div>
<div class="footer">
  <span>{question[:60]}{'...' if len(question) > 60 else ''}</span>
  <span>Multi-Agent Hackathon · AG2 · arXiv</span>
</div>
</body></html>"""

    with open(filename, "w") as f:
        f.write(html)
    print(f"✓ HTML report → {filename}")

# ─────────────────────────────────────────
# SAVE MARKDOWN BACKUP
# ─────────────────────────────────────────

def save_markdown(result, filename):
    with open(filename, "w") as f:
        for msg in result.chat_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "") or ""
            if content.strip():
                f.write(f"## [{role.upper()}]\n{content}\n\n")
    print(f"✓ Markdown → {filename}")

# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────

# ═══════════════════════════════════════════
# ✏️  CHANGE YOUR QUESTION HERE — nothing else needs editing
QUESTION = "What is the binding energy of water to hBN?"
# ═══════════════════════════════════════════

print("\n--- DFT PARAMETER AGENT ---\n")
dft_result = orchestrator.initiate_chat(dft_agent, message=QUESTION)

print("\n[Waiting 60s between agents to avoid rate limiting...]\n")
time.sleep(60)

print("\n--- DATA / LITERATURE AGENT ---\n")
data_result = orchestrator.initiate_chat(data_agent, message=QUESTION)

save_markdown(dft_result, "dft_params_answer.md")
save_markdown(data_result, "data_literature_answer.md")

generate_html_report(
    question=QUESTION,
    dft_history=dft_result.chat_history,
    data_history=data_result.chat_history,
    filename="report.html"
)
