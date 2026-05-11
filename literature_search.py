import time
import urllib.parse
import xml.etree.ElementTree as ET

import requests

_cache: dict[str, str] = {}


def search_arxiv(query: str) -> str:
    """Search arXiv for scientific papers. No rate limits."""
    if query in _cache:
        return _cache[query]

    time.sleep(3)
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


def search_semantic_scholar(query: str) -> str:
    """Fallback search on Semantic Scholar."""
    if query in _cache:
        return _cache[query]

    time.sleep(5)
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": 5,
        "fields": "title,authors,year,abstract,externalIds",
    }
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
            result += f"Authors: {authors}\nYear: {p.get('year', 'N/A')}\n"
            result += f"DOI: {doi}\nAbstract: {abstract}\n"
        _cache[query] = result
        return result
    except Exception as e:
        return f"Search failed: {str(e)}"


ARXIV_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_arxiv",
        "description": (
            "Search arXiv for scientific papers. Use as PRIMARY search tool. "
            "Good for physics, chemistry, materials science, ML."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Short search query, 3-6 words",
                }
            },
            "required": ["query"],
        },
    },
}

SEMANTIC_SCHOLAR_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_semantic_scholar",
        "description": "Fallback search tool. Use only if arXiv returns irrelevant results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"],
        },
    },
}
