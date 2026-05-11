"""Web tools the script generator agent can call to consult code documentation.

- `search_docs(code, query)` runs a DuckDuckGo HTML search restricted to that code's
  docs domain (e.g. site:docs.lammps.org).
- `fetch_url(url)` downloads a page and returns plain text. URLs are restricted to
  the docs domains registered in `templates.TEMPLATES` to keep the agent on the map.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse

import requests

from .templates import TEMPLATES

USER_AGENT = "MultiAgentHackathon-script-generator/0.1"
TIMEOUT_S = 10
MAX_CHARS = 8000  # truncate fetched pages to keep token usage bounded


# ---------------------------------------------------------------------------
# HTML -> text (stdlib only)
# ---------------------------------------------------------------------------

_SKIP_TAGS = {"script", "style", "noscript", "svg", "head"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in {"p", "br", "li", "tr", "div", "h1", "h2", "h3", "h4", "h5", "h6", "pre"}:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in {"p", "li", "tr", "div", "h1", "h2", "h3", "h4", "h5", "h6", "pre"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._chunks.append(data)

    def text(self) -> str:
        joined = "".join(self._chunks)
        # collapse runs of blank lines / whitespace
        joined = re.sub(r"[ \t]+", " ", joined)
        joined = re.sub(r"\n{3,}", "\n\n", joined)
        return joined.strip()


def _html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return parser.text()


# ---------------------------------------------------------------------------
# fetch_url
# ---------------------------------------------------------------------------

def _allowed_domains() -> set[str]:
    return {t.docs_domain for t in TEMPLATES.values() if t.docs_domain}


def fetch_url(url: str) -> str:
    """Download `url` (must be on a registered docs domain) and return plain text.

    Truncated to MAX_CHARS to keep responses bounded.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Refusing non-http(s) URL: {url}")

    allowed = _allowed_domains()
    host = parsed.netloc.lower()
    if not any(host == d or host.endswith("." + d) for d in allowed):
        raise ValueError(
            f"Domain {host!r} is not in the docs allowlist {sorted(allowed)}. "
            "Use search_docs(code, query) first to find a permitted URL."
        )

    resp = requests.get(
        url, timeout=TIMEOUT_S, headers={"User-Agent": USER_AGENT}
    )
    resp.raise_for_status()
    text = _html_to_text(resp.text)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + f"\n\n... [truncated at {MAX_CHARS} chars]"
    return text


# ---------------------------------------------------------------------------
# search_docs (DuckDuckGo HTML)
# ---------------------------------------------------------------------------

_DDG_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
    r'.*?<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
    re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(s: str) -> str:
    return re.sub(r"\s+", " ", _TAG_RE.sub("", s)).strip()


def _unwrap_ddg_redirect(href: str) -> str:
    """DuckDuckGo HTML wraps result links in /l/?uddg=<encoded>. Unwrap them."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if parsed.path.startswith("/l/"):
        qs = parse_qs(parsed.query)
        target = qs.get("uddg", [""])[0]
        if target:
            return unquote(target)
    return href


def search_docs(code: str, query: str, max_results: int = 5) -> list[dict]:
    """Search the docs of `code` for `query`. Returns up to `max_results` hits.

    Each hit is `{"title": str, "url": str, "snippet": str}`. The agent should
    pick the most relevant URL and pass it to `fetch_url` to read the page.
    """
    code_key = code.lower()
    if code_key not in TEMPLATES:
        raise ValueError(
            f"Unknown code {code!r}. Registered: {sorted(TEMPLATES.keys())}"
        )
    domain = TEMPLATES[code_key].docs_domain
    if not domain:
        raise ValueError(f"No docs_domain registered for {code!r}")

    full_query = f"site:{domain} {query}"
    resp = requests.post(
        "https://html.duckduckgo.com/html/",
        data={"q": full_query},
        timeout=TIMEOUT_S,
        headers={"User-Agent": USER_AGENT},
    )
    resp.raise_for_status()

    hits: list[dict] = []
    for match in _DDG_RESULT_RE.finditer(resp.text):
        href, title, snippet = match.groups()
        url = _unwrap_ddg_redirect(href)
        # filter to the requested domain just in case
        host = urlparse(url).netloc.lower()
        if not (host == domain or host.endswith("." + domain)):
            continue
        hits.append(
            {
                "title": _strip_tags(title),
                "url": url,
                "snippet": _strip_tags(snippet),
            }
        )
        if len(hits) >= max_results:
            break
    return hits
