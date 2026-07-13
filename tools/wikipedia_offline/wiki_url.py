"""Shared Wikipedia URL helpers (link mode, slug, parse, build)."""

from __future__ import annotations

import re
from typing import Any, Literal
from urllib.parse import quote, unquote, urlparse

LinkMode = Literal["wikiid", "title"]
DEFAULT_LINK_MODE: LinkMode = "wikiid"


def normalize_link_mode(mode: Any) -> LinkMode:
    return "title" if str(mode or "").strip().lower() == "title" else "wikiid"


def wiki_title_to_slug(title: str) -> str:
    """Convert a Wikipedia title to a URL slug: spaces to underscores, URL-encode the rest (keep underscores)."""
    return quote((title or "").strip().replace(" ", "_"), safe="_")


def build_wiki_url(
    doc_id: str,
    doc_title: str | None = None,
    chunk_no_1based: int | None = None,
    link_mode: Any = DEFAULT_LINK_MODE,
) -> str:
    """Shared Wiki URL builder; falls back to wikiid when title mode lacks a title."""
    ident = ""
    if normalize_link_mode(link_mode) == "title":
        ident = wiki_title_to_slug(doc_title or "")
    if not ident:
        ident = str(doc_id or "").strip()
    base = f"https://en.wikipedia.org/wiki/{ident}"
    if chunk_no_1based is not None:
        return f"{base}#chunk-{int(chunk_no_1based)}"
    return base


def make_wiki_chunk_url(
    doc_id: str,
    chunk_no_1based: int,
    doc_title: str | None = None,
    link_mode: Any = DEFAULT_LINK_MODE,
) -> str:
    return build_wiki_url(
        doc_id, doc_title, chunk_no_1based=int(chunk_no_1based), link_mode=link_mode
    )


def make_wiki_page_url(
    doc_id: str,
    doc_title: str | None = None,
    link_mode: Any = DEFAULT_LINK_MODE,
) -> str:
    return build_wiki_url(doc_id, doc_title, link_mode=link_mode)


def parse_wikipedia_url(url: str) -> tuple[str | None, int | None]:
    """Parse wiki page identifier and optional 1-based chunk number from a Wikipedia URL."""
    u = (url or "").strip()
    if not u:
        return None, None
    try:
        parsed = urlparse(u)
    except Exception:
        return None, None
    path = parsed.path or ""
    if not path.startswith("/wiki/"):
        return None, None
    raw_identifier = path[len("/wiki/") :]
    if not raw_identifier:
        return None, None
    identifier = unquote(raw_identifier).strip()
    if not identifier:
        return None, None

    frag = (parsed.fragment or "").strip()
    if not frag:
        return identifier, None
    m = re.fullmatch(r"chunk-(\d+)", frag)
    if not m:
        return identifier, None
    try:
        n = int(m.group(1))
    except Exception:
        return identifier, None
    if n <= 0:
        return identifier, None
    return identifier, n
