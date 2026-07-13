"""Normalize search-method aliases used by tools and MCP server."""

from __future__ import annotations

from typing import Any


def normalize_search_method(method: Any) -> str:
    m = str(method or "").strip().lower()
    if m in ("vec", "vector"):
        return "vector"
    if m in ("kw", "kws", "keyword", "keywords"):
        return "keywords"
    if m in ("rrf", "mix", ""):
        return "rrf"
    return m
