"""Shared TypedDict shapes for offline Wikipedia retrieval payloads."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class KeywordHit(TypedDict):
    """One keyword/BM25 chunk hit from ``WikiSearchClient.search``."""

    title_with_id: str
    doc_id: str
    chunk_id: str
    score: float
    snippet: str


class TitleHit(TypedDict):
    """One title-index hit from ``WikiSearchClient.search_title``."""

    doc_id: str
    doc_title: str
    score: float


class ChunkRecord(TypedDict):
    """One stored chunk row from ``get_all_chunks_for_doc``."""

    chunk_id: str
    text: str


class DocInfo(TypedDict):
    """Document metadata from ``get_doc_by_id``."""

    doc_id: str
    doc_title: str
    toc: list[dict[str, Any]]


class ChunkInfo(TypedDict):
    """Hydrated chunk payload used by RRF / open_url paths."""

    doc_id: str
    doc_title: str
    chunk_id: str
    chunk_index: int
    total_chunks: int
    text: str


class RrfHit(TypedDict):
    """One fused hit from ``RRFSearchClient.search``."""

    title_with_id: str
    doc_id: str
    doc_title: str
    chunk_id: str
    chunk_index: int
    total_chunks: int
    text: str
    snippet: str
    rrf_score: float
    text_rank: int | None
    vector_rank: int | None
    source: Literal["both", "keywords", "vector"]


class RandomChunkHit(TypedDict):
    doc_id: str
    doc_title: str
    chunk_no_1based: int
    snippet: str


class RandomPageHit(TypedDict):
    doc_id: str
    doc_title: str
    snippet: str
