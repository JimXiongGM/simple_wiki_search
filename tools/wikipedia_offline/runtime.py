"""Process-local Wikipedia tool runtime (init + search/open/random + RRF helpers).

This is the business logic the HTTP server should call. Keep FastAPI / pools
out of this module.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass

from loguru import logger

from settings import (
    DEFAULT_EMBEDDING_API_KEY,
    DEFAULT_EMBEDDING_API_URL,
    DEFAULT_EMBEDDING_MODEL,
    INDEX_PATHS,
)
from tools.wikipedia_offline.search_method import normalize_search_method
from tools.wikipedia_offline.types import ChunkInfo, KeywordHit
from tools.wikipedia_offline.wiki_url import (
    make_wiki_chunk_url,
    make_wiki_page_url,
    normalize_link_mode,
)


@dataclass
class WikiRuntimeConfig:
    text_index_dir: str = INDEX_PATHS.text_index_dir
    vector_index_dir: str = INDEX_PATHS.vector_index_dir
    embedding_api_url: str = DEFAULT_EMBEDDING_API_URL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    # Qwen3-Embedding 查询侧 instruction，与建索引时一致，不对外暴露配置。
    embedding_task: str = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )
    embedding_api_key: str = DEFAULT_EMBEDDING_API_KEY
    embedding_no_proxy: bool = True
    embedding_dimensions: int | None = None
    lazy_load_faiss: bool = False
    link_mode: str = "wikiid"


_INIT_LOCK = threading.Lock()
_TEXT_INITTED = False
_RRF_INITTED = False


def ensure_text_inited(cfg: WikiRuntimeConfig) -> None:
    global _TEXT_INITTED
    if _TEXT_INITTED:
        return
    with _INIT_LOCK:
        if _TEXT_INITTED:
            return
        from tools.wikipedia_offline.search_kws import init_client

        init_client(index_dir=str(cfg.text_index_dir).strip())
        _TEXT_INITTED = True


def ensure_rrf_inited(cfg: WikiRuntimeConfig) -> None:
    global _RRF_INITTED
    if _RRF_INITTED:
        return
    with _INIT_LOCK:
        if _RRF_INITTED:
            return
        try:
            from tools.wikipedia_offline.search_rrf import init_client
        except ModuleNotFoundError as e:
            if getattr(e, "name", "") == "faiss":
                raise RuntimeError(
                    "faiss is not installed (required for method=rrf/vector)."
                ) from e
            raise

        init_client(
            text_index_dir=str(cfg.text_index_dir).strip(),
            vector_index_dir=str(cfg.vector_index_dir).strip(),
            embedding_api_url=str(cfg.embedding_api_url).strip(),
            embedding_model=str(cfg.embedding_model).strip(),
            embedding_task=str(cfg.embedding_task or "").strip(),
            embedding_api_key=str(cfg.embedding_api_key or "simple_wiki").strip(),
            embedding_no_proxy=bool(cfg.embedding_no_proxy),
            embedding_dimensions=cfg.embedding_dimensions,
        )
        _RRF_INITTED = True


def text_worker_initializer(cfg: WikiRuntimeConfig) -> None:
    ensure_text_inited(cfg)
    logger.info(
        "Text worker ready pid={} text_index_dir={}",
        os.getpid(),
        cfg.text_index_dir,
    )


def vector_runtime_preload(cfg: WikiRuntimeConfig) -> None:
    """Load shared in-process FAISS (and text hydrate deps) once for thread-pool use."""
    ensure_text_inited(cfg)
    if not bool(cfg.lazy_load_faiss):
        ensure_rrf_inited(cfg)
    logger.info(
        "Vector runtime ready pid={} vector_index_dir={} lazy_load_faiss={}",
        os.getpid(),
        cfg.vector_index_dir,
        cfg.lazy_load_faiss,
    )


def worker_ping(role: str) -> str:
    return f"{role}:{os.getpid()}"


def open_url(url: str, cfg: WikiRuntimeConfig | None = None) -> str:
    cfg = cfg or WikiRuntimeConfig()
    ensure_text_inited(cfg)
    from tools.wikipedia_offline.search_kws import OpenURL

    return OpenURL((url or "").strip())


def random_page_or_chunk(
    *,
    mode: str,
    seed: int | None,
    cfg: WikiRuntimeConfig,
) -> str:
    ensure_text_inited(cfg)
    from tools.wikipedia_offline.search_kws import get_client

    client = get_client()
    mode_n = str(mode or "").strip().lower()
    if mode_n not in ("chunk", "page"):
        raise ValueError('mode must be "chunk" or "page"')

    if mode_n == "chunk":
        rows = client.random_chunks(top_k=1, seed=seed)
        if not rows:
            return "No results found."
        row = rows[0]
        doc_id = str(row.get("doc_id") or "").strip()
        chunk_no = int(row.get("chunk_no_1based") or 1)
        if not doc_id:
            return "No results found."
        return open_url(make_wiki_chunk_url(doc_id, chunk_no), cfg)

    rows_p = client.random_pages(top_k=1, seed=seed)
    if not rows_p:
        return "No results found."
    doc_id_p = str(rows_p[0].get("doc_id") or "").strip()
    if not doc_id_p:
        return "No results found."
    return open_url(make_wiki_page_url(doc_id_p), cfg)


def search_text_rrf_candidates(
    *,
    query: str,
    top_k: int,
    max_chars_per_result: int,
    cfg: WikiRuntimeConfig,
    fetch_multiplier: int = 3,
) -> list[KeywordHit]:
    q = (query or "").strip()
    if not q:
        raise ValueError("query is empty")
    ensure_text_inited(cfg)
    from tools.wikipedia_offline.search_kws import get_client

    client = get_client()
    fetch_k = max(1, int(top_k)) * max(1, int(fetch_multiplier))
    return client.search(
        q,
        top_k=fetch_k,
        dedupe=True,
        snippet_max_length=int(max_chars_per_result),
    )


def search_vector_candidates(
    *,
    query: str,
    top_k: int,
    cfg: WikiRuntimeConfig,
    fetch_multiplier: int = 3,
) -> list[tuple[str, float]]:
    q = (query or "").strip()
    if not q:
        raise ValueError("query is empty")
    ensure_rrf_inited(cfg)
    from tools.wikipedia_offline.search_rrf import get_client as get_rrf_client

    client = get_rrf_client()
    fetch_k = max(1, int(top_k)) * max(1, int(fetch_multiplier))
    return client.vector_tool.search(q, top_k=fetch_k)


def hydrate_chunk_infos(
    *,
    chunk_ids: list[str],
    cfg: WikiRuntimeConfig,
) -> dict[str, ChunkInfo]:
    ensure_text_inited(cfg)
    from tools.wikipedia_offline.search_kws import get_client

    client = get_client()
    out: dict[str, ChunkInfo] = {}
    for chunk_id in chunk_ids:
        parts = str(chunk_id or "").rsplit("_", 1)
        if len(parts) != 2:
            continue
        doc_id, chunk_idx_str = parts
        try:
            chunk_index = int(chunk_idx_str)
        except ValueError:
            continue
        doc_info = client.get_doc_by_id(doc_id)
        if not doc_info:
            continue
        chunks = client.get_all_chunks_for_doc(doc_id)
        if not chunks or chunk_index < 0 or chunk_index >= len(chunks):
            continue
        chunk = chunks[chunk_index]
        text = chunk.get("text")
        chunk_id_real = chunk.get("chunk_id")
        if not isinstance(text, str) or not isinstance(chunk_id_real, str):
            continue
        out[str(chunk_id)] = ChunkInfo(
            doc_id=doc_id,
            doc_title=doc_info["doc_title"],
            chunk_id=chunk_id_real,
            chunk_index=chunk_index,
            total_chunks=len(chunks),
            text=text,
        )
    return out


def build_rrf_markdown(
    *,
    top_k: int,
    max_chars_per_result: int,
    text_results: list[KeywordHit],
    vector_results: list[tuple[str, float]],
    chunk_infos: dict[str, ChunkInfo],
    link_mode: str = "wikiid",
    ranked_chunk_ids: list[str] | None = None,
) -> str:
    from tools.wikipedia_offline.search_rrf import (
        build_rrf_rank_maps,
        fuse_rrf_ranked_chunk_ids,
    )
    from tools.wikipedia_offline.search_vector import make_vector_snippet

    link_mode = normalize_link_mode(link_mode)
    text_ranks, vector_ranks, text_snippets = build_rrf_rank_maps(
        text_results, vector_results
    )
    if not text_ranks and not vector_ranks:
        return "No results found."

    if ranked_chunk_ids is None:
        ranked_chunk_ids = [
            chunk_id
            for chunk_id, _score in fuse_rrf_ranked_chunk_ids(text_ranks, vector_ranks)
        ]

    lines: list[str] = []
    rank_idx = 0
    for chunk_id in ranked_chunk_ids:
        info = chunk_infos.get(chunk_id)
        if not info:
            continue
        rank_idx += 1
        if rank_idx > int(top_k):
            break
        doc_title = str(info["doc_title"])
        doc_id = str(info["doc_id"])
        chunk_no = int(info["chunk_index"]) + 1
        url = make_wiki_chunk_url(doc_id, chunk_no, doc_title, link_mode)
        snippet = text_snippets.get(chunk_id) or make_vector_snippet(
            str(info.get("text") or ""), max_length=int(max_chars_per_result)
        )
        lines.append(f"{rank_idx}. [{doc_title}]({url})")
        if snippet:
            lines.append(f"   - {snippet}")
        lines.append("")

    return "\n".join(lines).strip() or "No results found."


def _format_keyword_hits_md(
    *,
    results: list[KeywordHit],
    link_mode: str,
) -> str:
    from tools.wikipedia_offline.search_kws import get_client

    client = get_client()
    link_mode = normalize_link_mode(link_mode)
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        doc_id = str(r.get("doc_id") or "").strip()
        chunk_id = str(r.get("chunk_id") or "").strip()
        snippet = str(r.get("snippet") or "")
        if not doc_id or not chunk_id:
            continue
        try:
            idx0 = int(chunk_id.split("_")[-1])
        except Exception:
            idx0 = 0
        doc_title = ""
        try:
            info = client.get_doc_by_id(doc_id)
            if isinstance(info, dict):
                doc_title = str(info.get("doc_title") or "").strip()
        except Exception:
            doc_title = ""
        if not doc_title:
            doc_title = doc_id
        url = make_wiki_chunk_url(doc_id, idx0 + 1, doc_title, link_mode)
        lines.append(f"{i}. [{doc_title}]({url})")
        if snippet.strip():
            lines.append(f"   - {snippet}")
        lines.append("")
    return "\n".join(lines).strip() or "No results found."


def search_markdown(
    *,
    query: str,
    top_k: int,
    only_title: bool,
    method: str,
    max_chars_per_result: int,
    cfg: WikiRuntimeConfig,
    link_mode: str | None = None,
) -> str:
    """Unified search markdown used by the HTTP server (non-parallel paths)."""
    q = (query or "").strip()
    m = normalize_search_method(method)
    if not q:
        raise ValueError("query is empty")
    k = max(1, int(top_k))
    mode = normalize_link_mode(link_mode or cfg.link_mode)
    ensure_text_inited(cfg)

    if bool(only_title):
        from tools.wikipedia_offline.search_kws import get_client

        client = get_client()
        hits = client.search_title(q, top_k=k)
        if not hits:
            return "No results found."
        lines: list[str] = []
        for i, h in enumerate(hits, 1):
            doc_id = str(h.get("doc_id") or "").strip()
            doc_title = str(h.get("doc_title") or "").strip()
            if not doc_id or not doc_title:
                continue
            url = make_wiki_chunk_url(doc_id, 1, doc_title, mode)
            snippet = ""
            try:
                chunks = client.get_all_chunks_for_doc(doc_id)
                if chunks:
                    raw = str(chunks[0].get("text") or "")[:500]
                    snippet = re.sub(
                        r"\s+", " ", raw.strip().replace("\n", " ").replace("█", " ")
                    ).strip()
            except Exception:
                snippet = ""
            lines.append(f"{i}. [{doc_title}]({url})")
            if snippet:
                lines.append(f"   - {snippet}")
            lines.append("")
        return "\n".join(lines).strip() or "No results found."

    if m == "keywords":
        from tools.wikipedia_offline.search_kws import get_client

        client = get_client()
        results = client.search(
            q,
            top_k=k,
            dedupe=True,
            snippet_max_length=int(max_chars_per_result),
        )
        return _format_keyword_hits_md(results=results, link_mode=mode)

    ensure_rrf_inited(cfg)
    from tools.wikipedia_offline.search_rrf import Search as SearchRRF

    return SearchRRF(
        query=q,
        top_k=k,
        only_title=False,
        method=m,  # type: ignore[arg-type]
        max_chars_per_result=int(max_chars_per_result),
        link_mode=mode,
    )
