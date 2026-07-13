"""
RRF (Reciprocal Rank Fusion) Search combining text search and vector search.

Combines results from:
- text_search.py: keywords text search using Tantivy
- vector_search.py: Vector search using FAISS HNSW
"""

from __future__ import annotations

from typing import Any, Literal

from settings import (
    DEFAULT_EMBEDDING_API_KEY,
    DEFAULT_EMBEDDING_API_URL,
    DEFAULT_EMBEDDING_MODEL,
    INDEX_PATHS,
)
from tools.wikipedia_offline.search_kws import OpenURL as open_wikipedia_url
from tools.wikipedia_offline.search_kws import WikiSearchClient, make_title_with_id
from tools.wikipedia_offline.search_vector import VectorSearchTool, make_vector_snippet
from tools.wikipedia_offline.types import ChunkInfo, RrfHit
from tools.wikipedia_offline.wiki_url import (
    DEFAULT_LINK_MODE,
    make_wiki_chunk_url,
    normalize_link_mode,
)


def compute_rrf_score(ranks: list[int] | list[float], k: int = 60) -> float:
    """
    Compute RRF score from multiple rank positions.

    RRF(d) = sum(1 / (k + rank_i)) for each ranking list

    Args:
        ranks: List of rank positions (1-indexed); may be weight-adjusted
        k: RRF constant (default 60)

    Returns:
        RRF score
    """
    return sum(1.0 / (k + r) for r in ranks)


def build_rrf_rank_maps(
    text_results: list[dict[str, Any]] | list[Any],
    vector_results: list[tuple[str, float]],
) -> tuple[dict[str, int], dict[str, int], dict[str, str]]:
    """Build 1-indexed rank maps and text snippets from raw search hits."""
    text_ranks: dict[str, int] = {}
    text_snippets: dict[str, str] = {}
    for rank, item in enumerate(text_results, 1):
        chunk_id = str(item.get("chunk_id") or "").strip()
        if not chunk_id or chunk_id in text_ranks:
            continue
        text_ranks[chunk_id] = rank
        text_snippets[chunk_id] = str(item.get("snippet") or "")

    vector_ranks: dict[str, int] = {}
    for rank, (chunk_id, _score) in enumerate(vector_results, 1):
        chunk_id_s = str(chunk_id or "").strip()
        if not chunk_id_s or chunk_id_s in vector_ranks:
            continue
        vector_ranks[chunk_id_s] = rank
    return text_ranks, vector_ranks, text_snippets


def fuse_rrf_ranked_chunk_ids(
    text_ranks: dict[str, int],
    vector_ranks: dict[str, int],
    *,
    text_weight: float = 1.0,
    vector_weight: float = 1.0,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse text/vector ranks into (chunk_id, rrf_score) sorted descending.

    Shared by the in-process RRF client and the HTTP parallel search path.
    """
    tw = float(text_weight) if float(text_weight) > 0 else 1.0
    vw = float(vector_weight) if float(vector_weight) > 0 else 1.0
    all_chunk_ids = set(text_ranks) | set(vector_ranks)
    scored: list[tuple[str, float]] = []
    for chunk_id in all_chunk_ids:
        ranks: list[float] = []
        if chunk_id in text_ranks:
            ranks.append(float(text_ranks[chunk_id]) * (1.0 / tw))
        if chunk_id in vector_ranks:
            ranks.append(float(vector_ranks[chunk_id]) * (1.0 / vw))
        if not ranks:
            continue
        scored.append((chunk_id, compute_rrf_score(ranks, k=rrf_k)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


class RRFSearchClient:
    """
    RRF fusion search combining text (keywords) and vector search results.
    """

    def __init__(
        self,
        text_index_dir: str = INDEX_PATHS.text_index_dir,
        vector_index_dir: str = INDEX_PATHS.vector_index_dir,
        embedding_api_url: str = DEFAULT_EMBEDDING_API_URL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_task: str = "",
        embedding_api_key: str = DEFAULT_EMBEDDING_API_KEY,
        embedding_no_proxy: bool = True,
        embedding_dimensions: int | None = None,
    ):
        """
        Initialize RRF search client.

        Args:
            text_index_dir: Path to Tantivy text index
            vector_index_dir: Path to FAISS vector index
            embedding_api_url: URL for embedding API
            embedding_model: Model name for embeddings
            embedding_api_key: Bearer token for the embedding API
            embedding_no_proxy: True for direct connection; False to use env proxy
            embedding_dimensions: Optional output dim (e.g. DashScope text-embedding-v4)
        """
        self.text_client = WikiSearchClient(text_index_dir)
        self.vector_tool = VectorSearchTool(
            vector_index_dir,
            api_url=embedding_api_url,
            model_name=embedding_model,
            task=str(embedding_task or "").strip(),
            api_key=embedding_api_key,
            no_proxy=embedding_no_proxy,
            dimensions=embedding_dimensions,
        )

    def _get_chunk_info(self, chunk_id: str) -> ChunkInfo | None:
        """
        Get chunk information by chunk_id.

        Args:
            chunk_id: Chunk ID (e.g., "12345_3")

        Returns:
            ChunkInfo with doc_id, doc_title, chunk_index, total_chunks, text
        """
        # Extract doc_id from chunk_id
        parts = chunk_id.rsplit("_", 1)
        if len(parts) != 2:
            return None
        doc_id, chunk_idx_str = parts

        # Get document info
        doc_info = self.text_client.get_doc_by_id(doc_id)
        if not doc_info:
            return None

        # Get all chunks for this document
        all_chunks = self.text_client.get_all_chunks_for_doc(doc_id)
        total_chunks = len(all_chunks)

        # Find the specific chunk
        chunk_text = None
        chunk_index = int(chunk_idx_str)
        for c in all_chunks:
            if c["chunk_id"] == chunk_id:
                chunk_text = c["text"]
                break

        if chunk_text is None:
            return None

        return ChunkInfo(
            doc_id=doc_id,
            doc_title=doc_info["doc_title"],
            chunk_id=chunk_id,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            text=chunk_text,
        )

    def get_doc_id_by_title(self, title: str) -> str | None:
        q = (title or "").strip()
        if not q:
            return None
        hits = self.text_client.search_title(q, top_k=1)
        if not hits:
            return None
        doc_id = hits[0].get("doc_id")
        if not isinstance(doc_id, str) or not doc_id.strip():
            return None
        return doc_id.strip()

    def get_chunk_text_by_doc(
        self, doc_id: str, chunk_no_1based: int
    ) -> ChunkInfo | None:
        try:
            idx0 = int(chunk_no_1based) - 1
        except Exception:
            return None
        if idx0 < 0:
            return None

        doc_info = self.text_client.get_doc_by_id(doc_id)
        if not doc_info:
            return None

        chunks = self.text_client.get_all_chunks_for_doc(doc_id)
        if not chunks:
            return None
        if idx0 >= len(chunks):
            return None

        chunk = chunks[idx0]
        text = chunk.get("text")
        chunk_id = chunk.get("chunk_id")
        if not isinstance(text, str) or not isinstance(chunk_id, str):
            return None
        return ChunkInfo(
            doc_id=doc_id,
            doc_title=doc_info["doc_title"],
            chunk_id=chunk_id,
            chunk_index=idx0,
            total_chunks=len(chunks),
            text=text,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        text_weight: float = 1.0,
        vector_weight: float = 1.0,
        rrf_k: int = 60,
        fetch_multiplier: int = 3,
        method: str = "rrf",  # "rrf", "keywords", or "vector"
        snippet_max_length: int = 3000,
        verbose: bool = False,
    ) -> list[RrfHit]:
        """
        Perform RRF fusion search.

        Args:
            query: Search query
            top_k: Number of final results
            text_weight: Weight for text search scores
            vector_weight: Weight for vector search scores
            rrf_k: RRF constant
            fetch_multiplier: Fetch top_k * multiplier from each source
            method: Search method - "rrf" (RRF), "keywords" (text only), "vector" (vector only)
            snippet_max_length: Max snippet length for vector-derived snippets
            verbose: Print timing information

        Returns:
            List of RrfHit dicts with title_with_id, chunk_index, total_chunks, text, rrf_score
        """
        import time

        fetch_k = top_k * fetch_multiplier

        # Get text search results
        t0 = time.time()
        text_results = (
            self.text_client.search(
                query,
                top_k=fetch_k,
                snippet_max_length=int(snippet_max_length),
            )
            if method in ("rrf", "mix", "keywords")
            else []
        )
        t_text = time.time() - t0

        # Get vector search results
        t0 = time.time()
        vector_results = (
            self.vector_tool.search(query, top_k=fetch_k)
            if method in ("rrf", "mix", "vector")
            else []
        )
        t_vector = time.time() - t0

        # Build rank maps, fuse with shared RRF, then hydrate top hits.
        t0 = time.time()
        text_ranks, vector_ranks, text_snippets = build_rrf_rank_maps(
            text_results, vector_results
        )
        ranked = fuse_rrf_ranked_chunk_ids(
            text_ranks,
            vector_ranks,
            text_weight=text_weight,
            vector_weight=vector_weight,
            rrf_k=rrf_k,
        )
        rrf_scores = {chunk_id: score for chunk_id, score in ranked}
        t_rrf = time.time() - t0

        # Build final results
        t0 = time.time()
        results: list[RrfHit] = []
        for chunk_id, _score in ranked:
            if len(results) >= top_k:
                break
            chunk_info = self._get_chunk_info(chunk_id)
            if not chunk_info:
                continue
            # Determine source: both, keywords, or vector
            in_text = chunk_id in text_ranks
            in_vec = chunk_id in vector_ranks
            source: Literal["both", "keywords", "vector"] = (
                "both"
                if (in_text and in_vec)
                else ("keywords" if in_text else "vector")
            )
            if chunk_id in text_snippets and text_snippets[chunk_id].strip():
                snippet = text_snippets[chunk_id]
            else:
                snippet = make_vector_snippet(
                    chunk_info["text"], max_length=int(snippet_max_length)
                )

            results.append(
                RrfHit(
                    title_with_id=make_title_with_id(
                        chunk_info["doc_title"], chunk_info["doc_id"]
                    ),
                    doc_id=chunk_info["doc_id"],
                    doc_title=chunk_info["doc_title"],
                    chunk_id=chunk_id,
                    chunk_index=chunk_info["chunk_index"],
                    total_chunks=chunk_info["total_chunks"],
                    text=chunk_info["text"],
                    snippet=snippet,
                    rrf_score=rrf_scores[chunk_id],
                    text_rank=text_ranks.get(chunk_id),
                    vector_rank=vector_ranks.get(chunk_id),
                    source=source,
                )
            )
        t_build = time.time() - t0

        if verbose:
            print(
                f"  [Timing] text_search: {t_text*1000:.1f}ms, vector_search: {t_vector*1000:.1f}ms, "
                f"rrf_compute: {t_rrf*1000:.1f}ms, build_results: {t_build*1000:.1f}ms"
            )

        return results


# Global client
_client: RRFSearchClient | None = None


def init_client(
    text_index_dir: str = INDEX_PATHS.text_index_dir,
    vector_index_dir: str = INDEX_PATHS.vector_index_dir,
    embedding_api_url: str = DEFAULT_EMBEDDING_API_URL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_task: str = "",
    embedding_api_key: str = DEFAULT_EMBEDDING_API_KEY,
    embedding_no_proxy: bool = True,
    embedding_dimensions: int | None = None,
) -> RRFSearchClient:
    """Initialize the global RRF search client."""
    global _client
    _client = RRFSearchClient(
        text_index_dir=text_index_dir,
        vector_index_dir=vector_index_dir,
        embedding_api_url=embedding_api_url,
        embedding_model=embedding_model,
        embedding_task=embedding_task,
        embedding_api_key=embedding_api_key,
        embedding_no_proxy=embedding_no_proxy,
        embedding_dimensions=embedding_dimensions,
    )
    return _client


def get_client() -> RRFSearchClient:
    """Get the global client, initializing if needed."""
    global _client
    if _client is None:
        _client = init_client()
    return _client


def Search(
    query: str,
    top_k: int = 10,
    only_title: bool = False,
    method: Literal["rrf", "mix", "keywords", "vector"] = "rrf",
    max_chars_per_result: int = 240,
    link_mode: Any = DEFAULT_LINK_MODE,
) -> str:
    """
    Offline Wikipedia retrieval.
    - only_title=True: title search only (better for known titles/entity names).
    - only_title=False: chunk-level search with optional rrf/keywords/vector.
    - link_mode: use "wikiid" (default) or "title" in returned links.
    """
    client = get_client()
    link_mode = normalize_link_mode(link_mode)
    lines: list[str] = []

    if bool(only_title):
        hits = client.text_client.search_title(query, top_k=int(top_k))
        if not hits:
            return "No results found."
        for i, h in enumerate(hits, 1):
            doc_title = h["doc_title"]
            doc_id = h["doc_id"]
            chunk = client.get_chunk_text_by_doc(doc_id, 1)
            snippet = (
                client.text_client.make_snippet(
                    chunk["text"], max_length=int(max_chars_per_result)
                )
                if chunk
                else ""
            )
            url = make_wiki_chunk_url(doc_id, 1, doc_title, link_mode)
            lines.append(f"{i}. [{doc_title}]({url})")
            if snippet:
                lines.append(f"   - {snippet}")
            lines.append("")
        return "\n".join(lines).strip()

    results = client.search(
        query,
        top_k=int(top_k),
        method=str(method),
        snippet_max_length=int(max_chars_per_result),
        verbose=False,
    )
    if not results:
        return "No results found."
    for i, r in enumerate(results, 1):
        doc_title = r["doc_title"]
        doc_id = r["doc_id"]
        chunk_no = int(r["chunk_index"]) + 1
        url = make_wiki_chunk_url(doc_id, chunk_no, doc_title, link_mode)
        snippet = str(r.get("snippet") or "")
        lines.append(f"{i}. [{doc_title}]({url})")
        if snippet:
            lines.append(f"   - {snippet}")
        lines.append("")
    return "\n".join(lines).strip()


def OpenURL(url: str) -> str:
    """
    Open a Wikipedia URL.
    - Plain URL (no #chunk-N): return full article content (may be truncated).
    - Chunk URL (with #chunk-N): return the corresponding chunk content.
    """
    # Text-index open path lives in search_kws; RRF only needs the text client ready.
    _ = get_client()
    return open_wikipedia_url(url)


if __name__ == "__main__":
    q = "Donald Trump"

    # results = Search(q, top_k=10, only_title=False, method="rrf")
    # print(results)

    # url = "https://en.wikipedia.org/wiki/53219662#chunk-1"
    # text = OpenURL(url)
    # print(text)

    url = "https://en.wikipedia.org/wiki/53219662"
    text = OpenURL(url)
    print(text)

"""
python -m tools.wikipedia_offline.search_rrf
"""
