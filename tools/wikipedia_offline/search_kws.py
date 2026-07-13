"""
Wikipedia Text Search Tools using Tantivy.

Tools:
1. Search(query) - Search chunks, returns deduplicated documents with title_with_id
2. SearchTitle(title) - Search documents by title
3. GetDocument(identifier) - Get full document (truncated if too long)
"""

from __future__ import annotations

import random
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cachetools
import tantivy

from settings import INDEX_PATHS
from tools.wikipedia_offline.types import (
    ChunkRecord,
    DocInfo,
    KeywordHit,
    RandomChunkHit,
    RandomPageHit,
    TitleHit,
)
from tools.wikipedia_offline.wiki_url import parse_wikipedia_url
from utils.common import colorful, extract_toc_from_md
from utils.data_cleaning import drop_empty_md_table_rows


def make_title_with_id(title: str, doc_id: str) -> str:
    return f"Title: {title} (ID: {doc_id})"


def _extract_doc_id(identifier: str) -> str | None:
    """Extract doc_id from identifier. Supports: pure ID, or 'xxx (ID: 123)'."""
    # Check pure numeric ID
    if identifier.strip().isdigit():
        return identifier.strip()
    # Check pattern 'xxx (ID: 123)'
    m = re.search(r"\(ID:\s*(\d+)\)", identifier)
    if m:
        return m.group(1)
    return None


class LRUCache:
    """Thread-safe LRU cache backed by cachetools; get returns (found, value).

    ``None`` is a valid cached value (negative cache), so membership is separate
    from the stored payload.
    """

    def __init__(self, max_size: int = 100):
        self._cache: cachetools.LRUCache[str, Any] = cachetools.LRUCache(
            maxsize=max_size
        )
        self._lock = threading.Lock()

    def get(self, key: str) -> tuple[bool, Any]:
        """Get item from cache. Returns (found, value)."""
        with self._lock:
            if key in self._cache:
                return True, self._cache[key]
            return False, None

    def put(self, key: str, value: Any) -> None:
        """Put item into cache."""
        with self._lock:
            self._cache[key] = value


class WikiSearchClient:
    """Wikipedia search client using Tantivy indexes."""

    def __init__(self, index_dir: str, cache_size: int = 100):
        """
        Args:
            index_dir: Directory containing 'chunks' and 'titles' subdirectories
            cache_size: Maximum number of items to cache
        """
        self.index_dir = Path(index_dir)
        chunks_dir = self.index_dir / "chunks"
        titles_dir = self.index_dir / "titles"

        if not chunks_dir.exists() or not titles_dir.exists():
            raise FileNotFoundError(f"Index not found in {index_dir}")

        self.chunks_index = tantivy.Index.open(str(chunks_dir))
        self.titles_index = tantivy.Index.open(str(titles_dir))
        # Tantivy python bindings are not guaranteed to be thread-safe when sharing
        # searcher objects across threads. For stability, keep per-thread searchers.
        self._local = threading.local()
        # Also guard against any internal non-thread-safe paths.
        self._tantivy_lock = threading.RLock()

        # Caches
        self._doc_info_cache = LRUCache(cache_size)
        self._chunks_cache = LRUCache(cache_size)

    def _chunks_searcher(self):
        s = getattr(self._local, "chunks_searcher", None)
        if s is None:
            s = self.chunks_index.searcher()
            self._local.chunks_searcher = s
        return s

    def _titles_searcher(self):
        s = getattr(self._local, "titles_searcher", None)
        if s is None:
            s = self.titles_index.searcher()
            self._local.titles_searcher = s
        return s

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_fuzzy: bool = False,
        dedupe: bool = True,
        snippet_max_length: int = 3000,
    ) -> list[KeywordHit]:
        """
        Search chunks by text query with optimized strategy:
        - Boost doc_title field for better relevance
        - Use fuzzy matching for typo tolerance (optional, slow on large indexes)
        - Use lenient parsing for robustness
        - Deduplicate by document (keep highest scoring chunk per doc)
        Returns list of {title_with_id, doc_id, chunk_id, score, snippet}.
        """
        # Field boosts: title matches are more relevant
        field_boosts = {"doc_title": 2.0, "text": 1.0}

        # Fuzzy matching: (prefix, distance, transpose_cost_one)
        # WARNING: fuzzy is very slow on large indexes!
        fuzzy_fields = {"text": (True, 1, True)} if use_fuzzy else None

        try:
            # Use lenient parsing for robustness against syntax errors
            if fuzzy_fields:
                q, _errors = self.chunks_index.parse_query_lenient(
                    query,
                    ["text", "doc_title"],
                    field_boosts=field_boosts,
                    fuzzy_fields=fuzzy_fields,
                )
            else:
                q, _errors = self.chunks_index.parse_query_lenient(
                    query,
                    ["text", "doc_title"],
                    field_boosts=field_boosts,
                )
        except Exception:
            # Fallback: simple phrase query
            try:
                q = self.chunks_index.parse_query(f'"{query}"', ["text"])
            except Exception:
                return []

        # Fetch more results if deduping to ensure enough unique docs
        fetch_k = top_k * 3 if dedupe else top_k
        with self._tantivy_lock:
            searcher = self._chunks_searcher()
            results = searcher.search(q, fetch_k).hits
        output = []
        for score, doc_addr in results:
            with self._tantivy_lock:
                doc = searcher.doc(doc_addr)
            doc_id = doc["doc_id"][0]
            doc_title = doc["doc_title"][0]
            text = doc["text"][0]
            chunk_id = doc["chunk_id"][0]
            # Smart snippet: try to break at word boundary
            snippet = self.make_snippet(text, max_length=int(snippet_max_length))
            output.append(
                KeywordHit(
                    title_with_id=make_title_with_id(doc_title, doc_id),
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    score=score,
                    snippet=snippet,
                )
            )

        # Deduplicate: keep highest scoring chunk per document
        if dedupe:
            seen_docs: dict[str, KeywordHit] = {}
            for item in output:
                doc_id = item["doc_id"]
                if (
                    doc_id not in seen_docs
                    or item["score"] > seen_docs[doc_id]["score"]
                ):
                    seen_docs[doc_id] = item
            output = sorted(seen_docs.values(), key=lambda x: x["score"], reverse=True)

        return output[:top_k]

    def make_snippet(self, snippet: str, max_length: int = 3000) -> str:
        """Create snippet, breaking at word boundary."""
        if len(snippet) <= max_length:
            return snippet
        else:
            # Truncate if too long
            snippet = (
                snippet[:max_length]
                + f"... (Omitted {len(snippet) - max_length} characters)"
            )
            return snippet

    def get_doc_by_id(self, doc_id: str) -> DocInfo | None:
        """Get document metadata by exact doc_id (with caching)."""
        # Check cache first
        found, cached = self._doc_info_cache.get(doc_id)
        if found:
            return cached

        q = self.titles_index.parse_query(f'"{doc_id}"', ["doc_id"])
        with self._tantivy_lock:
            searcher = self._titles_searcher()
            results = searcher.search(q, 1).hits
        if not results:
            self._doc_info_cache.put(doc_id, None)
            return None
        _, doc_addr = results[0]
        with self._tantivy_lock:
            doc = searcher.doc(doc_addr)
        doc_id = doc["doc_id"][0]
        doc_title = doc["doc_title"][0]

        # Extract TOC in real-time from full text
        full_text = self.get_full_text_for_doc(doc_id)
        toc = extract_toc_from_md(full_text)

        result = DocInfo(
            doc_id=doc_id,
            doc_title=doc_title,
            toc=toc,
        )
        self._doc_info_cache.put(doc_id, result)
        return result

    def search_title(self, title: str, top_k: int = 5) -> list[TitleHit]:
        """Search titles index by title text."""
        try:
            q = self.titles_index.parse_query(title, ["doc_title"])
        except Exception:
            q = self.titles_index.parse_query(f'"{title}"', ["doc_title"])
        with self._tantivy_lock:
            searcher = self._titles_searcher()
            results = searcher.search(q, top_k).hits
        output: list[TitleHit] = []
        for score, doc_addr in results:
            with self._tantivy_lock:
                doc = searcher.doc(doc_addr)
            output.append(
                TitleHit(
                    doc_id=doc["doc_id"][0],
                    doc_title=doc["doc_title"][0],
                    score=score,
                )
            )
        return output

    def random_chunks(
        self, top_k: int = 1, seed: int | None = None
    ) -> list[RandomChunkHit]:
        """Sample random documents and one random chunk per sampled document."""
        k = max(1, int(top_k))
        rng = random.Random(int(seed)) if seed is not None else random.Random()
        out: list[RandomChunkHit] = []
        picked_doc_ids: set[str] = set()

        with self._tantivy_lock:
            searcher = self._titles_searcher()
            total_docs = int(searcher.num_docs)
            all_query = self.titles_index.parse_query("*", ["doc_title"])
        if total_docs <= 0:
            return []

        max_tries = max(200, k * 200)
        tries = 0
        while len(out) < k and tries < max_tries:
            tries += 1
            offset = rng.randrange(total_docs)
            try:
                with self._tantivy_lock:
                    hits = searcher.search(all_query, limit=1, offset=offset).hits
            except Exception:
                continue
            if not hits:
                continue

            _score, doc_addr = hits[0]
            try:
                with self._tantivy_lock:
                    doc = searcher.doc(doc_addr)
            except Exception:
                continue
            try:
                doc_id = str(doc["doc_id"][0]).strip()
            except Exception:
                doc_id = ""
            try:
                doc_title = str(doc["doc_title"][0]).strip()
            except Exception:
                doc_title = ""
            doc_title = doc_title or doc_id
            if not doc_id or doc_id in picked_doc_ids:
                continue

            chunks = self.get_all_chunks_for_doc(doc_id)
            if not chunks:
                continue
            idx0 = rng.randrange(len(chunks))
            text = str(chunks[idx0].get("text") or "")
            out.append(
                RandomChunkHit(
                    doc_id=doc_id,
                    doc_title=doc_title,
                    chunk_no_1based=idx0 + 1,
                    snippet=self.make_snippet(text, max_length=5000),
                )
            )
            picked_doc_ids.add(doc_id)

        return out

    def random_pages(
        self, top_k: int = 1, seed: int | None = None
    ) -> list[RandomPageHit]:
        """Sample random distinct documents (links target full articles, not chunks)."""
        k = max(1, int(top_k))
        rng = random.Random(int(seed)) if seed is not None else random.Random()
        out: list[RandomPageHit] = []
        picked_doc_ids: set[str] = set()

        with self._tantivy_lock:
            searcher = self._titles_searcher()
            total_docs = int(searcher.num_docs)
            all_query = self.titles_index.parse_query("*", ["doc_title"])
        if total_docs <= 0:
            return []

        max_tries = max(200, k * 200)
        tries = 0
        while len(out) < k and tries < max_tries:
            tries += 1
            offset = rng.randrange(total_docs)
            try:
                with self._tantivy_lock:
                    hits = searcher.search(all_query, limit=1, offset=offset).hits
            except Exception:
                continue
            if not hits:
                continue

            _score, doc_addr = hits[0]
            try:
                with self._tantivy_lock:
                    doc = searcher.doc(doc_addr)
            except Exception:
                continue
            try:
                doc_id = str(doc["doc_id"][0]).strip()
            except Exception:
                doc_id = ""
            try:
                doc_title = str(doc["doc_title"][0]).strip()
            except Exception:
                doc_title = ""
            doc_title = doc_title or doc_id
            if not doc_id or doc_id in picked_doc_ids:
                continue

            chunks = self.get_all_chunks_for_doc(doc_id)
            if not chunks:
                continue
            text0 = str(chunks[0].get("text") or "")
            out.append(
                RandomPageHit(
                    doc_id=doc_id,
                    doc_title=doc_title,
                    snippet=self.make_snippet(text0, max_length=5000),
                )
            )
            picked_doc_ids.add(doc_id)

        return out

    def get_all_chunks_for_doc(self, doc_id: str) -> list[ChunkRecord]:
        """Get all chunks for a document (with caching)."""
        # Check cache first
        found, cached = self._chunks_cache.get(doc_id)
        if found:
            return cached

        q = self.chunks_index.parse_query(f'"{doc_id}"', ["doc_id"])
        with self._tantivy_lock:
            searcher = self._chunks_searcher()
            results = searcher.search(q, 1000).hits
        chunks: list[ChunkRecord] = []
        for _, doc_addr in results:
            with self._tantivy_lock:
                doc = searcher.doc(doc_addr)
            chunks.append(
                ChunkRecord(
                    chunk_id=doc["chunk_id"][0],
                    text=doc["text"][0],
                )
            )
        # Sort by chunk_id (e.g., "12345_0", "12345_1", ...)
        chunks.sort(key=lambda x: int(x["chunk_id"].split("_")[-1]))

        self._chunks_cache.put(doc_id, chunks)
        return chunks

    def get_full_text_for_doc(self, doc_id: str) -> str:
        """Get full text for a document."""
        chunks = self.get_all_chunks_for_doc(doc_id)
        return "\n\n".join(c["text"] for c in chunks)

    def get_document(self, identifier: str) -> str:
        """
        Get full document text (truncated if too long).
        identifier: doc_id or 'xxx (ID: 123)'
        max_chars: Maximum characters to return (default 50000)
        """
        doc_id = _extract_doc_id(identifier)
        if not doc_id:
            return f"Cannot extract doc_id from '{identifier}'. Use Search(query) first to get ID."

        doc_info = self.get_doc_by_id(doc_id)
        if not doc_info:
            return f"Document with ID '{doc_id}' not found."

        full_text = self.get_full_text_for_doc(doc_id)
        header = f"# {doc_info['doc_title']} (ID: {doc_id})\n\n"

        return header + full_text


# Thread-safe global client with lock
_client: WikiSearchClient | None = None
_client_lock = threading.Lock()


def _init_client_unlocked(
    index_dir: str = INDEX_PATHS.text_index_dir,
) -> WikiSearchClient:
    """Construct the global client. Caller must hold ``_client_lock``."""
    global _client
    _client = WikiSearchClient(index_dir)
    return _client


def init_client(
    index_dir: str = INDEX_PATHS.text_index_dir,
) -> WikiSearchClient:
    """Initialize the global client."""
    with _client_lock:
        return _init_client_unlocked(index_dir)


def get_client() -> WikiSearchClient:
    """Get the global client, initializing if needed."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _init_client_unlocked()
    return _client


# ============================================================================
# Tool functions (exposed to LLM agents)
# ============================================================================


def Search(query: str, top_k: int = 10) -> str:
    """
    Search Wikipedia by query text. Returns deduplicated list of documents.
    Use this to find relevant documents for a topic or question.
    """
    client = get_client()
    results = client.search(query, top_k)
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {colorful(r['title_with_id'])}")
        lines.append(f"   Score: {r['score']:.2f}")
        lines.append(f"   {r['snippet']}")
        lines.append("")
    return "\n".join(lines)


def SearchTitle(title: str, top_k: int = 5) -> str:
    """
    Search documents by title. Use this when you know the exact or approximate title.
    Returns list of matching documents with their IDs.
    """
    client = get_client()
    results = client.search_title(title, top_k)
    if not results:
        return f"No documents found with title matching '{title}'."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. Title: {colorful(r['doc_title'])} (ID: {r['doc_id']}), Score: {r['score']:.2f}"
        )
    return "\n".join(lines)


def GetDocument(identifier: str) -> str:
    """
    Get full document content by ID. May be truncated for very long documents.
    For long documents, use GetContents() + ReadSection() instead.
    """
    return get_client().get_document(identifier)


def render_section_headers_md(text: str) -> str:
    """Turn ``Section N:`` lines into markdown headings."""
    s = text or ""
    if not s:
        return s
    lines = s.splitlines()
    out: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            out.append(line)
            continue
        if not stripped.startswith("Section "):
            out.append(line)
            continue
        m = re.match(r"^Section\s+(\d+(?:\.\d+)*)\s*:(.*)$", stripped)
        if not m:
            out.append(line)
            continue
        sec_no = m.group(1)
        rest = (m.group(2) or "").strip()
        depth = len([p for p in sec_no.split(".") if p])
        level = max(2, min(6, depth))
        if rest:
            out.append(f"{'#' * level} Section {sec_no}: {rest}")
        else:
            out.append(f"{'#' * level} Section {sec_no}")
    rendered = "\n".join(out)
    if s.endswith("\n"):
        rendered += "\n"
    return rendered


def OpenURL(url: str, *, client: WikiSearchClient | None = None) -> str:
    """
    Open a Wikipedia URL.
    - Plain URL (no #chunk-N): return full article content.
    - Chunk URL (with #chunk-N): return the corresponding chunk content.
    """
    client = client or get_client()
    identifier, chunk_no = parse_wikipedia_url(url)
    if not identifier:
        return (
            "Invalid Wikipedia URL. Expect "
            "https://en.wikipedia.org/wiki/<wikipedia_id> or with #chunk-N."
        )

    doc_id = identifier
    doc_info = client.get_doc_by_id(doc_id)
    if not doc_info:
        title = identifier.replace("_", " ").strip()
        hits = client.search_title(title, top_k=1)
        if not hits:
            return f"No document found for identifier: {identifier}"
        doc_id = str(hits[0].get("doc_id") or "").strip()
        if not doc_id:
            return f"No document found for identifier: {identifier}"
        doc_info = client.get_doc_by_id(doc_id)
        if not doc_info:
            return f"Document with ID '{doc_id}' not found."

    if chunk_no is not None:
        chunks = client.get_all_chunks_for_doc(doc_id)
        if not chunks:
            return f"Document with ID '{doc_id}' not found."
        idx0 = int(chunk_no) - 1
        if idx0 < 0 or idx0 >= len(chunks):
            return f"Chunk not found: {identifier}#chunk-{chunk_no}"
        lines = drop_empty_md_table_rows(chunks[idx0]["text"].split("\n"))
        text = render_section_headers_md("\n".join(lines))
        return (
            f"# {doc_info['doc_title']} (chunk {int(chunk_no)}/{len(chunks)})\n\n"
            + text
            + "\n"
        )

    text = client.get_document(doc_id)
    lines = drop_empty_md_table_rows(text.split("\n"))
    return render_section_headers_md("\n".join(lines)).strip()


# ============================================================================
# CLI for testing
# ============================================================================


@dataclass
class CliArgs:
    index_dir: str = INDEX_PATHS.text_index_dir
    top_k: int = 5


def _run_cli(args: CliArgs) -> None:
    import time

    print(f"Loading index from {args.index_dir}...")
    load_start = time.time()
    init_client(args.index_dir)
    load_time = time.time() - load_start
    print(f"Index loaded in {load_time:.2f}s")
    print("\nCommands:")
    print("  search <query>         - Search documents by content")
    print("  title <title>          - Search documents by title")
    print("  random <k> [seed]      - Sample k random chunks (optional seed)")
    print("  get <id>               - Get full document")
    print("  exit                   - Exit the program")
    print()

    import readline  # Enable arrow keys and history in input()

    while True:
        try:
            cmd = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not cmd:
            continue
        if cmd.lower() == "exit":
            print("Bye!")
            break

        parts = cmd.split(maxsplit=1)
        cmd_name = parts[0].lower()
        cmd_args = parts[1] if len(parts) > 1 else ""

        exec_start = time.time()
        try:
            if cmd_name == "search" and cmd_args:
                result = Search(cmd_args, top_k=args.top_k)
            elif cmd_name == "title" and cmd_args:
                result = SearchTitle(cmd_args, top_k=args.top_k)
            elif cmd_name == "random":
                toks = [t for t in cmd_args.split() if t]
                k = 1
                seed = None
                if len(toks) >= 1:
                    k = int(toks[0])
                if len(toks) >= 2:
                    seed = int(toks[1])
                rows = get_client().random_chunks(top_k=k, seed=seed)
                if not rows:
                    result = "No results found."
                else:
                    lines = []
                    for i, r in enumerate(rows, 1):
                        lines.append(
                            f"{i}. Title: {colorful(r['doc_title'])} (ID: {r['doc_id']}), chunk={r['chunk_no_1based']}"
                        )
                        if r.get("snippet"):
                            lines.append(f"   {r['snippet']}")
                        lines.append("")
                    result = "\n".join(lines).strip()
            elif cmd_name == "get" and cmd_args:
                result = GetDocument(cmd_args)
            else:
                result = f"Unknown command: {cmd_name}. Try: search, title, random, get, exit"
        except Exception as e:
            result = f"Error: {e}"

        exec_time = time.time() - exec_start
        print(result)
        print(f"\n[Executed in {exec_time*1000:.1f}ms]")


if __name__ == "__main__":
    import tyro

    _run_cli(tyro.cli(CliArgs, use_underscores=True))

"""
python -m tools.wikipedia_offline.search_kws
"""
