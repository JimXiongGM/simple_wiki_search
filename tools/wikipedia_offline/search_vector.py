"""Vector search tool using FAISS HNSW index."""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass

import faiss
import numpy as np
import psutil
import requests
from loguru import logger

from settings import (
    DEFAULT_EMBEDDING_API_KEY,
    DEFAULT_EMBEDDING_API_URL,
    DEFAULT_EMBEDDING_MODEL,
    INDEX_PATHS,
)
from storage.search_vector_cache import get_search_vector_cache


def get_memory_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024


def _env_https_proxy() -> str | None:
    for name in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"):
        value = os.environ.get(name)
        if value and str(value).strip():
            return str(value).strip()
    return None


def make_vector_snippet(text: str, max_length: int = 3000) -> str:
    """Build display snippet for vector retrieval results."""
    snippet = str(text or "")
    if len(snippet) <= max_length:
        return snippet
    return (
        snippet[:max_length] + f"... (Omitted {len(snippet) - max_length} characters)"
    )


def _format_query_with_task(task: str, query: str) -> str:
    """Format Qwen3-Embedding style instruction query.

    Ref: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
    """
    t = (task or "").strip()
    q = (query or "").strip()
    if not t:
        return q
    return f"Instruct: {t}\nQuery: {q}"


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    arr = np.ascontiguousarray(arr)
    # Avoid NaN/Inf and divide-by-zero
    if not np.isfinite(arr).all():
        raise ValueError("query_vec contains NaN/Inf")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (arr / norms).astype(np.float32, copy=False)


class VectorSearcher:
    """FAISS HNSW-based vector search."""

    def __init__(self, index_dir: str):
        """Load FAISS index and chunk IDs.

        Args:
            index_dir: Path to index folder (contains index.faiss and ids.pkl)
        """
        index_path = os.path.join(index_dir, "index.faiss")
        ids_path = os.path.join(index_dir, "ids.pkl")

        logger.info(f"Loading index from {index_path}, mem: {get_memory_mb():.0f}MB")
        start = time.time()

        self.index = faiss.read_index(index_path)
        with open(ids_path, "rb") as f:
            self.chunk_ids = pickle.load(f)

        load_time = time.time() - start
        logger.info(
            f"Loaded {self.index.ntotal} vectors in {load_time:.2f}s, mem: {get_memory_mb():.0f}MB"
        )

        # Disk size info
        index_size = os.path.getsize(index_path) / 1024 / 1024
        ids_size = os.path.getsize(ids_path) / 1024 / 1024
        logger.info(f"Disk: index={index_size:.1f}MB, ids={ids_size:.1f}MB")

        self.index_dir = index_dir
        # CPU FAISS is safe for concurrent read-only search. Keep OpenMP at 1 by
        # default so multiple Python threads each running search() do not
        # oversubscribe cores (override via FAISS_NUM_THREADS).
        try:
            n = os.environ.get("FAISS_NUM_THREADS")
            if n is None or not str(n).strip():
                num_threads = 1
            else:
                num_threads = max(1, int(float(n)))
            faiss.omp_set_num_threads(int(num_threads))
        except Exception:
            pass

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        """Search for similar vectors.

        Args:
            query_vec: Query vector (1D or 2D array)
            top_k: Number of results

        Returns:
            List of (chunk_id, score) tuples
        """
        q = _l2_normalize_rows(query_vec)
        scores, indices = self.index.search(q, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:  # -1 means no result
                results.append((self.chunk_ids[idx], float(score)))
        return results

    def search_batch(
        self, query_vecs: np.ndarray, top_k: int = 10
    ) -> list[list[tuple[str, float]]]:
        """Batch search for similar vectors."""
        q = _l2_normalize_rows(query_vecs)
        scores, indices = self.index.search(q, top_k)

        all_results = []
        for i in range(len(query_vecs)):
            results = []
            for idx, score in zip(indices[i], scores[i]):
                if idx >= 0:
                    results.append((self.chunk_ids[idx], float(score)))
            all_results.append(results)
        return all_results


class EmbeddingClient:
    """Client for embedding API (local OpenAI-compatible, AiHubMix fallback)."""

    def __init__(
        self,
        api_url: str = DEFAULT_EMBEDDING_API_URL,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        task: str = "",
        api_key: str = DEFAULT_EMBEDDING_API_KEY,
        no_proxy: bool = True,
        dimensions: int | None = None,
    ):
        self.api_url = api_url
        self.model_name = model_name
        self.task = (task or "").strip()
        # api_key/no_proxy describe any OpenAI-compatible endpoint (local/aihubmix)
        self.api_key = api_key or DEFAULT_EMBEDDING_API_KEY
        self.no_proxy = bool(no_proxy)
        self.dimensions = int(dimensions) if dimensions is not None else None
        self._use_online = False
        self._online_client = None

    def _init_online_client(self):
        """Initialize AiHubMix embeddings client."""
        if self._online_client is not None:
            return True

        api_key = os.environ.get("AIHUBMIX_EMBEDDING_API_KEY")
        if not api_key:
            logger.warning(
                "AiHubMix embedding key not set (AIHUBMIX_EMBEDDING_API_KEY), "
                "cannot use online embedding API"
            )
            return False

        import httpx
        import openai

        timeout = float(os.environ.get("EMBEDDING_API_TIMEOUT", "60"))
        # Prefer explicit proxy so NO_PROXY cannot bypass AiHubMix.
        proxy = _env_https_proxy()
        http_client = httpx.Client(
            timeout=timeout,
            trust_env=proxy is None,
            proxy=proxy,
        )
        self._online_client = openai.OpenAI(
            api_key=api_key,
            base_url="https://aihubmix.com/v1",
            timeout=timeout,
            http_client=http_client,
        )
        logger.info("Initialized AiHubMix embedding client")
        return True

    def _backend_key(self) -> str:
        if self._use_online:
            return "aihubmix:qwen3-embedding-0.6b"
        dim = f"|dim={self.dimensions}" if self.dimensions is not None else ""
        return f"local:{self.api_url}|{self.model_name}{dim}"

    def _embed_local(self, texts: list[str]) -> np.ndarray:
        last_exc: Exception | None = None
        with requests.Session() as session:
            # no_proxy=True: direct. Else: explicit proxy so NO_PROXY cannot bypass.
            proxies = None
            if self.no_proxy:
                session.trust_env = False
            else:
                session.trust_env = False
                proxy = _env_https_proxy()
                if proxy:
                    proxies = {"http": proxy, "https": proxy}
                else:
                    session.trust_env = True
            for attempt in range(3):
                try:
                    response = session.post(
                        f"{self.api_url}/v1/embeddings",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "model": self.model_name,
                            "input": texts,
                            "encoding_format": "float",
                            **(
                                {"dimensions": self.dimensions}
                                if self.dimensions is not None
                                else {}
                            ),
                        },
                        timeout=60,
                        proxies=proxies,
                    )
                    response.raise_for_status()
                    embeddings = [d["embedding"] for d in response.json()["data"]]
                    return np.array(embeddings, dtype=np.float32)
                except Exception as e:
                    last_exc = e
                    if attempt < 2:
                        time.sleep(min(2.0, 0.5 * (2**attempt)))
                        continue
                    raise
        raise last_exc or RuntimeError("local embedding failed")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Get embeddings for texts (with cache)."""
        cache = get_search_vector_cache()
        backend = self._backend_key()

        out: list[np.ndarray | None] = [None] * len(texts)
        miss_texts: list[str] = []
        miss_indices: list[int] = []

        for i, t in enumerate(texts):
            v = cache.get(t, backend)
            if v is None:
                miss_texts.append(t)
                miss_indices.append(i)
            else:
                out[i] = v

        if miss_texts:
            try:
                if self._use_online:
                    miss_vecs = self._embed_online(miss_texts)
                else:
                    miss_vecs = self._embed_local(miss_texts)
            except Exception as e:
                if not self._use_online:
                    logger.warning(f"Local embedding API call failed: {e}")
                    logger.info("Switching to AiHubMix embedding API...")
                    self._use_online = True
                    backend = self._backend_key()

                    # retry (and re-check cache with new backend)
                    retry_out: list[np.ndarray] = []
                    retry_texts: list[str] = []
                    retry_map: list[int] = []
                    for idx, t in zip(miss_indices, miss_texts):
                        v = cache.get(t, backend)
                        if v is None:
                            retry_texts.append(t)
                            retry_map.append(idx)
                        else:
                            out[idx] = v
                    if retry_texts:
                        retry_out = list(self._embed_online(retry_texts))
                        for idx, t, v in zip(retry_map, retry_texts, retry_out):
                            cache.set(t, backend, v)
                            out[idx] = np.asarray(v, dtype=np.float32).reshape(-1)
                else:
                    raise
            else:
                for idx, t, v in zip(miss_indices, miss_texts, miss_vecs):
                    cache.set(t, backend, v)
                    out[idx] = np.asarray(v, dtype=np.float32).reshape(-1)

        if any(v is None for v in out):
            raise RuntimeError("embedding cache/internal error: missing vectors")
        return np.stack([np.asarray(v, dtype=np.float32) for v in out]).astype(
            np.float32, copy=False
        )

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        """Embed queries with optional task instruction prefix."""
        if not self.task:
            return self.embed(queries)
        formatted = [_format_query_with_task(self.task, q) for q in queries]
        return self.embed(formatted)

    def embed_docs(self, docs: list[str]) -> np.ndarray:
        """Embed documents (no task prefix)."""
        return self.embed(docs)

    def _embed_online(self, texts: list[str]) -> np.ndarray:
        """Get embeddings using AiHubMix."""
        if not self._init_online_client():
            raise RuntimeError(
                "Online embedding API unavailable: set AIHUBMIX_EMBEDDING_API_KEY"
            )

        response = self._online_client.embeddings.create(
            input=texts, model="qwen3-embedding-0.6b"
        )
        embeddings = [d.embedding for d in response.data]
        return np.array(embeddings, dtype=np.float32)


class VectorSearchTool:
    """Complete vector search tool with embedding support."""

    def __init__(
        self,
        index_dir: str,
        api_url: str = DEFAULT_EMBEDDING_API_URL,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        task: str = "",
        api_key: str = DEFAULT_EMBEDDING_API_KEY,
        no_proxy: bool = True,
        dimensions: int | None = None,
    ):
        self.searcher = VectorSearcher(index_dir)
        self.embedder = EmbeddingClient(
            api_url,
            model_name,
            task=task,
            api_key=api_key,
            no_proxy=no_proxy,
            dimensions=dimensions,
        )

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search by text query.

        Args:
            query: Text query
            top_k: Number of results

        Returns:
            List of (chunk_id, score) tuples
        """
        query_vec = self.embedder.embed_queries([query])[0]
        return self.searcher.search(query_vec, top_k)


def get_memory_str():
    """Get memory usage as human-readable string (auto MB/GB)."""
    mem_mb = get_memory_mb()
    if mem_mb >= 1024:
        return f"{mem_mb / 1024:.2f}GB"
    return f"{mem_mb:.0f}MB"


@dataclass
class CliArgs:
    index_dir: str = INDEX_PATHS.vector_index_dir
    api_url: str = DEFAULT_EMBEDDING_API_URL
    model: str = DEFAULT_EMBEDDING_MODEL
    task: str = ""
    top_k: int = 5


def _run_cli(args: CliArgs) -> None:
    print(f"Loading index from {args.index_dir}...")
    print(f"Memory before load: {get_memory_str()}")
    load_start = time.time()
    searcher = VectorSearcher(args.index_dir)
    load_time = time.time() - load_start
    print(f"Index loaded in {load_time:.2f}s, memory: {get_memory_str()}")

    embedder = EmbeddingClient(args.api_url, args.model, task=str(args.task))
    print(f"\nEmbedding API: {args.api_url}")
    print(f"Model: {args.model}")
    if str(args.task).strip():
        print(f"Task prefix: {str(args.task).strip()}")

    print("\nCommands:")
    print("  search <query>  - Search by text query (requires embedding API)")
    print("  mem             - Show current memory usage")
    print("  exit            - Exit the program")
    print()

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
        if cmd.lower() == "mem":
            print(f"Memory usage: {get_memory_str()}")
            continue

        parts = cmd.split(maxsplit=1)
        cmd_name = parts[0].lower()
        cmd_args = parts[1] if len(parts) > 1 else ""

        exec_start = time.time()
        try:
            if cmd_name == "search" and cmd_args:
                embed_start = time.time()
                query_vec = embedder.embed_queries([cmd_args])[0]
                embed_time = time.time() - embed_start

                search_start = time.time()
                results = searcher.search(query_vec, top_k=args.top_k)
                search_time = time.time() - search_start

                lines = [f"Query: {cmd_args}", ""]
                for i, (chunk_id, score) in enumerate(results, 1):
                    lines.append(f"{i}. {chunk_id} (score: {score:.4f})")
                lines.append("")
                lines.append(
                    f"Embed: {embed_time*1000:.1f}ms, Search: {search_time*1000:.1f}ms"
                )
                result = "\n".join(lines)
            else:
                result = f"Unknown command: {cmd_name}. Try: search, mem, exit"
        except Exception as e:
            result = f"Error: {e}"

        exec_time = time.time() - exec_start
        print(result)
        print(f"[Total: {exec_time*1000:.1f}ms, Memory: {get_memory_str()}]")


if __name__ == "__main__":
    import tyro

    _run_cli(tyro.cli(CliArgs, use_underscores=True))

"""
python -m tools.wikipedia_offline.search_vector --index_dir data/database/wikipedia-index/enwiki-20260601/faiss/hnsw-Qwen3-Embedding-0.6B-chunk-1536-fp16
"""
