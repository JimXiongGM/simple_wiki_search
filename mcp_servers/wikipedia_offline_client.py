from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

import httpx
from cachetools.func import lru_cache

from settings import DEFAULT_TOOL_BASE_URL
from tools.wikipedia_offline.search_method import normalize_search_method
from tools.wikipedia_offline.wiki_url import LinkMode

Method = Literal[
    "rrf",
    "mix",
    "kw",
    "kws",
    "keyword",
    "keywords",
    "vector",
    "vec",
]

RandomMode = Literal["chunk", "page"]


@dataclass
class CliArgs:
    search: str | None = None
    open_url: str | None = None
    random: RandomMode | None = None
    bench: bool = False
    top_k: int = 10
    only_title: bool = False
    method: Method = "rrf"
    seed: int | None = None
    link_mode: LinkMode | None = None
    max_chars_per_result: int = 1024
    base_url: str = DEFAULT_TOOL_BASE_URL
    timeout: float = 180.0
    qps: int = 100
    seconds: int = 10
    concurrency: int = 100


@dataclass
class WikipediaOfflineClient:
    """Synchronous Wikipedia tool client with a long-lived HTTP connection pool."""

    base_url: str = DEFAULT_TOOL_BASE_URL
    timeout: float = 180.0
    _client: httpx.Client | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.base_url = str(self.base_url).rstrip("/")
        self.timeout = float(self.timeout)

    def _api_url(self, path: str) -> str:
        return str(self.base_url).rstrip("/") + path

    def _http(self) -> httpx.Client:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = httpx.Client(
                        timeout=float(self.timeout),
                        trust_env=False,
                    )
        return self._client

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None

    def _post(self, path: str, payload: dict[str, Any]) -> str:
        response = self._http().post(self._api_url(path), json=payload)
        response.raise_for_status()
        data = response.json()
        return str(data.get("content") or "")

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        only_title: bool = False,
        method: Method = "rrf",
        max_chars_per_result: int = 1024,
        link_mode: LinkMode | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "query": query,
            "top_k": int(top_k),
            "only_title": bool(only_title),
            "method": normalize_search_method(method),
            "max_chars_per_result": int(max_chars_per_result),
        }
        if link_mode is not None:
            payload["link_mode"] = str(link_mode).strip().lower()
        return self._post("/search", payload)

    def random(
        self,
        mode: RandomMode = "chunk",
        *,
        seed: int | None = None,
    ) -> str:
        payload: dict[str, Any] = {"mode": str(mode).strip().lower()}
        if seed is not None:
            payload["seed"] = int(seed)
        return self._post("/random", payload)

    def open_url(self, url: str) -> str:
        return self._post("/open_url", {"url": url})

    def benchmark_search(
        self,
        *,
        query: str,
        qps: int = 100,
        seconds: int = 10,
        concurrency: int = 100,
        top_k: int = 5,
        method: Method = "keywords",
        only_title: bool = False,
        max_chars_per_result: int = 512,
    ) -> dict[str, float | int]:
        """Synchronous QPS-style bench using a thread pool + shared HTTP client."""
        import concurrent.futures
        import time

        total_requests = max(1, int(qps) * max(1, int(seconds)))
        interval = 1.0 / max(1, int(qps))
        payload = {
            "query": str(query),
            "top_k": int(top_k),
            "only_title": bool(only_title),
            "method": normalize_search_method(method),
            "max_chars_per_result": int(max_chars_per_result),
        }

        latencies: list[float] = []
        ok = 0
        err = 0
        lock = threading.Lock()
        t0 = perf_counter()

        def one_request() -> None:
            nonlocal ok, err
            st = perf_counter()
            try:
                self._post("/search", payload)
                with lock:
                    ok += 1
            except Exception:
                with lock:
                    err += 1
            finally:
                with lock:
                    latencies.append((perf_counter() - st) * 1000.0)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, int(concurrency))
        ) as pool:
            futures: list[concurrent.futures.Future[None]] = []
            for i in range(total_requests):
                target_ts = t0 + i * interval
                now = perf_counter()
                if target_ts > now:
                    time.sleep(target_ts - now)
                futures.append(pool.submit(one_request))
                if (i + 1) % max(1, int(qps)) == 0:
                    elapsed_now = max(1e-9, perf_counter() - t0)
                    print(
                        "[wikipedia_offline][bench] "
                        f"sent={i + 1}/{total_requests} "
                        f"ok={ok} err={err} "
                        f"elapsed={elapsed_now:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )
            concurrent.futures.wait(futures)

        elapsed = max(1e-9, perf_counter() - t0)
        lat_sorted = sorted(latencies) if latencies else [0.0]
        p50_idx = min(len(lat_sorted) - 1, int(len(lat_sorted) * 0.50))
        p95_idx = min(len(lat_sorted) - 1, int(len(lat_sorted) * 0.95))
        return {
            "requested": total_requests,
            "success": ok,
            "errors": err,
            "elapsed_s": round(elapsed, 3),
            "actual_qps": round(total_requests / elapsed, 2),
            "p50_ms": round(lat_sorted[p50_idx], 2),
            "p95_ms": round(lat_sorted[p95_idx], 2),
        }


DEFAULT_BASE_URL = DEFAULT_TOOL_BASE_URL
_explicit_base_url: str | None = None


def configure_default_client(base_url: str) -> None:
    global _explicit_base_url
    if get_default_client.cache_info().currsize > 0:
        try:
            get_default_client().close()
        except Exception:
            pass
    _explicit_base_url = str(base_url).rstrip("/")
    get_default_client.cache_clear()


@lru_cache(maxsize=1)
def get_default_client() -> WikipediaOfflineClient:
    url = (
        _explicit_base_url or os.environ.get("TOOL_BASE_URL") or DEFAULT_TOOL_BASE_URL
    ).rstrip("/")
    return WikipediaOfflineClient(base_url=url)


def _safe_top_k(value: Any, default: int = 10) -> int:
    try:
        top_k = int(value)
    except (TypeError, ValueError):
        return default
    return top_k if top_k >= 1 else default


def _safe_method(value: Any, default: str = "rrf") -> str:
    method = normalize_search_method(value)
    return method if method in {"rrf", "keywords", "vector"} else default


def execute_tool(
    name: str,
    arguments_json: str,
    client: WikipediaOfflineClient | None = None,
) -> str:
    wiki_client = client or get_default_client()
    args = json.loads(arguments_json)
    if name == "search":
        return wiki_client.search(
            str(args.get("query") or ""),
            top_k=_safe_top_k(args.get("top_k", 10)),
            only_title=bool(args.get("only_title", False)),
            method=_safe_method(args.get("method", "rrf")),
        )
    if name == "open_url":
        return wiki_client.open_url(str(args["url"]))
    if name == "submit_answer":
        return json.dumps(
            {"status": "done", "answer": str(args.get("answer") or "")},
            ensure_ascii=False,
        )
    raise ValueError(f"Unknown tool: {name}")


if __name__ == "__main__":
    import tyro

    args = tyro.cli(CliArgs, use_underscores=True)
    client = WikipediaOfflineClient(base_url=args.base_url, timeout=float(args.timeout))

    selected = sum(x is not None for x in (args.search, args.open_url, args.random))
    if selected != 1:
        raise SystemExit("exactly one of --search, --open_url, --random is required")

    if args.random is not None:
        md = client.random(mode=args.random, seed=args.seed)
        print(
            f"[wikipedia_offline] tool=Random mode={args.random} seed={args.seed}",
            file=sys.stderr,
        )
        print(md)
        raise SystemExit(0)

    if args.search is not None:
        if bool(args.bench):
            stats = client.benchmark_search(
                query=args.search,
                qps=int(args.qps),
                seconds=int(args.seconds),
                concurrency=int(args.concurrency),
                top_k=int(args.top_k),
                method=args.method,
                only_title=bool(args.only_title),
                max_chars_per_result=int(args.max_chars_per_result),
            )
            print(
                f"[wikipedia_offline] bench qps={int(args.qps)} seconds={int(args.seconds)} concurrency={int(args.concurrency)}",
                file=sys.stderr,
            )
            print(stats)
            raise SystemExit(0)

        md = client.search(
            args.search,
            top_k=int(args.top_k),
            only_title=bool(args.only_title),
            method=args.method,
            max_chars_per_result=int(args.max_chars_per_result),
            link_mode=args.link_mode,
        )
        effective_method = (
            "title" if bool(args.only_title) else normalize_search_method(args.method)
        )
        print(
            f"[wikipedia_offline] tool=Search top_k={int(args.top_k)} only_title={bool(args.only_title)} effective_method={effective_method} requested_method={args.method} max_chars_per_result={int(args.max_chars_per_result)}",
            file=sys.stderr,
        )
        print(md)
        raise SystemExit(0)

    assert args.open_url is not None
    md = client.open_url(args.open_url)
    print("[wikipedia_offline] tool=OpenURL", file=sys.stderr)
    print(md)
    raise SystemExit(0)

"""
# search
python -m mcp_servers.wikipedia_offline_client --search "capital of China" --top_k 5
python -m mcp_servers.wikipedia_offline_client --search "Albert Einstein theory of relativity" --top_k 5 --method keywords
python -m mcp_servers.wikipedia_offline_client --search "Albert Einstein" --only_title
# python -m mcp_servers.wikipedia_offline_client --search "capital of China" --bench --qps 100 --seconds 10 --concurrency 100

# open url
python -m mcp_servers.wikipedia_offline_client --open_url "https://en.wikipedia.org/wiki/Albert_Einstein"
python -m mcp_servers.wikipedia_offline_client --open_url "https://en.wikipedia.org/wiki/Albert_Einstein#chunk-1"

# random chunk / random page (single body, same as OpenURL)
python -m mcp_servers.wikipedia_offline_client --random chunk --seed 42
python -m mcp_servers.wikipedia_offline_client --random page --seed 42
"""
