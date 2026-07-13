"""HTTP server that exposes tools.wikipedia_offline as FastAPI endpoints."""

from __future__ import annotations

import asyncio
import functools
import multiprocessing as mp
import os
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from mcp_servers.common import (
    exc_location,
    format_tool_error_md,
    is_timeout_exc,
    mem_heartbeat_task,
)
from settings import (
    DEFAULT_EMBEDDING_API_KEY,
    DEFAULT_EMBEDDING_API_URL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOOL_HOST,
    DEFAULT_TOOL_PORT,
    INDEX_PATHS,
)
from tools.wikipedia_offline.runtime import (
    WikiRuntimeConfig,
    build_rrf_markdown,
    hydrate_chunk_infos,
    open_url,
    random_page_or_chunk,
    search_markdown,
    search_text_rrf_candidates,
    search_vector_candidates,
    text_worker_initializer,
    vector_runtime_preload,
    worker_ping,
)
from tools.wikipedia_offline.search_method import normalize_search_method
from tools.wikipedia_offline.wiki_url import normalize_link_mode
from utils.logging import setup_loguru


@dataclass
class Args:
    host: str = DEFAULT_TOOL_HOST
    port: int = DEFAULT_TOOL_PORT
    log_level: str = "INFO"
    text_index_dir: str = INDEX_PATHS.text_index_dir
    vector_index_dir: str = INDEX_PATHS.vector_index_dir
    embedding_api_url: str = DEFAULT_EMBEDDING_API_URL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_api_key: str = DEFAULT_EMBEDDING_API_KEY
    embedding_no_proxy: bool = True
    embedding_auto_detect: bool = True
    embedding_dimensions: int | None = None
    lazy_load_faiss: bool = False
    text_workers: int = 12
    worker_start_method: Literal["fork", "spawn", "forkserver"] = "fork"
    max_search_concurrency: int = 32
    max_open_concurrency: int = 16
    max_random_concurrency: int = 8
    acquire_timeout_s: float = 5.0
    search_timeout_s: float = 180.0
    open_timeout_s: float = 180.0
    random_timeout_s: float = 180.0
    link_mode: Literal["wikiid", "title"] = "wikiid"

    def runtime_config(self) -> WikiRuntimeConfig:
        return WikiRuntimeConfig(
            text_index_dir=self.text_index_dir,
            vector_index_dir=self.vector_index_dir,
            embedding_api_url=self.embedding_api_url,
            embedding_model=self.embedding_model,
            embedding_api_key=self.embedding_api_key,
            embedding_no_proxy=self.embedding_no_proxy,
            embedding_dimensions=self.embedding_dimensions,
            lazy_load_faiss=self.lazy_load_faiss,
            link_mode=self.link_mode,
        )


class SearchRequest(BaseModel):
    query: str = Field(description="Search query string.")
    top_k: int = Field(default=10, ge=1, description="Max results to return.")
    only_title: bool = Field(default=False, description="Only search article titles.")
    method: Literal["rrf", "keywords", "vector"] = Field(
        default="rrf",
        description='Search method: "rrf" (hybrid), "keywords" (BM25), "vector" (embedding).',
    )
    max_chars_per_result: int = Field(
        default=1024,
        ge=32,
        le=100_000,
        description="Snippet max length passed down to retrieval modules.",
    )
    link_mode: Literal["wikiid", "title"] | None = Field(
        default=None,
        description='Wiki link id mode: "wikiid" or "title". None uses server default.',
    )


class SearchResponse(BaseModel):
    content: str


class RandomRequest(BaseModel):
    mode: Literal["chunk", "page"] = Field(default="chunk")
    seed: int | None = Field(default=None)


class OpenURLRequest(BaseModel):
    url: str = Field(description="Wikipedia URL to open.")


class MarkdownResponse(BaseModel):
    content: str


def _tool_http_error(
    *,
    tool: str,
    exc: BaseException,
    meta: dict[str, Any],
) -> HTTPException:
    """Map tool failures to HTTP status codes + structured JSON detail."""
    message = str(exc)
    lowered = message.lower()
    if isinstance(exc, (ValueError, ValidationError)):
        status_code = 400
        code = "invalid_argument"
    elif is_timeout_exc(exc):
        if "overloaded" in lowered:
            status_code = 503
            code = "overloaded"
        else:
            status_code = 504
            code = "timeout"
    else:
        status_code = 503
        code = "backend_error"
    detail = {
        "status": "error",
        "tool": tool,
        "code": code,
        "message": message,
        "meta": {**meta, "location": exc_location(exc)},
        # Keep markdown for human debugging / older clients.
        "content": format_tool_error_md(
            tool=tool,
            code=code,
            message=message,
            meta={**meta, "location": exc_location(exc)},
        ),
    }
    return HTTPException(status_code=status_code, detail=detail)


async def _run_in_pool(
    pool: Executor,
    fn: Any,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    loop = asyncio.get_running_loop()
    call = functools.partial(fn, *args, **kwargs)
    return await loop.run_in_executor(pool, call)


async def _run_limited(
    *,
    sem: asyncio.Semaphore,
    name: str,
    acquire_timeout_s: float,
    timeout_s: float,
    make_coro: Callable[[], Awaitable[Any]],
) -> Any:
    try:
        await asyncio.wait_for(sem.acquire(), timeout=max(0.001, acquire_timeout_s))
    except asyncio.TimeoutError as e:
        raise TimeoutError(
            f"{name} overloaded: no worker slot available after {acquire_timeout_s:.1f}s"
        ) from e

    held = True

    def release_once() -> None:
        nonlocal held
        if held:
            held = False
            sem.release()

    # Process-pool work cannot be cancelled; keep the slot until the underlying
    # task finishes even if the asyncio waiter times out.
    task = asyncio.create_task(make_coro())
    try:
        if timeout_s > 0:
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(task), timeout=float(timeout_s)
                )
            except asyncio.TimeoutError as e:
                task.add_done_callback(lambda _t: release_once())
                raise TimeoutError(f"{name} timed out after {timeout_s:.1f}s") from e
        else:
            result = await task
    except TimeoutError:
        raise
    except Exception:
        if task.done():
            release_once()
        else:
            task.add_done_callback(lambda _t: release_once())
        raise
    else:
        release_once()
        return result


async def _search_rrf_async(
    *,
    req: SearchRequest,
    cfg: WikiRuntimeConfig,
    text_pool: ProcessPoolExecutor,
    vector_pool: ThreadPoolExecutor,
    link_mode: str,
) -> str:
    from tools.wikipedia_offline.search_rrf import (
        build_rrf_rank_maps,
        fuse_rrf_ranked_chunk_ids,
    )

    text_task = _run_in_pool(
        text_pool,
        search_text_rrf_candidates,
        query=req.query,
        top_k=req.top_k,
        max_chars_per_result=req.max_chars_per_result,
        cfg=cfg,
    )
    vector_task = _run_in_pool(
        vector_pool,
        search_vector_candidates,
        query=req.query,
        top_k=req.top_k,
        cfg=cfg,
    )
    text_results, vector_results = await asyncio.gather(text_task, vector_task)

    # Rank all candidates first, then hydrate the top of the fused ranking.
    text_ranks, vector_ranks, _snippets = build_rrf_rank_maps(
        text_results, vector_results
    )
    ranked = fuse_rrf_ranked_chunk_ids(text_ranks, vector_ranks)
    ranked_ids = [chunk_id for chunk_id, _score in ranked]
    # Over-fetch hydrate window so occasional lookup misses still fill top_k.
    hydrate_n = min(len(ranked_ids), max(1, int(req.top_k) * 3))
    chunk_infos = await _run_in_pool(
        text_pool,
        hydrate_chunk_infos,
        chunk_ids=ranked_ids[:hydrate_n],
        cfg=cfg,
    )
    return build_rrf_markdown(
        top_k=req.top_k,
        max_chars_per_result=req.max_chars_per_result,
        text_results=text_results,
        vector_results=vector_results,
        chunk_infos=chunk_infos,
        link_mode=link_mode,
        ranked_chunk_ids=ranked_ids,
    )


def create_app(args: Args) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.args = args
        app.state.cfg = args.runtime_config()
        app.state.text_pool = None
        app.state.vector_pool = None
        app.state.search_sem = asyncio.Semaphore(
            max(1, int(args.max_search_concurrency))
        )
        app.state.open_sem = asyncio.Semaphore(max(1, int(args.max_open_concurrency)))
        app.state.random_sem = asyncio.Semaphore(
            max(1, int(args.max_random_concurrency))
        )

        interval_s = float(
            (os.environ.get("REALWEB_MEM_LOG_INTERVAL_S") or "0").strip() or 0
        )
        hb_task: asyncio.Task | None = None
        if interval_s > 0:
            hb_task = asyncio.create_task(mem_heartbeat_task(interval_s))

        logger.info(
            "wikipedia_offline_server startup text_index_dir={} lazy_load_faiss={} "
            "text_workers={} start_method={}",
            args.text_index_dir,
            args.lazy_load_faiss,
            args.text_workers,
            args.worker_start_method,
        )

        if bool(args.embedding_auto_detect):
            from tools.wikipedia_offline.embedding_detect import detect_provider

            provider = detect_provider(
                args.embedding_model,
                local_api_url=args.embedding_api_url,
                local_api_key=args.embedding_api_key,
            )
            if provider is not None:
                args.embedding_api_url = provider.api_url
                args.embedding_api_key = provider.api_key
                args.embedding_model = provider.model
                args.embedding_no_proxy = provider.no_proxy
                args.embedding_dimensions = provider.dimensions
                app.state.cfg = args.runtime_config()
            else:
                logger.error(
                    "Embedding auto-detect found no live endpoint; "
                    "falling back to default {} (rrf/vector may fail)",
                    args.embedding_api_url,
                )

        start_method = str(args.worker_start_method).strip().lower()
        if start_method not in {"fork", "spawn", "forkserver"}:
            raise ValueError(
                f"Unsupported worker_start_method={args.worker_start_method!r}"
            )
        if start_method == "fork" and os.name != "posix":
            raise ValueError("worker_start_method='fork' is only supported on POSIX")

        cfg: WikiRuntimeConfig = app.state.cfg
        text_workers = max(1, int(args.text_workers))
        mp_ctx = mp.get_context(start_method)

        # Fork text workers before loading FAISS so they do not inherit the
        # ~tens-of-GB index mapping via copy-on-write.
        text_pool = ProcessPoolExecutor(
            max_workers=text_workers,
            mp_context=mp_ctx,
            initializer=text_worker_initializer,
            initargs=(cfg,),
        )
        text_warmup = await asyncio.gather(
            *[_run_in_pool(text_pool, worker_ping, "text") for _ in range(text_workers)]
        )

        # One shared FAISS index in this process; threads only for concurrent search.
        vector_runtime_preload(cfg)
        vector_pool = ThreadPoolExecutor(
            max_workers=max(1, int(args.max_search_concurrency)),
            thread_name_prefix="wiki-vector",
        )
        app.state.text_pool = text_pool
        app.state.vector_pool = vector_pool

        vector_warmup = await _run_in_pool(vector_pool, worker_ping, "vector")
        logger.info(
            "Worker warmup ok results={}",
            [*text_warmup, vector_warmup],
        )

        try:
            yield
        finally:
            if hb_task is not None:
                hb_task.cancel()
            if app.state.text_pool is not None:
                app.state.text_pool.shutdown(wait=True, cancel_futures=True)
            if app.state.vector_pool is not None:
                app.state.vector_pool.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="wikipedia_offline_server", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "search_available": getattr(app.state.search_sem, "_value", None),
            "open_available": getattr(app.state.open_sem, "_value", None),
            "random_available": getattr(app.state.random_sem, "_value", None),
        }

    @app.post("/search", response_model=SearchResponse)
    async def search(req: SearchRequest) -> SearchResponse:
        t0 = time.perf_counter()
        args_local: Args = app.state.args
        cfg: WikiRuntimeConfig = app.state.cfg
        method_n = normalize_search_method(req.method)
        is_rrf = (not bool(req.only_title)) and method_n == "rrf"
        use_text_pool = bool(req.only_title) or method_n == "keywords"
        pool: Executor = app.state.text_pool if use_text_pool else app.state.vector_pool
        pool_name = "rrf-split" if is_rrf else ("text" if use_text_pool else "vector")
        link_mode = normalize_link_mode(req.link_mode or args_local.link_mode)
        try:

            async def run_search() -> str:
                if is_rrf:
                    return await _search_rrf_async(
                        req=req,
                        cfg=cfg,
                        text_pool=app.state.text_pool,
                        vector_pool=app.state.vector_pool,
                        link_mode=link_mode,
                    )
                return await _run_in_pool(
                    pool,
                    search_markdown,
                    query=req.query,
                    top_k=req.top_k,
                    only_title=req.only_title,
                    method=req.method,
                    max_chars_per_result=req.max_chars_per_result,
                    cfg=cfg,
                    link_mode=link_mode,
                )

            md = await _run_limited(
                sem=app.state.search_sem,
                name="Search",
                acquire_timeout_s=float(args_local.acquire_timeout_s),
                timeout_s=float(args_local.search_timeout_s),
                make_coro=run_search,
            )
            dur_ms = int((time.perf_counter() - t0) * 1000)
            logger.info(
                "Search ok query=[{}] top_k={} only_title={} method={} pool={} dur_ms={} bytes={}",
                req.query,
                req.top_k,
                bool(req.only_title),
                "title" if bool(req.only_title) else method_n,
                pool_name,
                dur_ms,
                len(md.encode("utf-8", errors="ignore")),
            )
            return SearchResponse(content=md)
        except Exception as e:
            raise _tool_http_error(
                tool="Search",
                exc=e,
                meta={
                    "query": req.query,
                    "top_k": req.top_k,
                    "method": req.method,
                },
            ) from e

    @app.post("/random", response_model=MarkdownResponse)
    async def random_endpoint(req: RandomRequest) -> MarkdownResponse:
        t0 = time.perf_counter()
        args_local: Args = app.state.args
        cfg: WikiRuntimeConfig = app.state.cfg
        try:
            md = await _run_limited(
                sem=app.state.random_sem,
                name="Random",
                acquire_timeout_s=float(args_local.acquire_timeout_s),
                timeout_s=float(args_local.random_timeout_s),
                make_coro=lambda: _run_in_pool(
                    app.state.text_pool,
                    random_page_or_chunk,
                    mode=req.mode,
                    seed=req.seed,
                    cfg=cfg,
                ),
            )
            dur_ms = int((time.perf_counter() - t0) * 1000)
            logger.info(
                "Random ok mode={} seed={} dur_ms={} bytes={}",
                req.mode,
                req.seed,
                dur_ms,
                len(md.encode("utf-8", errors="ignore")),
            )
            return MarkdownResponse(content=md)
        except Exception as e:
            raise _tool_http_error(
                tool="Random",
                exc=e,
                meta={"mode": req.mode, "seed": req.seed},
            ) from e

    @app.post("/open_url", response_model=MarkdownResponse)
    async def open_url_endpoint(req: OpenURLRequest) -> MarkdownResponse:
        t0 = time.perf_counter()
        args_local: Args = app.state.args
        cfg: WikiRuntimeConfig = app.state.cfg
        try:
            md = await _run_limited(
                sem=app.state.open_sem,
                name="OpenURL",
                acquire_timeout_s=float(args_local.acquire_timeout_s),
                timeout_s=float(args_local.open_timeout_s),
                make_coro=lambda: _run_in_pool(
                    app.state.text_pool,
                    open_url,
                    (req.url or "").strip(),
                    cfg,
                ),
            )
            dur_ms = int((time.perf_counter() - t0) * 1000)
            logger.info(
                "OpenURL ok url={} dur_ms={} bytes={}",
                req.url,
                dur_ms,
                len(md.encode("utf-8", errors="ignore")),
            )
            return MarkdownResponse(content=md)
        except Exception as e:
            raise _tool_http_error(
                tool="OpenURL",
                exc=e,
                meta={"url": req.url},
            ) from e

    return app


def main() -> None:
    args = tyro.cli(Args, use_underscores=True)
    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = f"logs/{Path(__file__).stem}-{start_time}.log"
    setup_loguru(level=args.log_level, intercept_stdlib=True, log_file=log_file)
    logger.info(
        "wikipedia_offline_server http starting host={} port={}",
        args.host,
        args.port,
    )
    uvicorn.run(
        create_app(args),
        host=args.host,
        port=int(args.port),
        log_level=str(args.log_level).lower(),
    )


if __name__ == "__main__":
    main()
