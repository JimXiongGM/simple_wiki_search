from __future__ import annotations

import asyncio
import atexit
import itertools
import json
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from loguru import logger

from serving.tmux import (
    BackendLike,
    TmuxRegistry,
    start_backend_in_tmux,
    stop_backend,
    wait_backend_ready,
)

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}

MODEL_VALIDATE_PATHS = frozenset(
    {
        "v1/chat/completions",
        "v1/completions",
        "v1/embeddings",
        "v1/responses",
    }
)


class RouterState:
    def __init__(
        self,
        backends: list[Any],
        served_model_name: str,
        api_key: str | None = None,
    ):
        self.backends = backends
        self._cycle = itertools.cycle(backends)
        self.served_model_name = served_model_name
        self.api_key = api_key
        self.http_client: httpx.AsyncClient | None = None

    def next_backend(self) -> Any:
        return next(self._cycle)


@dataclass
class UpstreamRetryConfig:
    retries: int = 2
    backoff_s: float = 0.5
    max_backoff_s: float = 5.0


async def proxy_to_backend(
    *,
    method: str,
    target_url: str,
    headers: dict[str, str],
    body: bytes,
    params: Any,
    client: httpx.AsyncClient,
    retry: UpstreamRetryConfig,
) -> Response:
    last_exc: Exception | None = None
    resp: httpx.Response | None = None
    try:
        for attempt in range(retry.retries + 1):
            try:
                req = client.build_request(
                    method,
                    target_url,
                    headers=headers,
                    content=body,
                    params=params,
                )
                resp = await client.send(req, stream=False)
                if resp.status_code >= 500 and attempt < retry.retries:
                    await resp.aclose()
                    backoff = min(retry.max_backoff_s, retry.backoff_s * (2**attempt))
                    logger.warning(
                        f"Upstream {resp.status_code} ({target_url}), "
                        f"retrying in {backoff:.2f}s "
                        f"(attempt {attempt + 1}/{retry.retries})"
                    )
                    await asyncio.sleep(backoff)
                    continue
                break
            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.RemoteProtocolError,
            ) as e:
                last_exc = e
                if attempt >= retry.retries:
                    raise
                backoff = min(retry.max_backoff_s, retry.backoff_s * (2**attempt))
                logger.warning(
                    f"Upstream transient error ({target_url}) [{type(e).__name__}] "
                    f"repr={repr(e)}; retrying in {backoff:.2f}s "
                    f"(attempt {attempt + 1}/{retry.retries})"
                )
                await asyncio.sleep(backoff)
        else:
            raise last_exc or RuntimeError("unreachable")
    except Exception as e:
        err_type = type(e).__name__
        err_msg = (str(e) or "").strip()
        err_repr = repr(e)
        logger.error(
            f"Upstream Error ({target_url}) [{err_type}] msg={err_msg!r} repr={err_repr}"
        )
        status = 504 if isinstance(e, httpx.TimeoutException) else 502
        return Response(
            status_code=status,
            content=f"Upstream Error [{err_type}]: {err_msg or err_repr}",
        )

    assert resp is not None
    resp_headers = {
        k: v for k, v in resp.headers.items() if k.lower() not in HOP_BY_HOP_HEADERS
    }
    content = resp.content
    await resp.aclose()
    return Response(
        content=content,
        status_code=resp.status_code,
        headers=resp_headers,
        media_type=resp.headers.get("content-type"),
    )


def _extract_request_key(request: Request, body_json: Any | None) -> str:
    auth = request.headers.get("authorization", "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    if request.headers.get("x-api-key", "").strip():
        return request.headers["x-api-key"].strip()
    if isinstance(body_json, dict) and isinstance(body_json.get("key"), str):
        return body_json["key"].strip()
    return ""


def register_router_routes(
    app: FastAPI,
    *,
    retry: UpstreamRetryConfig,
    require_api_key: bool = False,
    validate_model: bool = False,
) -> None:
    @app.get("/ping")
    async def ping(request: Request):
        state: RouterState = request.app.state.router
        client = state.http_client

        async def _check_one(b: BackendLike) -> dict[str, Any]:
            last_err: str | None = None
            headers = b.auth_headers() or None
            for path in ("/ping", "/v1/models"):
                try:
                    r = await client.get(
                        f"{b.base_url}{path}", headers=headers, timeout=2.0
                    )
                    if r.status_code == 200:
                        return {"base_url": b.base_url, "ok": True, "path": path}
                except Exception as e:
                    last_err = str(e)
            return {"base_url": b.base_url, "ok": False, "error": last_err or "unknown"}

        results = await asyncio.gather(*(_check_one(b) for b in state.backends))
        ok = all(r.get("ok") for r in results)
        return JSONResponse(
            status_code=200 if ok else 503, content={"ok": ok, "backends": results}
        )

    @app.get("/backend-info")
    async def backend_info(request: Request):
        state: RouterState = request.app.state.router
        return {
            "model_name": state.served_model_name,
            "count": len(state.backends),
            "backends": [
                {
                    "gpu_id": b.gpu_id,
                    "port": b.port,
                    "base_url": b.base_url,
                    "tmux_session": b.tmux_session,
                    **(
                        {"log_file": b.log_file}
                        if getattr(b, "log_file", None) is not None
                        else {}
                    ),
                }
                for b in state.backends
            ],
        }

    @app.api_route(
        "/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    )
    async def proxy(full_path: str, request: Request):
        state: RouterState = request.app.state.router
        backend = state.next_backend()
        target_url = f"{backend.base_url}/{full_path}"

        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"-> GPU {backend.gpu_id} (port {backend.port})"
        )

        headers = dict(request.headers)
        headers.pop("host", None)
        headers = {
            k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP_HEADERS
        }

        body = await request.body()
        body_json: Any | None = None
        if (
            (require_api_key or validate_model)
            and "application/json" in request.headers.get("content-type", "").lower()
            and body
        ):
            try:
                body_json = json.loads(body)
            except json.JSONDecodeError:
                body_json = None

        if require_api_key:
            assert state.api_key is not None
            req_key = _extract_request_key(request, body_json)
            if req_key != state.api_key:
                return JSONResponse(
                    status_code=401,
                    content={"error": "invalid key"},
                )
            headers["authorization"] = f"Bearer {state.api_key}"
            headers.pop("x-api-key", None)

        if validate_model and full_path in MODEL_VALIDATE_PATHS:
            req_model = ""
            if isinstance(body_json, dict) and isinstance(body_json.get("model"), str):
                req_model = body_json["model"].strip()
            if req_model != state.served_model_name:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": f"invalid model: expected {state.served_model_name!r}"
                    },
                )

        return await proxy_to_backend(
            method=request.method,
            target_url=target_url,
            headers=headers,
            body=body,
            params=request.query_params,
            client=state.http_client,
            retry=retry,
        )


BuildBackendsFn = Callable[[], Awaitable[list[Any]]]


def create_router_app(
    *,
    title: str,
    registry: TmuxRegistry,
    build_backends: BuildBackendsFn,
    served_model_name: str,
    api_key: str | None,
    kill_existing: bool,
    wait_timeout_s: int,
    upstream_timeout_s: float,
    upstream_connect_timeout_s: float,
    retry: UpstreamRetryConfig,
    router_host: str,
    start_port: int,
    require_api_key: bool = False,
    validate_model: bool = False,
    log_label: str = "SGLang server",
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        backends: list[Any] = []
        state: RouterState | None = None
        try:
            backends = await build_backends()
            if not backends:
                raise RuntimeError("No backends configured.")

            await asyncio.gather(
                *(
                    start_backend_in_tmux(
                        b,
                        kill_existing=kill_existing,
                        registry=registry,
                        log_label=log_label,
                    )
                    for b in backends
                )
            )

            async with httpx.AsyncClient() as client:
                ok_list = await asyncio.gather(
                    *(
                        wait_backend_ready(b, client, timeout_s=wait_timeout_s)
                        for b in backends
                    )
                )
            if not all(ok_list):
                logger.error("Some backends failed to start. Shutting down...")
                raise RuntimeError("backend startup failed")

            state = RouterState(
                backends=backends,
                served_model_name=served_model_name,
                api_key=api_key,
            )
            state.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    upstream_timeout_s, connect=upstream_connect_timeout_s
                ),
                limits=httpx.Limits(
                    max_keepalive_connections=100, max_connections=1000
                ),
            )
            app.state.router = state

            logger.success("All backends are ready.")
            logger.success(f"Router: http://{router_host}:{start_port}")
            logger.success("Backend pool:")
            for b in backends:
                logger.success(f"  {b.base_url}  (tmux: {b.tmux_session})")

            yield
        finally:
            logger.info("Shutting down router and backends...")
            if state is not None and state.http_client is not None:
                try:
                    await state.http_client.aclose()
                except Exception:
                    pass
            if backends:
                await asyncio.gather(*(stop_backend(b, registry) for b in backends))

    app = FastAPI(title=title, version="1.0.0", lifespan=lifespan)
    register_router_routes(
        app,
        retry=retry,
        require_api_key=require_api_key,
        validate_model=validate_model,
    )
    return app


class _RouterUvicornServer(uvicorn.Server):
    """Kill tmux backends on SIGINT/SIGTERM before uvicorn's graceful shutdown."""

    def __init__(self, config: uvicorn.Config, registry: TmuxRegistry):
        super().__init__(config)
        self._registry = registry

    def handle_exit(self, sig: int, frame) -> None:
        logger.warning(f"Received signal {sig}, killing started tmux sessions...")
        killed = self._registry.kill_all()
        if killed:
            logger.info(f"Killed tmux sessions: {', '.join(sorted(set(killed)))}")
        super().handle_exit(sig, frame)


def run_router_server(
    app: FastAPI,
    *,
    host: str,
    port: int,
    registry: TmuxRegistry,
) -> None:
    def _cleanup_tmux_on_exit() -> None:
        killed = registry.kill_all()
        if killed:
            logger.info(
                f"atexit cleaned tmux sessions: {', '.join(sorted(set(killed)))}"
            )

    atexit.register(_cleanup_tmux_on_exit)

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        timeout_graceful_shutdown=5,
    )
    server = _RouterUvicornServer(config, registry)
    try:
        server.run()
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt, cleaning up tmux sessions...")
        _cleanup_tmux_on_exit()
    finally:
        _cleanup_tmux_on_exit()
