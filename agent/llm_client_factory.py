from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import httpx
import openai

DEFAULT_SERVER_URL = "http://127.0.0.1:19000"

_clients: dict[tuple[str, str, float], openai.OpenAI] = {}
_lock = threading.Lock()

_PROVIDER_ENDPOINTS: dict[str, tuple[str, str]] = {
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
    "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1"),
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com/v1"),
}


@dataclass(frozen=True)
class LlmEndpoint:
    api_key: str
    base_url: str
    trust_env: bool = False


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    if value and str(value).strip():
        return str(value).strip()
    return None


def resolve_llm_endpoint(
    model_name: str | None = None,
    *,
    server_url: str = DEFAULT_SERVER_URL,
    provider: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> LlmEndpoint:
    """Resolve API endpoint from explicit provider/base_url, not model-name heuristics.

    Priority:
    1. Explicit ``base_url`` (+ ``api_key`` / env)
    2. Explicit ``provider``
    3. Local ``server_url`` (default)
    """
    explicit_base = (base_url or "").rstrip("/")
    explicit_provider = (provider or "").strip().lower()
    explicit_key = api_key

    if explicit_base:
        if explicit_base.endswith("/v1"):
            resolved = explicit_base
        else:
            resolved = explicit_base + "/v1"
        key = explicit_key or _env("OPENAI_API_KEY") or "simple_wiki"
        # Remote HTTPS typically needs proxy; local loopback should not.
        trust_env = not resolved.startswith(
            ("http://127.0.0.1", "http://localhost", "https://127.0.0.1")
        )
        return LlmEndpoint(api_key=key, base_url=resolved, trust_env=trust_env)

    if explicit_provider:
        if explicit_provider in {"local", "sglang", "vllm"}:
            key = explicit_key or "simple_wiki"
            return LlmEndpoint(
                api_key=key,
                base_url=str(server_url).rstrip("/") + "/v1",
                trust_env=False,
            )
        if explicit_provider not in _PROVIDER_ENDPOINTS:
            raise RuntimeError(
                f"unknown provider={explicit_provider!r}; "
                f"expected one of {sorted(_PROVIDER_ENDPOINTS) + ['local', 'sglang', 'vllm']}"
            )
        env_key, default_url = _PROVIDER_ENDPOINTS[explicit_provider]
        key = explicit_key or _env(env_key)
        if not key:
            raise RuntimeError(
                f"missing API key for provider={explicit_provider!r} "
                f"(expected {env_key} or api_key argument)"
            )
        return LlmEndpoint(api_key=key, base_url=default_url, trust_env=True)

    # Default: treat as local OpenAI-compatible server. Model name is only an ID.
    _ = model_name  # kept for call-site compatibility
    key = explicit_key or "simple_wiki"
    return LlmEndpoint(
        api_key=key,
        base_url=str(server_url).rstrip("/") + "/v1",
        trust_env=False,
    )


def create_llm_client(
    model_name: str | None = None,
    *,
    server_url: str = DEFAULT_SERVER_URL,
    timeout: float = 600.0,
    provider: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> openai.OpenAI:
    """Create (or reuse) an OpenAI-compatible client from explicit endpoint config."""
    endpoint = resolve_llm_endpoint(
        model_name,
        server_url=server_url,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
    )
    cache_key = (endpoint.base_url, endpoint.api_key, float(timeout))
    with _lock:
        client = _clients.get(cache_key)
        if client is None:
            client = openai.OpenAI(
                api_key=endpoint.api_key,
                base_url=endpoint.base_url,
                timeout=timeout,
                http_client=httpx.Client(timeout=timeout, trust_env=endpoint.trust_env),
            )
            _clients[cache_key] = client
        return client
