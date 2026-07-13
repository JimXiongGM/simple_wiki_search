"""Automatic embedding endpoint detection.

Probe candidates in priority order ``local → aihubmix`` and return the first
available provider. Both expose OpenAI-compatible ``/v1/embeddings``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests
from loguru import logger

from settings import DEFAULT_EMBEDDING_API_KEY, DEFAULT_EMBEDDING_API_URL


@dataclass(frozen=True)
class EmbeddingProvider:
    """Complete description of a working embedding endpoint."""

    name: str
    api_url: str  # Prefix before /v1; calls use {api_url}/v1/embeddings
    api_key: str
    model: str
    no_proxy: bool  # True=direct (trust_env=False)
    dimensions: int | None = None


def _env_https_proxy() -> str | None:
    """Return first configured HTTP(S) proxy URL, if any."""
    for name in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"):
        value = os.environ.get(name)
        if value and str(value).strip():
            return str(value).strip()
    return None


def build_candidates(
    model: str,
    *,
    local_api_url: str = DEFAULT_EMBEDDING_API_URL,
    local_api_key: str = DEFAULT_EMBEDDING_API_KEY,
) -> list[EmbeddingProvider]:
    """Build candidates in order ``local → aihubmix``.

    ``local_api_url`` / ``local_api_key`` come from CLI/config so a temporary
    override (e.g. LAN IP) is probed first; aihubmix is only a fallback.
    """
    cands: list[EmbeddingProvider] = [
        EmbeddingProvider(
            "local",
            local_api_url.rstrip("/"),
            local_api_key,
            model,
            no_proxy=True,
        ),
    ]

    aihubmix_key = os.environ.get("AIHUBMIX_EMBEDDING_API_KEY")
    if aihubmix_key:
        cands.append(
            EmbeddingProvider(
                "aihubmix",
                "https://aihubmix.com",
                aihubmix_key,
                # Platform id for Qwen3-Embedding-0.6B weights.
                "qwen3-embedding-0.6b",
                # Reach via env proxy; direct route is often unreachable.
                no_proxy=False,
            )
        )
    else:
        logger.info("Skip aihubmix candidate: AIHUBMIX_EMBEDDING_API_KEY not set")

    return cands


def _probe(p: EmbeddingProvider, timeout_s: float) -> bool:
    """Send a minimal embeddings request; success if a vector is returned."""
    try:
        body: dict = {
            "model": p.model,
            "input": ["ping"],
            "encoding_format": "float",
        }
        if p.dimensions is not None:
            body["dimensions"] = int(p.dimensions)
        with requests.Session() as s:
            # When using proxy, pass it explicitly so NO_PROXY cannot bypass it
            # (aihubmix.com is sometimes listed in NO_PROXY).
            proxies = None
            if p.no_proxy:
                s.trust_env = False
            else:
                s.trust_env = False
                proxy = _env_https_proxy()
                if proxy:
                    proxies = {"http": proxy, "https": proxy}
                else:
                    s.trust_env = True
            r = s.post(
                f"{p.api_url}/v1/embeddings",
                headers={"Authorization": f"Bearer {p.api_key}"},
                json=body,
                timeout=timeout_s,
                proxies=proxies,
            )
            r.raise_for_status()
            data = (r.json() or {}).get("data") or []
            return bool(data and data[0].get("embedding"))
    except Exception as e:
        logger.warning("Embedding probe failed [{}] {}: {}", p.name, p.api_url, e)
        return False


def detect_provider(
    model: str,
    timeout_s: float = 30.0,
    *,
    local_api_url: str = DEFAULT_EMBEDDING_API_URL,
    local_api_key: str = DEFAULT_EMBEDDING_API_KEY,
) -> EmbeddingProvider | None:
    """Probe candidates in order; return the first hit, or None."""
    for p in build_candidates(
        model,
        local_api_url=local_api_url,
        local_api_key=local_api_key,
    ):
        logger.info("Probing embedding endpoint [{}] {} ...", p.name, p.api_url)
        if _probe(p, timeout_s):
            logger.success(
                "Embedding endpoint selected: [{}] {} (model={})",
                p.name,
                p.api_url,
                p.model,
            )
            return p
    logger.error("No embedding endpoint available after probing all candidates")
    return None
