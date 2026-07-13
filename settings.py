"""Central defaults shared across the repo.

Defaults match the historical hard-coded values so behavior stays identical;
call sites should import from here instead of duplicating string literals.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IndexPaths:
    text_index_dir: str = (
        "data/database/wikipedia-index/enwiki-20260601/tantivy/chunk-1536"
    )
    vector_index_dir: str = (
        "data/database/wikipedia-index/enwiki-20260601/faiss/"
        "hnsw-Qwen3-Embedding-0.6B-chunk-1536-fp16"
    )


INDEX_PATHS = IndexPaths()

# Tool / MCP HTTP defaults
DEFAULT_TOOL_HOST = "127.0.0.1"
DEFAULT_TOOL_PORT = 11536
DEFAULT_TOOL_BASE_URL = f"http://127.0.0.1:{DEFAULT_TOOL_PORT}"

# Embedding endpoint default (local embedding service)
DEFAULT_EMBEDDING_API_URL = "http://127.0.0.1:17000"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_EMBEDDING_API_KEY = "simple_wiki"

# Agent LLM router default
DEFAULT_LLM_SERVER_URL = "http://127.0.0.1:19000"
