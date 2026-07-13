"""Compatibility barrel for older ``from serving.common import ...`` call sites.

Prefer importing from ``serving.router_app`` / ``serving.tmux`` directly in new code.
"""

from __future__ import annotations

from serving.router_app import (
    HOP_BY_HOP_HEADERS,
    MODEL_VALIDATE_PATHS,
    RouterState,
    UpstreamRetryConfig,
    create_router_app,
    proxy_to_backend,
    register_router_routes,
    run_router_server,
)
from serving.tmux import (
    BackendLike,
    TmuxRegistry,
    backend_base_url,
    detect_num_gpus,
    kill_tmux_session,
    kill_tmux_sessions_by_prefix,
    read_log_tail,
    run_capture,
    shell_cmd_with_gpu_and_tee,
    start_backend_in_tmux,
    stop_backend,
    tmux_safe_name,
    tmux_session_exists,
    wait_backend_ready,
)

__all__ = [
    "HOP_BY_HOP_HEADERS",
    "MODEL_VALIDATE_PATHS",
    "BackendLike",
    "RouterState",
    "TmuxRegistry",
    "UpstreamRetryConfig",
    "backend_base_url",
    "create_router_app",
    "detect_num_gpus",
    "kill_tmux_session",
    "kill_tmux_sessions_by_prefix",
    "proxy_to_backend",
    "read_log_tail",
    "register_router_routes",
    "run_capture",
    "run_router_server",
    "shell_cmd_with_gpu_and_tee",
    "start_backend_in_tmux",
    "stop_backend",
    "tmux_safe_name",
    "tmux_session_exists",
    "wait_backend_ready",
]
