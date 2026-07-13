from __future__ import annotations

import asyncio
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Protocol

import httpx
from loguru import logger


def run_capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, text=True, capture_output=True)


def detect_num_gpus() -> int:
    """Return GPU count from nvidia-smi. Missing/failed nvidia-smi yields 0."""
    try:
        p = run_capture(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    except FileNotFoundError:
        return 0
    if p.returncode != 0:
        return 0
    lines = [x.strip() for x in (p.stdout or "").splitlines() if x.strip()]
    return len(lines)


def tmux_safe_name(name: str) -> str:
    """Dots/colons in model names break tmux target parsing."""
    return re.sub(r"[.:]", "_", name)


def _tmux_session_candidates(name: str) -> list[str]:
    safe = tmux_safe_name(name)
    if safe == name:
        return [name]
    return [safe, name]


def tmux_session_exists(name: str) -> bool:
    try:
        for candidate in _tmux_session_candidates(name):
            p = run_capture(["tmux", "has-session", "-t", candidate])
            if p.returncode == 0:
                return True
    except FileNotFoundError as e:
        raise RuntimeError(
            "`tmux` command not found. Please install it: sudo apt-get install -y tmux"
        ) from e
    return False


def kill_tmux_session(name: str) -> bool:
    for candidate in _tmux_session_candidates(name):
        if not tmux_session_exists(candidate):
            continue
        p = run_capture(["tmux", "kill-session", "-t", candidate])
        if p.returncode == 0:
            return True
        logger.warning(
            f"Failed to kill tmux session {candidate!r}: "
            f"{(p.stderr or p.stdout or '').strip()}"
        )
    return False


def kill_tmux_sessions_by_prefix(prefix: str) -> list[str]:
    try:
        p = run_capture(["tmux", "list-sessions", "-F", "#{session_name}"])
    except FileNotFoundError:
        return []
    if p.returncode != 0:
        return []
    killed: list[str] = []
    for line in (p.stdout or "").splitlines():
        name = line.strip()
        if name.startswith(prefix) and kill_tmux_session(name):
            killed.append(name)
    return killed


class TmuxRegistry:
    """Tracks tmux sessions owned by a router process for cleanup."""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.started: set[str] = set()
        self.pending: set[str] = set()

    def mark_pending(self, name: str) -> None:
        self.pending.add(name)

    def mark_started(self, name: str) -> None:
        self.started.add(name)

    def discard(self, name: str) -> None:
        self.started.discard(name)
        self.pending.discard(name)

    def owns(self, name: str) -> bool:
        return name in self.started or name in self.pending

    def kill_all(self) -> list[str]:
        names = set(self.started) | set(self.pending)
        killed: list[str] = []
        for name in sorted(names):
            try:
                if kill_tmux_session(name):
                    killed.append(name)
            except Exception as exc:
                logger.warning(f"Failed to kill tmux session {name!r}: {exc}")
            self.discard(name)
        killed.extend(kill_tmux_sessions_by_prefix(self.prefix))
        return killed


def shell_cmd_with_gpu_and_tee(gpu_id: int, cmd: list[str], log_file: str) -> str:
    venv_bin = str(Path(sys.executable).resolve().parent)
    cmd_str = " ".join(shlex.quote(x) for x in cmd)
    log_path = shlex.quote(log_file)
    return (
        f"export CUDA_VISIBLE_DEVICES={shlex.quote(str(gpu_id))}; "
        f"export PATH={shlex.quote(venv_bin)}:$PATH; "
        f"{cmd_str} 2>&1 | tee {log_path}"
    )


def read_log_tail(log_file: str, lines: int = 80) -> str:
    path = Path(log_file)
    if not path.exists():
        return f"(log file not found: {log_file})"
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        return f"(failed to read log file {log_file}: {e})"
    if not content:
        return "(log file is empty)"
    return "\n".join(content[-lines:])


class BackendLike(Protocol):
    gpu_id: int
    port: int
    backend_host: str
    tmux_session: str
    log_file: str

    @property
    def base_url(self) -> str: ...

    def auth_headers(self) -> dict[str, str]: ...

    def build_shell_cmd(self) -> str: ...


def backend_base_url(backend_host: str, port: int) -> str:
    return f"http://{backend_host}:{port}"


async def start_backend_in_tmux(
    backend: BackendLike,
    *,
    kill_existing: bool,
    registry: TmuxRegistry,
    log_label: str,
) -> None:
    if kill_existing and tmux_session_exists(backend.tmux_session):
        logger.warning(
            f"Found existing tmux session {backend.tmux_session}, killing it first..."
        )
        kill_tmux_session(backend.tmux_session)
        await asyncio.sleep(1.0)

    shell_cmd = backend.build_shell_cmd()
    tmux_cmd = [
        "tmux",
        "new-session",
        "-d",
        "-s",
        backend.tmux_session,
        "bash",
        "-lc",
        shell_cmd,
    ]

    Path(backend.log_file).parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Starting {log_label} gpu={backend.gpu_id} port={backend.port} "
        f"log={backend.log_file}"
    )
    logger.debug(f"Shell command: {shell_cmd}")

    def _run() -> None:
        subprocess.run(tmux_cmd, check=True, text=True, capture_output=True)

    try:
        await asyncio.to_thread(_run)
    except FileNotFoundError:
        raise RuntimeError(
            "`tmux` command not found. Please install it: sudo apt-get install -y tmux"
        ) from None

    registry.mark_started(backend.tmux_session)


async def wait_backend_ready(
    backend: BackendLike,
    client: httpx.AsyncClient,
    timeout_s: int,
) -> bool:
    urls = [f"{backend.base_url}/ping", f"{backend.base_url}/v1/models"]
    headers = backend.auth_headers()
    start_time = time.time()
    while time.time() - start_time < timeout_s:
        try:
            for url in urls:
                r = await client.get(url, headers=headers or None, timeout=2.0)
                if r.status_code == 200:
                    logger.success(f"Backend ready: {backend.base_url}")
                    return True
        except Exception:
            pass
        await asyncio.sleep(1.0)
    logger.error(f"Backend timed out: {backend.base_url}")
    logger.error(
        f"Backend startup logs (tmux={backend.tmux_session}, gpu={backend.gpu_id}):\n"
        f"{read_log_tail(backend.log_file)}"
    )
    return False


async def stop_backend(backend: BackendLike, registry: TmuxRegistry) -> None:
    if not registry.owns(backend.tmux_session):
        return
    logger.info(f"Stopping backend tmux={backend.tmux_session} ...")

    def _run() -> None:
        kill_tmux_session(backend.tmux_session)

    await asyncio.to_thread(_run)
    registry.discard(backend.tmux_session)
