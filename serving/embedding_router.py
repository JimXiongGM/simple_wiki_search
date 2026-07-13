from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from serving.router_app import UpstreamRetryConfig, create_router_app, run_router_server
from serving.tmux import (
    TmuxRegistry,
    backend_base_url,
    detect_num_gpus,
    run_capture,
    shell_cmd_with_gpu_and_tee,
    tmux_safe_name,
)
from utils.logging import setup_loguru


def get_gpu_total_mib(gpu_id: int) -> int:
    try:
        p = run_capture(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
    except FileNotFoundError as e:
        raise RuntimeError("nvidia-smi not found") from e
    if p.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi query failed for GPU {gpu_id}: {p.stderr.strip()}"
        )
    s = (p.stdout or "").strip().replace(" ", "")
    if not s.isdigit():
        raise RuntimeError(f"failed to parse GPU {gpu_id} memory.total: {s!r}")
    return int(s)


def parse_memory_size(size_str: str) -> int:
    """Parse memory size string like "10GB", "8G", "512MB" to MiB."""
    size_str = size_str.strip().upper()
    match = re.match(r"^(\d+(?:\.\d+)?)\s*([GM]B?)$", size_str)
    if not match:
        raise ValueError(
            f"Invalid memory size format: {size_str}. Expected format like '10GB' or '8G'"
        )
    value = float(match.group(1))
    unit = match.group(2)
    if unit.startswith("G"):
        return int(value * 1024)
    if unit.startswith("M"):
        return int(value)
    raise ValueError(f"Unknown unit: {unit}")


def calc_mem_fraction_static(total_mib: int, target_mib: int = 4096) -> float:
    if total_mib <= 0:
        return 0.1
    frac = target_mib / float(total_mib)
    frac = max(0.01, min(0.99, frac))
    return float(f"{frac:.4f}")


@dataclass(frozen=True)
class BackendInstance:
    gpu_id: int
    port: int
    model_name: str
    backend_host: str
    tmux_session: str
    log_file: str
    mem_fraction_static: float
    dtype: str

    @property
    def base_url(self) -> str:
        return backend_base_url(self.backend_host, self.port)

    def auth_headers(self) -> dict[str, str]:
        return {}

    def build_shell_cmd(self) -> str:
        cmd = [
            "sglang",
            "serve",
            "--model-path",
            self.model_name,
            "--is-embedding",
            "--port",
            str(self.port),
            "--host",
            self.backend_host,
            "--mem-fraction-static",
            str(self.mem_fraction_static),
            "--dtype",
            self.dtype,
        ]
        return shell_cmd_with_gpu_and_tee(self.gpu_id, cmd, self.log_file)


@dataclass
class Args:
    model_name: str
    start_port: int = 17000
    served_model_name: str | None = None

    router_host: str = "0.0.0.0"
    backend_host: str = "127.0.0.1"

    tmux_prefix: str = "embed-"
    kill_existing: bool = True
    wait_timeout_s: int = 240

    target_mib: str = "4GB"
    """Keep embedding backends light (~4GB)."""
    dtype: str = "bfloat16"

    upstream_connect_timeout_s: float = 10.0
    upstream_timeout_s: float = 300.0

    upstream_retries: int = 2
    upstream_retry_backoff_s: float = 0.5
    upstream_retry_max_backoff_s: float = 5.0

    log_level: str = "INFO"


def create_app(args: Args, registry: TmuxRegistry):
    served_model_name = (args.served_model_name or args.model_name).strip()

    async def build_backends() -> list[BackendInstance]:
        num_gpus = detect_num_gpus()
        if num_gpus <= 0:
            raise RuntimeError("No GPU found (nvidia-smi).")
        logger.info(f"Found {num_gpus} GPUs")
        if not served_model_name:
            raise RuntimeError("--model_name/--served_model_name is empty")
        logger.info(f"Model: {args.model_name}")

        target_mib_int = parse_memory_size(args.target_mib)
        backend_log_dir = Path("logs") / "embedding_backends"
        backend_log_dir.mkdir(parents=True, exist_ok=True)

        mname = args.model_name.split("/")[-1]
        backends: list[BackendInstance] = []
        for gpu_id in range(num_gpus):
            total_mib = get_gpu_total_mib(gpu_id)
            mem_frac = calc_mem_fraction_static(total_mib, target_mib=target_mib_int)
            # router occupies args.start_port; backends start from args.start_port + 1
            port = args.start_port + 1 + gpu_id
            tmux_session = f"{args.tmux_prefix}{gpu_id}-{tmux_safe_name(mname)}"
            inst = BackendInstance(
                gpu_id=gpu_id,
                port=port,
                model_name=args.model_name,
                backend_host=args.backend_host,
                tmux_session=tmux_session,
                log_file=str(backend_log_dir / f"{mname}-gpu{gpu_id}.log"),
                mem_fraction_static=mem_frac,
                dtype=args.dtype,
            )
            registry.mark_pending(inst.tmux_session)
            backends.append(inst)
            logger.info(
                f"GPU {gpu_id}: total={total_mib}MiB, "
                f"mem-fraction-static={mem_frac:.4f}, port={port}"
            )
        return backends

    return create_router_app(
        title="embedding_router",
        registry=registry,
        build_backends=build_backends,
        served_model_name=served_model_name,
        api_key=None,
        kill_existing=args.kill_existing,
        wait_timeout_s=args.wait_timeout_s,
        upstream_timeout_s=args.upstream_timeout_s,
        upstream_connect_timeout_s=args.upstream_connect_timeout_s,
        retry=UpstreamRetryConfig(
            retries=args.upstream_retries,
            backoff_s=args.upstream_retry_backoff_s,
            max_backoff_s=args.upstream_retry_max_backoff_s,
        ),
        router_host=args.router_host,
        start_port=args.start_port,
        require_api_key=False,
        validate_model=False,
        log_label="SGLang embedding server",
    )


def main():
    import tyro

    args = tyro.cli(Args, use_underscores=True)
    args.served_model_name = (args.served_model_name or args.model_name).strip()

    if args.start_port <= 0 or args.start_port >= 65535:
        raise SystemExit(f"invalid --start_port: {args.start_port}")
    if not args.served_model_name:
        raise SystemExit("--model_name/--served_model_name is empty")

    setup_loguru(level=args.log_level, intercept_stdlib=True)

    registry = TmuxRegistry(args.tmux_prefix)
    app = create_app(args, registry)
    run_router_server(
        app, host=args.router_host, port=args.start_port, registry=registry
    )


if __name__ == "__main__":
    main()

"""
# embedding router (tmux backends, ~4GB/GPU)
python -m serving.embedding_router --model-name Qwen/Qwen3-Embedding-0.6B --start-port 17000
python -m serving.embedding_router --model-name Qwen/Qwen3-Embedding-0.6B --start-port 17000 --target-mib 4GB
"""
