from __future__ import annotations

import random
import socket
import sys
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from serving.model_family import is_gemma4_family_model, sglang_server_extra_args
from serving.router_app import UpstreamRetryConfig, create_router_app, run_router_server
from serving.sglang_compat import apply_sglang_compat_patches
from serving.tmux import (
    TmuxRegistry,
    backend_base_url,
    detect_num_gpus,
    shell_cmd_with_gpu_and_tee,
    tmux_safe_name,
)
from utils.logging import setup_loguru

# SGLang defaults gRPC to HTTP `--port` + 10000; keep HTTP in range.
_MAX_SGLANG_HTTP_PORT_FOR_DEFAULT_GRPC = 65535 - 10_000


def get_random_unused_port(
    low: int = 20000, high: int = _MAX_SGLANG_HTTP_PORT_FOR_DEFAULT_GRPC
) -> int:
    for _ in range(200):
        port = random.randint(low, high)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("Could not find unused port in range")


def preflight_sglang_runtime(model_name: str) -> None:
    """Fail fast with actionable errors before spawning tmux backends."""
    try:
        import sglang  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            f"sglang is not importable: {type(exc).__name__}: {exc}\n"
            "Install/upgrade with: pip install -U 'sglang>=0.5.13'"
        ) from exc

    patched = apply_sglang_compat_patches()
    if patched:
        logger.warning(
            "Applied in-process sglang compat patches: " + ", ".join(patched)
        )

    try:
        from sglang.srt.server_args import ServerArgs  # noqa: F401
    except ValueError as exc:
        msg = str(exc)
        if "already used by a Transformers config" in msg:
            raise SystemExit(
                f"sglang/transformers config registration conflict: {exc}\n"
                "Tried AutoConfig.register exist_ok monkeypatch; please upgrade "
                "sglang (>=0.5.13 for gemma-4-12B-it) or align transformers."
            ) from exc
        raise SystemExit(f"sglang import failed: {exc}") from exc
    except Exception as exc:
        raise SystemExit(
            f"sglang runtime preflight failed: {type(exc).__name__}: {exc}"
        ) from exc

    if is_gemma4_family_model(model_name):
        import sglang as _sglang

        unified = (
            Path(_sglang.__file__).resolve().parent
            / "srt"
            / "models"
            / "gemma4_unified.py"
        )
        if not unified.exists():
            ver = getattr(_sglang, "__version__", "unknown")
            raise SystemExit(
                "google/gemma-4-12B-it needs sglang with gemma4_unified "
                f"(sglang>=0.5.13), but current sglang={ver} lacks "
                f"{unified.name}. Upgrade: pip install -U 'sglang>=0.5.13'"
            )


@dataclass(frozen=True)
class BackendInstance:
    gpu_id: int
    port: int
    model_name: str
    api_key: str
    backend_host: str
    tmux_session: str
    log_file: str
    dtype: str
    enable_speculative: bool
    mem_fraction_static: float
    context_length: int
    max_running_requests: int

    @property
    def base_url(self) -> str:
        return backend_base_url(self.backend_host, self.port)

    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def build_shell_cmd(self) -> str:
        # Use project wrapper so Pixtral/MistralCommon pads skip applies in tmux.
        cmd = [
            sys.executable,
            "-m",
            "serving.sglang_launch_server",
            "--model-path",
            self.model_name,
            "--port",
            str(self.port),
            "--host",
            self.backend_host,
            "--api-key",
            self.api_key,
            "--dtype",
            self.dtype,
        ]
        family_args, parser_notes = sglang_server_extra_args(
            self.model_name,
            mem_fraction_static=self.mem_fraction_static,
            context_length=self.context_length,
            max_running_requests=self.max_running_requests,
        )
        cmd.extend(family_args)
        if parser_notes:
            logger.info(
                f"Add model-family launch args for {self.model_name}: "
                + ", ".join(parser_notes)
                + f" (mem_fraction={self.mem_fraction_static}, "
                f"max_running_requests={self.max_running_requests})"
            )
        if self.enable_speculative:
            if is_gemma4_family_model(self.model_name):
                raise RuntimeError(
                    "Gemma 4 speculative decoding needs a paired "
                    "*-assistant draft model; launch without --enable-speculative "
                    "or wire draft-model args explicitly."
                )
            cmd.extend(
                [
                    "--speculative-algo",
                    "NEXTN",
                    "--speculative-num-steps",
                    "3",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "4",
                ]
            )
            logger.info(
                f"Add speculative algo, num steps, eagle topk, num draft tokens "
                f"for {self.model_name}"
            )
        return shell_cmd_with_gpu_and_tee(self.gpu_id, cmd, self.log_file)


@dataclass
class Args:
    model_name: str
    start_port: int = 19000
    served_model_name: str | None = None
    key: str = "simple_wiki"

    router_host: str = "0.0.0.0"
    backend_host: str = "127.0.0.1"

    tmux_prefix: str = "sglang-"
    kill_existing: bool = True
    wait_timeout_s: int = 600

    dtype: str = "bfloat16"
    enable_speculative: bool = False

    mem_fraction_static: float = 0.85
    context_length: int = 65536
    max_running_requests: int = 8

    upstream_connect_timeout_s: float = 10.0
    upstream_timeout_s: float = 600.0

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

        backend_log_dir = Path("logs") / "sglang_backends"
        backend_log_dir.mkdir(parents=True, exist_ok=True)

        mname = args.model_name.split("/")[-1]
        used_ports: set[int] = {args.start_port}
        backends: list[BackendInstance] = []
        for gpu_id in range(num_gpus):
            while True:
                port = get_random_unused_port()
                if port not in used_ports:
                    used_ports.add(port)
                    break
            tmux_session = f"{args.tmux_prefix}{gpu_id}-{tmux_safe_name(mname)}"
            inst = BackendInstance(
                gpu_id=gpu_id,
                port=port,
                model_name=args.model_name,
                api_key=args.key,
                backend_host=args.backend_host,
                tmux_session=tmux_session,
                log_file=str(backend_log_dir / f"{mname}-gpu{gpu_id}.log"),
                dtype=args.dtype,
                enable_speculative=args.enable_speculative,
                mem_fraction_static=args.mem_fraction_static,
                context_length=args.context_length,
                max_running_requests=args.max_running_requests,
            )
            registry.mark_pending(inst.tmux_session)
            backends.append(inst)
            logger.info(f"GPU {gpu_id}: port={port}")
        return backends

    return create_router_app(
        title="llm_router",
        registry=registry,
        build_backends=build_backends,
        served_model_name=served_model_name,
        api_key=args.key,
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
        require_api_key=True,
        validate_model=True,
        log_label="SGLang server",
    )


def main():
    import tyro

    args = tyro.cli(Args, use_underscores=True)
    cli_model_name = args.model_name.strip()
    model_path = Path(cli_model_name).expanduser()
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    if not model_path.exists():
        raise SystemExit(f"local model path not found: {cli_model_name}")
    try:
        args.model_name = (
            model_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        )
    except ValueError:
        args.model_name = model_path.resolve().as_posix()
    # API model id: keep CLI name; backend uses resolved local path.
    args.served_model_name = (args.served_model_name or cli_model_name).strip()
    args.key = args.key.strip()

    if args.start_port <= 0 or args.start_port >= 65535:
        raise SystemExit(f"invalid --start_port: {args.start_port}")
    if not args.served_model_name:
        raise SystemExit("--model_name/--served_model_name is empty")
    if not args.key:
        raise SystemExit("--key is empty")

    setup_loguru(level=args.log_level, intercept_stdlib=True)
    preflight_sglang_runtime(args.model_name)

    registry = TmuxRegistry(args.tmux_prefix)
    app = create_app(args, registry)
    run_router_server(
        app, host=args.router_host, port=args.start_port, registry=registry
    )


if __name__ == "__main__":
    main()

"""
# Requires sglang>=0.5.13 for google/gemma-4-12B-it (gemma4_unified).
python -m serving.llm_router --model-name Qwen/Qwen3.5-9B --start-port 19000 --enable-speculative
python -m serving.llm_router --model-name Qwen/Qwen3.6-35B-A3B --start-port 19000
python -m serving.llm_router --model-name mistralai/Ministral-3-8B-Instruct-2512-BF16 --start-port 19000
python -m serving.llm_router --model-name google/gemma-4-12B-it --start-port 19000
python -m serving.llm_router --model-name output/checkpoint/my_hard_sft --start-port 19000

# Equivalent raw sglang cmds (tool parsers must match run_qa_agent OpenAI tools=...):
# Ministral Instruct: --tool-call-parser mistral --trust-remote-code
# Gemma4: --reasoning-parser gemma4 --tool-call-parser gemma4
#   (agent sends chat_template_kwargs.enable_thinking=False)
# Qwen3.6: --reasoning-parser qwen3 --tool-call-parser qwen3_coder

# CURL test:
curl http://localhost:19000/v1/chat/completions -X POST -H "Content-Type: application/json" -d '{"model": "Qwen/Qwen3.5-4B", "key": "simple_wiki", "messages": [{"role": "user", "content": "What is the capital of France?"}]}'
"""
