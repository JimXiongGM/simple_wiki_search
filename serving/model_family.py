"""Model family detection and lightweight compatibility settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_model_config(model_name: str) -> dict[str, Any]:
    """Read config.json from the local model directory when possible; return {} on failure."""
    model_path = Path(model_name).expanduser()
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    config_path = model_path / "config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _config_fields(model_name: str) -> tuple[str, str, str]:
    """Extract model_type, architectures, and text_config.model_type for family detection."""
    config = _load_model_config(model_name)
    model_type = str(config.get("model_type") or "").lower()
    architectures = " ".join(str(x).lower() for x in config.get("architectures", []))
    text_model_type = str(
        (config.get("text_config") or {}).get("model_type") or ""
    ).lower()
    return model_type, architectures, text_model_type


def is_qwen3_family_model(model_name: str) -> bool:
    """Return True if the model belongs to the Qwen3 / Qwen3.5 / Qwen3-Next family."""
    lowered = model_name.lower()
    if "qwen3" in lowered:
        return True
    model_type, architectures, text_model_type = _config_fields(model_name)
    return any(
        "qwen3" in field for field in (model_type, architectures, text_model_type)
    )


def is_gemma4_family_model(model_name: str) -> bool:
    """Return True if the model belongs to the Gemma 4 family (including gemma4_unified / 12B encoder-free)."""
    lowered = model_name.lower()
    if "gemma-4" in lowered or "gemma4" in lowered:
        return True
    model_type, architectures, text_model_type = _config_fields(model_name)
    return any(
        "gemma4" in field for field in (model_type, architectures, text_model_type)
    )


def is_mistral_family_model(model_name: str) -> bool:
    """Return True if the model belongs to the Mistral / Ministral family."""
    lowered = model_name.lower()
    if "mistral" in lowered or "ministral" in lowered:
        return True
    model_type, architectures, text_model_type = _config_fields(model_name)
    return any(
        key in field
        for field in (model_type, architectures, text_model_type)
        for key in ("mistral", "ministral")
    )


def is_mistral_reasoning_model(model_name: str) -> bool:
    """Only Ministral/Mistral Reasoning variants need the mistral reasoning parser.

    The official Instruct cookbook only requires `--tool-call-parser mistral`;
    add `--reasoning-parser mistral` for Reasoning / `[THINK]` protocol models.
    """
    if not is_mistral_family_model(model_name):
        return False
    lowered = Path(model_name).name.lower()
    return "reasoning" in lowered or "magistral" in lowered


def default_chat_template_kwargs_for_model(model_name: str) -> dict[str, Any]:
    """Return chat_template_kwargs recommended for inference."""
    if is_gemma4_family_model(model_name):
        # Gemma 4 best practice: disable hidden thinking for multi-turn tool calling
        # so historical thoughts are not re-injected by the template.
        return {"enable_thinking": False}
    return {}


def sglang_server_extra_args(
    model_name: str,
    *,
    mem_fraction_static: float,
    context_length: int,
    max_running_requests: int,
) -> tuple[list[str], list[str]]:
    """Return extra SGLang launch args and human-readable parser notes by model family.

    Returns (extra_argv, parser_notes). Aligned with official cookbooks / run_qa_agent tool protocol:
    - Qwen3*: reasoning=qwen3, tool=qwen3_coder
    - Gemma4*: reasoning=gemma4, tool=gemma4 (agent defaults to enable_thinking=False)
    - Mistral/Ministral Instruct: tool=mistral + trust_remote_code
    - Mistral/Ministral Reasoning: additionally reasoning=mistral
    """
    common = [
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--context-length",
        str(context_length),
        "--allow-auto-truncate",
        "--max-running-requests",
        str(max_running_requests),
    ]
    notes: list[str] = []

    if is_qwen3_family_model(model_name):
        args = common + [
            "--reasoning-parser",
            "qwen3",
            "--tool-call-parser",
            "qwen3_coder",
        ]
        notes.append("reasoning=qwen3 tool=qwen3_coder")
        return args, notes

    if is_gemma4_family_model(model_name):
        args = common + [
            "--reasoning-parser",
            "gemma4",
            "--tool-call-parser",
            "gemma4",
        ]
        notes.append("reasoning=gemma4 tool=gemma4")
        return args, notes

    if is_mistral_family_model(model_name):
        # Ministral-3 Instruct cookbook: --tool-call-parser mistral --trust-remote-code
        args = common + [
            "--tool-call-parser",
            "mistral",
            "--trust-remote-code",
        ]
        notes.append("tool=mistral trust_remote_code")
        if is_mistral_reasoning_model(model_name):
            args.extend(["--reasoning-parser", "mistral"])
            notes.append("reasoning=mistral")
        return args, notes

    return common, notes
