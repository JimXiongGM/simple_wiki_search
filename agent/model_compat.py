"""Model-specific request-message patches for chat-template quirks."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from serving.model_family import is_gemma4_family_model, is_mistral_family_model

# mistral_common serving validator: tool call ids must match ^[a-zA-Z0-9]{9}$
_MISTRAL_TOOL_CALL_ID_RE = re.compile(r"^[a-zA-Z0-9]{9}$")
# mistral_common: function names must be a-zA-Z0-9_- with length 1..64
_MISTRAL_FUNCTION_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def mistral_tool_call_violation(tool_call: dict[str, Any]) -> str | None:
    """Return the mistral_common-style rejection text, or None if the call is valid.

    Only checks function **names** here. Tool-call ids are remapped for the wire
    in ``rewrite_mistral_tool_call_ids`` (often server-assigned, not model text).
    Wording mirrors mistral_common InvalidFunctionCallException.
    """
    fn = (
        tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
    )
    name = str(fn.get("name") if fn.get("name") is not None else "")
    if not _MISTRAL_FUNCTION_NAME_RE.match(name):
        return (
            f"Function name was {name} but must be a-z, A-Z, 0-9, or contain "
            "underscores and dashes, with a maximum length of 64."
        )
    return None


def _to_mistral_tool_call_id(raw_id: str, *, used: set[str]) -> str:
    """Map any tool-call id to a mistral_common-compliant 9-char id."""
    raw = str(raw_id or "").strip()
    if _MISTRAL_TOOL_CALL_ID_RE.match(raw) and raw not in used:
        used.add(raw)
        return raw
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    candidate = digest[:9]
    if candidate in used:
        # Extremely unlikely collision; walk the hex digest for an unused 9-char window.
        for start in range(1, max(1, len(digest) - 8)):
            candidate = digest[start : start + 9]
            if candidate not in used:
                break
        else:
            n = 0
            while True:
                candidate = hashlib.sha1(f"{raw}:{n}".encode("utf-8")).hexdigest()[:9]
                if candidate not in used:
                    break
                n += 1
    used.add(candidate)
    return candidate


def rewrite_mistral_tool_call_ids(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rewrite assistant/tool ids so multi-turn mistral chat templates validate."""
    id_map: dict[str, str] = {}
    used: set[str] = set()
    rewritten: list[dict[str, Any]] = []
    for message in messages:
        msg = dict(message)
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            tool_calls = []
            for tool_call in msg["tool_calls"]:
                tc = dict(tool_call)
                raw = str(tc.get("id") or "")
                mapped = id_map.get(raw)
                if mapped is None:
                    mapped = _to_mistral_tool_call_id(raw, used=used)
                    id_map[raw] = mapped
                tc["id"] = mapped
                if isinstance(tc.get("function"), dict):
                    tc["function"] = dict(tc["function"])
                tool_calls.append(tc)
            msg["tool_calls"] = tool_calls
            if msg.get("content") is None:
                msg["content"] = ""
        elif role == "tool":
            raw = str(msg.get("tool_call_id") or "")
            if raw:
                mapped = id_map.get(raw)
                if mapped is None:
                    mapped = _to_mistral_tool_call_id(raw, used=used)
                    id_map[raw] = mapped
                msg["tool_call_id"] = mapped
        rewritten.append(msg)
    return rewritten


def strip_gemma4_historical_reasoning(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop historical assistant thoughts so Gemma 4 templates do not reinject them."""
    prepared: list[dict[str, Any]] = []
    for message in messages:
        msg = dict(message)
        if msg.get("role") == "assistant":
            msg.pop("reasoning", None)
            msg.pop("reasoning_content", None)
        prepared.append(msg)
    return prepared


def prepare_request_messages(
    model_name: str,
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply per-model chat-template sanitization before a completion request."""
    prepared = [dict(message) for message in messages]
    if is_mistral_family_model(model_name):
        # mistral_common rejects OpenAI-style ids like call_<uuid>; remap to
        # 9-char [a-zA-Z0-9] so multi-turn tool history can be tokenized.
        prepared = rewrite_mistral_tool_call_ids(prepared)
    if is_gemma4_family_model(model_name):
        prepared = strip_gemma4_historical_reasoning(prepared)
    return prepared
