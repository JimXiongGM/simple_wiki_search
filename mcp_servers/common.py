from __future__ import annotations

import asyncio
import json
import traceback
from pathlib import Path
from typing import Any

from loguru import logger


def proc_mem_mb() -> dict[str, int]:
    """
    Linux-only lightweight memory probe via /proc/self/status.
    Returns values in MB (integer, rounded down). Missing keys are omitted.
    """
    try:
        out: dict[str, int] = {}
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if not line:
                    continue
                if line.startswith(
                    ("VmRSS:", "VmHWM:", "VmSize:", "RssAnon:", "RssFile:", "Threads:")
                ):
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    k = parts[0].rstrip(":")
                    if k == "Threads":
                        try:
                            out["threads"] = int(parts[1])
                        except Exception:
                            pass
                        continue
                    try:
                        kb = int(parts[1])
                        out[k.lower() + "_mb"] = kb // 1024
                    except Exception:
                        continue
        return out
    except Exception:
        return {}


async def mem_heartbeat_task(interval_s: float) -> None:
    while True:
        mem = proc_mem_mb()
        if mem:
            logger.info(
                "mem_heartbeat rss_mb={} hwm_mb={} vms_mb={} threads={}",
                mem.get("vmrss_mb"),
                mem.get("vmhwm_mb"),
                mem.get("vmsize_mb"),
                mem.get("threads"),
            )
        await asyncio.sleep(float(interval_s))


def exc_location(e: BaseException) -> str:
    try:
        tb = traceback.extract_tb(e.__traceback__)
        if not tb:
            return ""
        last = tb[-1]
        p = str(last.filename or "")
        return f"{p}:{int(last.lineno)} in {last.name}"
    except Exception:
        return ""


def is_timeout_exc(e: BaseException) -> bool:
    try:
        if isinstance(e, (TimeoutError, asyncio.TimeoutError)):
            return True
    except Exception:
        pass
    name = type(e).__name__.lower()
    msg = str(e).lower()
    if "timeout" in name or "timed out" in msg or "timeout" in msg:
        return True
    return False


def format_tool_error_md(
    *,
    tool: str,
    code: str,
    message: str,
    meta: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append(f"# {tool} Error")
    lines.append("")
    lines.append(f"- code: `{code}`")
    lines.append(f"- message: `{message}`")
    for k in sorted(meta.keys()):
        v = meta.get(k)
        try:
            if isinstance(v, (dict, list)):
                vv = json.dumps(v, ensure_ascii=False)
            else:
                vv = str(v)
        except Exception:
            vv = str(v)
        if vv == "":
            continue
        vv = vv.replace("\n", "\\n")
        lines.append(f"- {k}: `{vv}`")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"
