"""Drop-in for ``python -m sglang.launch_server`` with project compat patches.

Backend tmux processes import this module so patches apply before SGLang init.
"""

from __future__ import annotations

import os
import sys
import warnings

from loguru import logger

from serving.sglang_compat import apply_sglang_compat_patches

# Apply early so imports that touch Tool / Ministral3 see patched symbols.
_EARLY_PATCHES = apply_sglang_compat_patches()

from sglang.launch_server import run_server  # noqa: E402
from sglang.srt.server_args import prepare_server_args  # noqa: E402
from sglang.srt.utils import kill_process_tree  # noqa: E402
from sglang.srt.utils.common import suppress_noisy_warnings  # noqa: E402

suppress_noisy_warnings()


if __name__ == "__main__":
    warnings.warn(
        "'python -m serving.sglang_launch_server' wraps sglang.launch_server "
        "with project compatibility patches.",
        UserWarning,
        stacklevel=1,
    )

    # Re-apply after sglang imports in case classes were rebound during load.
    patched = apply_sglang_compat_patches()
    # De-dupe while preserving order for the startup log line.
    seen: set[str] = set()
    all_patches: list[str] = []
    for name in [*_EARLY_PATCHES, *patched]:
        if name not in seen:
            seen.add(name)
            all_patches.append(name)
    if all_patches:
        logger.warning(
            "Applied in-process sglang compat patches: " + ", ".join(all_patches)
        )
    else:
        logger.warning("No sglang compat patches applied (unexpected)")

    from sglang.srt.plugins import load_plugins

    load_plugins()

    server_args = prepare_server_args(sys.argv[1:])

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
