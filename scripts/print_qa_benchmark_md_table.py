from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import tyro

DATASETS = (
    "2WikiMultiHopQA",
    "HotpotQA",
    "MuSiQue",
    "FRAMES",
    "PopQA",
    "Bamboogle",
)

HEADER_LABEL = {
    "2WikiMultiHopQA": "2Wiki",
    "HotpotQA": "HotpotQA",
    "MuSiQue": "MuSiQue",
    "FRAMES": "FRAMES",
    "PopQA": "PopQA",
    "Bamboogle": "Bamboogle",
}

DATASET_ALIASES = {
    "2wiki": "2WikiMultiHopQA",
    "2wikimultihopqa": "2WikiMultiHopQA",
    "hotpotqa": "HotpotQA",
    "musique": "MuSiQue",
    "frames": "FRAMES",
    "popqa": "PopQA",
    "bamboogle": "Bamboogle",
}

MODEL_LABELS = {
    "gpt-5.4": "gpt-5.4-2026-03-05",
}

# 表格行显示顺序；未列出的模型按名字排在末尾。
MODEL_ORDER = (
    "Ministral-3-8B-Instruct-2512-BF16",
    "Qwen3.5-4B",
    "Qwen3.5-9B",
    "gemma-4-12B-it",
    "deepseek-v4-pro",
    "gpt-5.4",
)
_MODEL_RANK = {name: i for i, name in enumerate(MODEL_ORDER)}

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None


def _loads(data: bytes) -> dict:
    """Prefer the faster JSON parser when available."""
    if orjson is not None:
        return orjson.loads(data)
    return json.loads(data)


def _parse_datasets(raw: str | None) -> tuple[str, ...]:
    """Parse dataset list; supports full names and aliases."""
    if not raw:
        return DATASETS
    datasets: list[str] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        dataset = DATASET_ALIASES.get(token.lower(), token)
        if dataset not in HEADER_LABEL:
            raise ValueError(f"unsupported dataset: {token}")
        if dataset not in datasets:
            datasets.append(dataset)
    return tuple(datasets)


def _infer_models(
    root: Path, datasets: tuple[str, ...], raw: str | None
) -> tuple[str, ...]:
    """Infer model directories under any dataset; missing cells render as '-'."""
    if raw:
        models = [item.strip() for item in raw.split(",") if item.strip()]
        if not models:
            raise ValueError("no model names provided")
        return tuple(models)

    found: set[str] = set()
    for dataset in datasets:
        dataset_root = root / dataset
        if not dataset_root.is_dir():
            continue
        found.update(path.name for path in dataset_root.iterdir() if path.is_dir())
    if not found:
        raise ValueError(
            f"cannot infer model directories under {root}; pass --models explicitly"
        )
    return tuple(sorted(found, key=lambda m: (_MODEL_RANK.get(m, len(MODEL_ORDER)), m)))


def _scan_model_dir(path_str: str) -> tuple[int, float, float]:
    """Scan one directory; aggregate f1 and gpt with total file count as denominator."""
    total = 0
    f1_sum = 0.0
    gpt_sum = 0.0
    with os.scandir(path_str) as entries:
        for entry in entries:
            if not entry.is_file() or not entry.name.endswith(".json"):
                continue
            total += 1
            payload = _loads(Path(entry.path).read_bytes())
            metrics = payload.get("eval") or {}
            f1 = metrics.get("f1")
            gpt = metrics.get("gpt_judge")
            f1_sum += 0.0 if f1 is None else float(f1)
            gpt_sum += 0.0 if gpt is None else float(gpt)
    return total, f1_sum, gpt_sum


def _center(text: str, width: int) -> str:
    """Center text within a fixed width."""
    if len(text) >= width:
        return text
    pad = width - len(text)
    left = pad // 2
    return " " * left + text + " " * (pad - left)


def _pipe(cells: list[str]) -> str:
    """Render a Markdown table row."""
    return "| " + " | ".join(cells) + " |"


def _format_score(value: float | None, total: int | None) -> str:
    """Format a 0-1 score as a percentage."""
    if total is None or total <= 0 or value is None:
        return "-"
    return f"{value * 100:.2f}"


def _model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def _collect_results(
    root: Path,
    datasets: tuple[str, ...],
    models: tuple[str, ...],
    jobs: int | None,
) -> dict[tuple[str, str], tuple[int | None, float | None, float | None]]:
    """Scan all model×dataset dirs under root in parallel and aggregate eval metrics."""
    tasks: list[tuple[str, str, Path]] = []
    for model in models:
        for dataset in datasets:
            tasks.append((model, dataset, root / dataset / model))

    cpu = os.cpu_count() or 4
    workers = jobs if jobs is not None else min(cpu, max(1, len(tasks)))
    results: dict[tuple[str, str], tuple[int | None, float | None, float | None]] = {}
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(_scan_model_dir, str(path)): (model, dataset, path)
            for model, dataset, path in tasks
            if path.is_dir()
        }
        for model, dataset, path in tasks:
            if not path.is_dir():
                results[(model, dataset)] = (None, None, None)
        for future in as_completed(futures):
            model, dataset, _path = futures[future]
            total, f1_sum, gpt_sum = future.result()
            if total <= 0:
                results[(model, dataset)] = (0, 0.0, 0.0)
            else:
                results[(model, dataset)] = (total, f1_sum / total, gpt_sum / total)
    return results


def _print_table(
    title: str,
    datasets: tuple[str, ...],
    models: tuple[str, ...],
    results: dict[tuple[str, str], tuple[int | None, float | None, float | None]],
    disable_n: bool,
) -> None:
    """Print a Markdown table with a two-level header."""
    print(title)

    # Compute column widths so two-level headers align with data rows.
    method_w = max(
        len("Method"), max((len(_model_label(model)) for model in models), default=0)
    )
    n_w = len("n")
    for model in models:
        for dataset in datasets:
            total, _f1, _gpt = results[(model, dataset)]
            if total is not None:
                n_w = max(n_w, len(str(total)))
    f_w = max(len("f1"), len("100.00"))
    g_w = max(len("gpt"), len("100.00"))
    widths: list[tuple[int, int, int]] = []
    for dataset in datasets:
        n_width = n_w if not disable_n else 0
        f_width = f_w
        g_width = g_w
        block_width = (
            (f_width + 3 + g_width)
            if disable_n
            else (n_width + 3 + f_width + 3 + g_width)
        )
        label_width = len(HEADER_LABEL[dataset])
        if label_width > block_width:
            g_width += label_width - block_width
        widths.append((n_width, f_width, g_width))

    top_cells = ["Method".ljust(method_w)]
    for dataset, (n_width, f_width, g_width) in zip(datasets, widths, strict=True):
        block_width = (
            (f_width + 3 + g_width)
            if disable_n
            else (n_width + 3 + f_width + 3 + g_width)
        )
        top_cells.append(_center(HEADER_LABEL[dataset], block_width))
    print(_pipe(top_cells))

    sub_cells = ["".ljust(method_w)]
    for n_width, f_width, g_width in widths:
        if not disable_n:
            sub_cells.append("n".rjust(n_width))
        sub_cells.append("f1".rjust(f_width))
        sub_cells.append("gpt".rjust(g_width))
    print(_pipe(sub_cells))

    sep_cells = ["-" * method_w]
    for n_width, f_width, g_width in widths:
        if not disable_n:
            sep_cells.append("-" * n_width)
        sep_cells.append("-" * f_width)
        sep_cells.append("-" * g_width)
    print(_pipe(sep_cells))

    for model in models:
        row = [_model_label(model).ljust(method_w)]
        for dataset, (n_width, f_width, g_width) in zip(datasets, widths, strict=True):
            total, f1, gpt = results[(model, dataset)]
            if not disable_n:
                row.append(
                    "-".rjust(n_width) if total is None else str(total).rjust(n_width)
                )
            row.append(_format_score(f1, total).rjust(f_width))
            row.append(_format_score(gpt, total).rjust(g_width))
        print(_pipe(row))


@dataclass
class Args:
    full_root: str = "output_full_test"
    sample_root: str = "output_sample_random_300"
    root: str | None = None  # Deprecated alias for full_root.
    datasets: str | None = None
    models: str | None = None
    jobs: int | None = None
    disable_n: bool = False


def main(args: Args) -> None:
    """Scan full_test and sample_random_300 and print two Markdown tables."""
    full_root = Path(args.root) if args.root is not None else Path(args.full_root)
    sample_root = Path(args.sample_root)
    datasets = _parse_datasets(args.datasets)

    # Each table infers its own model set; explicit --models applies to both.
    full_models = _infer_models(full_root, datasets, args.models)
    sample_models = _infer_models(sample_root, datasets, args.models)

    full_results = _collect_results(full_root, datasets, full_models, args.jobs)
    sample_results = _collect_results(sample_root, datasets, sample_models, args.jobs)

    _print_table("full_test", datasets, full_models, full_results, args.disable_n)
    print()
    _print_table(
        "sample_random_300",
        datasets,
        sample_models,
        sample_results,
        args.disable_n,
    )


if __name__ == "__main__":
    main(tyro.cli(Args, use_underscores=True))
