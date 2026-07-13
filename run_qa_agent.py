from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro
from loguru import logger
from tqdm import tqdm

from agent.wikipedia_offline import WikipediaOfflineAgent
from evaluation.qa_item import (
    QaItem,
    evaluate_item_metrics,
    item_answer_aliases,
    item_golden_answer,
    item_has_gold_answer,
)
from mcp_servers.wikipedia_offline_client import configure_default_client
from settings import DEFAULT_LLM_SERVER_URL, DEFAULT_TOOL_BASE_URL

FULL_TEST_GOLD_SPLITS: dict[str, str] = {
    "2WikiMultiHopQA": "data/dataset/full_test/2WikiMultiHopQA/dev.jsonl",
    "HotpotQA": "data/dataset/full_test/HotpotQA/hotpot_dev_fullwiki_v1.jsonl",
    "MuSiQue": "data/dataset/full_test/MuSiQue/musique_ans_v1.0_dev.jsonl",
    "FRAMES": "data/dataset/full_test/FRAMES/frames.jsonl",
    "PopQA": "data/dataset/full_test/PopQA/popqa.jsonl",
    "Bamboogle": "data/dataset/full_test/Bamboogle/bamboogle.jsonl",
}


@dataclass
class Args:
    model_name: str = "Qwen/Qwen3.5-9B"
    server_url: str = DEFAULT_LLM_SERVER_URL
    provider: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    tool_server_url: str = DEFAULT_TOOL_BASE_URL
    data_dir: str | None = None
    dataset_name: str = "MuSiQue"
    note: str = "v1"
    output_root: str = "output-v2/interaction"
    save_dir: str | None = None
    max_workers: int = 6
    max_round: int = 20
    max_tool_calls_per_step: int = 5
    max_completion_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    seed: int = 42
    limit: int | None = None
    debug: bool = False
    use_full_test_data: bool = False
    enable_llm_judge: bool = False


def _setup_logging(debug: bool) -> None:
    logger.remove()
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        level="DEBUG" if debug else "INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}",
        colorize=True,
    )


def _normalize_processed_item(item: QaItem, dataset_name: str) -> QaItem:
    row: QaItem = dict(item)
    if not str(row.get("question") or "").strip():
        prompt = str(row.get("prompt") or "").strip()
        if prompt:
            row["question"] = prompt
    if not str(row.get("type") or "").strip():
        ds = dataset_name.strip().lower()
        if ds == "musique":
            qid = str(row.get("id") or "")
            row["type"] = qid.split("__", 1)[0] if "__" in qid else "unknown"
        elif ds == "popqa":
            row["type"] = str(row.get("prop") or "unknown")
        else:
            row["type"] = "default"
    return row


def _load_jsonl(path: Path, dataset_name: str) -> list[QaItem]:
    items: list[QaItem] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = _normalize_processed_item(json.loads(line), dataset_name)
            if item_has_gold_answer(row):
                items.append(row)
    return items


def _load_json(path: Path) -> list[QaItem]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"expected list payload in {path}")
    return [dict(item) for item in payload]


def _load_items(data_path: Path, dataset_name: str) -> list[QaItem]:
    if data_path.suffix.lower() == ".jsonl":
        return _load_jsonl(data_path, dataset_name)
    return _load_json(data_path)


def _default_save_dir(
    *,
    output_root: str,
    dataset_name: str,
    model_name: str,
    note: str,
    subset_name: str,
) -> str:
    model_dir = model_name.split("/")[-1]
    if note and note not in model_dir:
        model_dir = f"{model_dir}-{note}"
    return f"{str(output_root).rstrip('/')}/{dataset_name}/{model_dir}/{subset_name}"


def _resolve_data_path(
    *,
    data_dir: str | None,
    dataset_name: str,
    use_full_test_data: bool,
) -> tuple[Path, str]:
    if data_dir:
        path = Path(data_dir)
        subset_name = path.parent.name if path.parent.name else path.stem
        return path, subset_name
    if use_full_test_data:
        return Path(FULL_TEST_GOLD_SPLITS[dataset_name]), "full_test"
    return (
        Path(f"data/dataset/random_300/{dataset_name}.json"),
        "random_300",
    )


def _result_path(save_dir: Path, item: QaItem) -> Path:
    qid = str(item.get("id") or item.get("qid") or "").strip()
    if not qid:
        raise ValueError(f"missing id/qid: {item}")
    return save_dir / f"{qid}.json"


def _build_record(
    *,
    item: QaItem,
    result: dict[str, Any],
    eval_metrics: dict[str, float] | None,
) -> dict[str, Any]:
    gold = (
        item_golden_answer(item) if item_has_gold_answer(item) else item.get("answer")
    )
    return {
        "id": str(item.get("id") or item.get("qid") or "").strip(),
        "type": str(item.get("type") or "").strip(),
        "question": item["question"],
        "answer": gold,
        "answer_aliases": item_answer_aliases(item),
        "model": result["model_name"],
        "messages": result["messages"],
        "tools": result["tools"],
        "predicted_answer": result["predicted_answer"],
        "stop_reason": result["stop_reason"],
        "time_cost": result["time_cost"],
        "usage": result["usage"],
        "eval": eval_metrics,
    }


def _metric_or_none(metrics: dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(metrics, dict) or key not in metrics:
        return None
    value = metrics.get(key)
    if value is None:
        return None
    return float(value)


def _result_status_payload(
    *,
    qid: str,
    status: str,
    eval_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "qid": qid,
        "status": status,
        "f1": _metric_or_none(eval_metrics, "f1"),
        "gpt_judge": _metric_or_none(eval_metrics, "gpt_judge"),
    }


def _format_running_avg_postfix(
    *,
    f1_sum: float,
    f1_count: int,
    gpt_sum: float,
    gpt_count: int,
) -> str:
    parts: list[str] = []
    if f1_count > 0:
        parts.append(f"avg F1={100.0 * f1_sum / f1_count:.2f}")
    if gpt_count > 0:
        parts.append(f"avg llm_judge={gpt_sum / gpt_count:.2f}")
    return " | ".join(parts)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp_path.replace(path)


def _load_cached_status(save_dir: Path, item: QaItem) -> dict[str, Any] | None:
    """Return status payload if a valid cache exists; remove corrupt files."""
    out_path = _result_path(save_dir, item)
    if not out_path.exists():
        return None
    try:
        cached = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cached = None
    if isinstance(cached, dict) and bool(str(cached.get("id") or "").strip()):
        eval_metrics = cached.get("eval")
        return _result_status_payload(
            qid=str(cached.get("id") or out_path.stem),
            status="cached",
            eval_metrics=eval_metrics if isinstance(eval_metrics, dict) else None,
        )
    logger.warning("Removing corrupt cache file: {}", out_path)
    out_path.unlink(missing_ok=True)
    return None


def _accumulate_metrics(
    payload: dict[str, Any],
    *,
    f1_sum: float,
    f1_count: int,
    gpt_sum: float,
    gpt_count: int,
) -> tuple[float, int, float, int]:
    f1 = payload.get("f1")
    if f1 is not None:
        f1_sum += float(f1)
        f1_count += 1
    gpt = payload.get("gpt_judge")
    if gpt is not None:
        gpt_sum += float(gpt)
        gpt_count += 1
    return f1_sum, f1_count, gpt_sum, gpt_count


def _partition_cached_items(
    items: list[QaItem], save_dir: Path
) -> tuple[list[dict[str, Any]], list[tuple[int, QaItem]]]:
    cached_payloads: list[dict[str, Any]] = []
    pending: list[tuple[int, QaItem]] = []
    for idx, item in enumerate(items):
        cached = _load_cached_status(save_dir, item)
        if cached is not None:
            cached_payloads.append(cached)
        else:
            pending.append((idx, item))
    return cached_payloads, pending


def _run_one(
    item: QaItem,
    *,
    model_name: str,
    server_url: str,
    provider: str | None,
    base_url: str | None,
    api_key: str | None,
    save_dir: Path,
    max_round: int,
    max_tool_calls_per_step: int,
    max_completion_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    debug: bool = False,
    enable_llm_judge: bool = False,
) -> dict[str, Any]:
    out_path = _result_path(save_dir, item)
    agent = WikipediaOfflineAgent(
        qid=str(item.get("id") or item.get("qid") or "").strip(),
        question=str(item["question"]).strip(),
        model_name=model_name,
        server_url=server_url,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        answer=(
            item_golden_answer(item)
            if item_has_gold_answer(item)
            else item.get("answer")
        ),
        max_round=max_round,
        max_tool_calls_per_step=max_tool_calls_per_step,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    result = agent.run(debug=debug)
    eval_metrics = evaluate_item_metrics(
        item,
        result["predicted_answer"],
        enable_llm_judge=enable_llm_judge,
    )
    record = _build_record(item=item, result=result, eval_metrics=eval_metrics)
    _atomic_write_json(out_path, record)
    return _result_status_payload(
        qid=record["id"],
        status="ok",
        eval_metrics=eval_metrics,
    )


def main(args: Args) -> None:
    _setup_logging(args.debug)
    configure_default_client(args.tool_server_url)

    data_path, subset_name = _resolve_data_path(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        use_full_test_data=args.use_full_test_data,
    )
    items = _load_items(data_path, args.dataset_name)
    if args.limit is not None:
        items = items[: int(args.limit)]
    if args.debug:
        items = items[:1]

    resolved_save_dir = Path(
        args.save_dir
        or _default_save_dir(
            output_root=args.output_root,
            dataset_name=args.dataset_name,
            model_name=args.model_name,
            note=args.note,
            subset_name=subset_name,
        )
    )
    resolved_save_dir.mkdir(parents=True, exist_ok=True)

    cached_payloads, pending = _partition_cached_items(items, resolved_save_dir)
    logger.info("dataset={} items={} data={}", args.dataset_name, len(items), data_path)
    logger.info(
        "cache_hit={} pending={}",
        len(cached_payloads),
        len(pending),
    )
    logger.info("tool_server={}", args.tool_server_url)
    logger.info(
        "actor_endpoint=provider={} base_url={}",
        args.provider or "local",
        args.base_url or args.server_url,
    )
    logger.info("enable_llm_judge={}", args.enable_llm_judge)
    logger.info("save_dir={}", resolved_save_dir)

    f1_sum = 0.0
    f1_count = 0
    gpt_sum = 0.0
    gpt_count = 0
    for payload in cached_payloads:
        f1_sum, f1_count, gpt_sum, gpt_count = _accumulate_metrics(
            payload,
            f1_sum=f1_sum,
            f1_count=f1_count,
            gpt_sum=gpt_sum,
            gpt_count=gpt_count,
        )

    if not pending:
        postfix = _format_running_avg_postfix(
            f1_sum=f1_sum,
            f1_count=f1_count,
            gpt_sum=gpt_sum,
            gpt_count=gpt_count,
        )
        logger.info("all items cached{}", f"; {postfix}" if postfix else "")
        return

    if args.debug or args.max_workers <= 1:
        for idx, item in pending:
            payload = _run_one(
                item,
                model_name=args.model_name,
                server_url=args.server_url,
                provider=args.provider,
                base_url=args.base_url,
                api_key=args.api_key,
                save_dir=resolved_save_dir,
                max_round=args.max_round,
                max_tool_calls_per_step=args.max_tool_calls_per_step,
                max_completion_tokens=args.max_completion_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed + idx,
                debug=args.debug,
                enable_llm_judge=args.enable_llm_judge,
            )
            f1_sum, f1_count, gpt_sum, gpt_count = _accumulate_metrics(
                payload,
                f1_sum=f1_sum,
                f1_count=f1_count,
                gpt_sum=gpt_sum,
                gpt_count=gpt_count,
            )
        return

    futures = []
    with ThreadPoolExecutor(max_workers=int(args.max_workers)) as executor:
        for idx, item in pending:
            futures.append(
                executor.submit(
                    _run_one,
                    item,
                    model_name=args.model_name,
                    server_url=args.server_url,
                    provider=args.provider,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    save_dir=resolved_save_dir,
                    max_round=args.max_round,
                    max_tool_calls_per_step=args.max_tool_calls_per_step,
                    max_completion_tokens=args.max_completion_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.seed + idx,
                    debug=args.debug,
                    enable_llm_judge=args.enable_llm_judge,
                )
            )
        failures = 0
        with tqdm(
            total=len(items),
            initial=len(cached_payloads),
            desc=args.dataset_name,
            dynamic_ncols=True,
            miniters=1,
        ) as pbar:
            postfix = _format_running_avg_postfix(
                f1_sum=f1_sum,
                f1_count=f1_count,
                gpt_sum=gpt_sum,
                gpt_count=gpt_count,
            )
            if postfix:
                pbar.set_postfix_str(postfix, refresh=False)
            for future in as_completed(futures):
                try:
                    payload = future.result()
                except Exception as exc:
                    failures += 1
                    logger.error(
                        "QA worker failed (continuing): {}: {}",
                        type(exc).__name__,
                        exc,
                    )
                else:
                    f1_sum, f1_count, gpt_sum, gpt_count = _accumulate_metrics(
                        payload,
                        f1_sum=f1_sum,
                        f1_count=f1_count,
                        gpt_sum=gpt_sum,
                        gpt_count=gpt_count,
                    )
                    postfix = _format_running_avg_postfix(
                        f1_sum=f1_sum,
                        f1_count=f1_count,
                        gpt_sum=gpt_sum,
                        gpt_count=gpt_count,
                    )
                    if postfix:
                        pbar.set_postfix_str(postfix, refresh=False)
                pbar.update(1)
        if failures:
            logger.error(
                "Finished with {} worker failure(s); re-run to retry missing outputs",
                failures,
            )


if __name__ == "__main__":
    main(tyro.cli(Args, use_underscores=True))
