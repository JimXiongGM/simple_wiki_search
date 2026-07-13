from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from agent.llm_client_factory import create_llm_client

from .exact_match import exact_match

JUDGE_MODEL = "gpt-5.4-mini"
JUDGE_TIMEOUT_SEC = 60.0
MAX_ALIASES = 30
CACHE_DIR = Path("~/.cache/simple_wiki/llm_judge").expanduser()
SYSTEM_PROMPT = (
    "You are an expert QA evaluator. Judge factual correctness only; "
    "ignore style, grammar, and punctuation."
)
VERDICT_RE = re.compile(
    r"\b(true|correct|yes)\b|\b(false|incorrect|wrong|no)\b",
    re.IGNORECASE,
)


def _ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


@dataclass(frozen=True)
class JudgeResult:
    """Store judge score, raw response, and cache hit info."""

    score: int | None
    raw: str = ""
    cached: bool = False


def _format_golden_answer(golden_answer: str, aliases: list[str] | None = None) -> str:
    """Format primary answer and aliases for the judge prompt."""
    gold = str(golden_answer or "").strip()
    alias_lines = [str(a).strip() for a in aliases or [] if str(a).strip()]
    if not alias_lines:
        return gold
    lines = [gold] + [f"- {a}" for a in alias_lines[:MAX_ALIASES]]
    return "Golden answer (any line below is acceptable):\n" + "\n".join(lines)


def judge_prompt(
    question: str,
    golden_answer: str,
    prediction: str,
    *,
    aliases: list[str] | None = None,
    extra_note: str = "",
) -> str:
    """Build the binary QA judge prompt."""
    gold_block = _format_golden_answer(golden_answer, aliases)
    note_block = f"\nNote: {extra_note.strip()}\n" if extra_note.strip() else ""
    return (
        "Decide whether the prediction is factually correct.\n"
        "- Credit answers embedded in a longer response.\n"
        "- Treat equivalent dates, spellings, and name variants as correct.\n"
        "- Mark incorrect if it contradicts the golden answer or fails to answer.\n"
        f"{note_block}\n"
        f"Question: {question.strip()}\n"
        f"{gold_block}\n"
        f"Prediction: {prediction.strip()}\n\n"
        "Give one short sentence of reasoning, then write exactly True or False "
        "on the last line."
    )


def parse_verdict(text: str) -> int | None:
    """Parse judge text into 1/0/None."""
    if not text or text.startswith("Error:"):
        return None
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return None
    for token in reversed(lines[-1].split()):
        lowered = token.lower().strip(".,;:!?\"'")
        if lowered in {"true", "correct", "yes"}:
            return 1
        if lowered in {"false", "incorrect", "wrong", "no"}:
            return 0
    matches = list(VERDICT_RE.finditer(text))
    if not matches:
        return None
    return 1 if matches[-1].group(0).lower() in {"true", "correct", "yes"} else 0


def _sample_id(question: str, golden_answer_block: str) -> str:
    """Build a stable sample id for disk cache reuse."""
    payload = f"{question}\0{golden_answer_block}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _cache_path(sample_id: str, prediction: str) -> Path:
    """Build the cache file path for one judge call."""
    model_key = JUDGE_MODEL.replace(".", "-")
    pred_hash = hashlib.sha256(prediction.encode("utf-8")).hexdigest()[:12]
    return _ensure_cache_dir() / f"{model_key}-{sample_id}-{pred_hash}.txt"


def _call_judge(
    question: str,
    golden_answer: str,
    prediction: str,
    *,
    aliases: list[str] | None = None,
    extra_note: str = "",
    use_cache: bool = True,
) -> JudgeResult:
    """Call the judge model; read/write local cache when enabled."""
    if not str(prediction or "").strip():
        return JudgeResult(score=0)

    golden_answer_block = _format_golden_answer(golden_answer, aliases)
    sample_id = _sample_id(question, golden_answer_block)
    cache_file = _cache_path(sample_id, prediction)
    if use_cache and cache_file.exists():
        raw = cache_file.read_text(encoding="utf-8")
        return JudgeResult(score=parse_verdict(raw), raw=raw, cached=True)

    prompt = judge_prompt(
        question,
        golden_answer,
        prediction,
        aliases=aliases,
        extra_note=extra_note,
    )
    try:
        response = create_llm_client(
            model_name=JUDGE_MODEL,
            timeout=JUDGE_TIMEOUT_SEC,
            provider="openai",
        ).chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_completion_tokens=256,
            seed=31415,
        )
        raw = response.choices[0].message.content or ""
    except Exception as exc:
        raw = f"Error: {exc}"
        return JudgeResult(score=None, raw=raw, cached=False)

    if use_cache:
        cache_file.write_text(raw, encoding="utf-8")
    return JudgeResult(score=parse_verdict(raw), raw=raw, cached=False)


def judge_answer(
    question: str,
    golden_answer: str,
    prediction: str,
    *,
    aliases: list[str] | None = None,
    extra_note: str = "",
    use_cache: bool = True,
) -> int | None:
    """Return 1/0/None; skip the judge model when EM matches."""
    if (
        exact_match(prediction or "", golden_answer or "", aliases=aliases).get(
            "em", 0.0
        )
        == 1.0
    ):
        return 1
    return _call_judge(
        question,
        golden_answer,
        prediction,
        aliases=aliases,
        extra_note=extra_note,
        use_cache=use_cache,
    ).score


def compute_gpt_judge(
    question: str,
    golden_answer: str,
    prediction: str,
    aliases: list[str] | None = None,
    golden_structured: dict | list | None = None,
) -> int | None:
    """Unified entry point used by the evaluation package."""
    extra_note = ""
    if isinstance(golden_structured, dict) and golden_structured:
        extra_note = (
            "Structured task: mark correct if every gold key's value appears "
            "factually in the prediction, even with extra compatible details."
        )
    return judge_answer(
        question=question,
        golden_answer=golden_answer,
        prediction=prediction,
        aliases=aliases,
        extra_note=extra_note,
    )
