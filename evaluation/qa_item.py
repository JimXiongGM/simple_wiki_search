"""QA dataset item helpers and shared TypedDict shape."""

from __future__ import annotations

from typing import Any, TypedDict

from evaluation import evaluate


class QaItem(TypedDict, total=False):
    """Loosely typed QA row across HotpotQA / PopQA / MuSiQue / etc."""

    id: str
    qid: str
    question: str
    prompt: str
    answer: Any
    answers: list[Any]
    answer_aliases: list[str]
    type: str
    prop: str


def item_has_gold_answer(item: QaItem) -> bool:
    """Return True if the item has a gold answer for evaluation."""
    if item.get("answer") is not None:
        return True
    answers = item.get("answers")
    return isinstance(answers, list) and bool(answers)


def item_golden_answer(item: QaItem) -> str:
    """Return the primary gold answer text."""
    if item.get("answer") is not None:
        return str(item["answer"])
    answers = item.get("answers")
    if isinstance(answers, list) and answers:
        return str(answers[0])
    raise ValueError("item has no answer or answers")


def item_answer_aliases(item: QaItem) -> list[str]:
    """Return acceptable aliases besides the primary answer."""
    raw_answers = item.get("answers")
    if isinstance(raw_answers, list):
        texts = [
            str(x).strip() for x in raw_answers if x is not None and str(x).strip()
        ]
        if item.get("answer") is not None:
            return texts
        return texts[1:]
    raw_aliases = item.get("answer_aliases")
    if not isinstance(raw_aliases, list):
        return []
    return [str(x).strip() for x in raw_aliases if x is not None and str(x).strip()]


def evaluate_item_metrics(
    item: QaItem,
    prediction: str,
    *,
    question: str | None = None,
    enable_llm_judge: bool = False,
) -> dict[str, float | int] | None:
    """Compute per-item metrics; F1 is 0.0 when prediction is empty."""
    if not item_has_gold_answer(item):
        return None
    q = question if question is not None else str(item.get("question") or "")
    return evaluate(
        golden_answer=item_golden_answer(item),
        prediction=str(prediction or ""),
        aliases=item_answer_aliases(item),
        enable_gpt_judge=enable_llm_judge and bool(q),
        question=q,
    )
