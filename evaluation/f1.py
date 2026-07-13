import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    s = str(s)

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in string.punctuation)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _compute_f1(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return (1.0, 1.0, 1.0) if pred_tokens == gold_tokens else (0.0, 0.0, 0.0)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def _is_bool_answer(s: str) -> bool:
    return normalize_answer(s) in {"yes", "no", "noanswer"}


def f1_score(
    prediction: str,
    ground_truth: str,
    aliases: list[str] | None = None,
) -> dict[str, float]:
    answers = [ground_truth] + list(aliases or [])
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_f1_bool = 0.0
    norm_pred = normalize_answer(prediction)

    for answer in answers:
        f1, precision, recall = _compute_f1(prediction, answer)
        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall

        norm_gold = normalize_answer(answer)
        if _is_bool_answer(norm_pred) or _is_bool_answer(norm_gold):
            f1_bool = 1.0 if norm_pred == norm_gold else 0.0
        else:
            f1_bool = f1
        best_f1_bool = max(best_f1_bool, f1_bool)

    return {
        "f1": best_f1,
        "precision": best_precision,
        "recall": best_recall,
        "f1_bool": best_f1_bool,
    }
