import re
import string


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


def _em_exact(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def _em_substring(prediction: str, ground_truth: str) -> bool:
    gt = ground_truth
    return gt in prediction or gt.lower() in prediction or gt.capitalize() in prediction


def _em_cover(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def exact_match(
    prediction: str,
    ground_truth: str,
    aliases: list[str] | None = None,
) -> dict[str, float]:
    all_answers = [ground_truth] + list(aliases or [])
    return {
        "em": float(any(_em_exact(prediction, ans) for ans in all_answers)),
        "em_substring": float(
            any(_em_substring(prediction, ans) for ans in all_answers)
        ),
        "em_cover": float(any(_em_cover(prediction, ans) for ans in all_answers)),
    }
