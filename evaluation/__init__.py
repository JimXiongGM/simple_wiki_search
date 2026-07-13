from .exact_match import exact_match
from .f1 import f1_score

__all__ = ["evaluate", "exact_match", "f1_score", "compute_gpt_judge"]


def compute_gpt_judge(*args, **kwargs):
    """Lazy re-export so importing evaluation does not load the LLM judge."""
    from .llm_judge import compute_gpt_judge as _compute_gpt_judge

    return _compute_gpt_judge(*args, **kwargs)


def evaluate(
    *,
    golden_answer: str,
    prediction: str,
    aliases: list[str] | None = None,
    enable_gpt_judge: bool = False,
    question: str | None = None,
) -> dict[str, float | int]:
    """Compute EM, F1, and optional LLM Judge for standard QA."""
    result: dict[str, float | int] = {}
    result.update(exact_match(prediction, golden_answer, aliases))
    result.update(f1_score(prediction, golden_answer, aliases))
    if enable_gpt_judge and question:
        from .llm_judge import compute_gpt_judge as _compute_gpt_judge

        score = _compute_gpt_judge(
            question=question,
            golden_answer=golden_answer,
            prediction=prediction,
            aliases=aliases,
        )
        if score is not None:
            result["gpt_judge"] = score
    return result
