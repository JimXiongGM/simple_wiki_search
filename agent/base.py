from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
import openai
from loguru import logger

from agent.llm_client_factory import create_llm_client
from agent.model_compat import mistral_tool_call_violation, prepare_request_messages
from serving.model_family import (
    default_chat_template_kwargs_for_model,
    is_mistral_family_model,
)

ToolExecutor = Callable[[str, str], str]
# Submit validator: takes the answer string from submit_answer; returns None on
# success. A returned string means validation failed; that string is fed back to
# the model as an error so it can fix and resubmit.
SubmitValidator = Callable[[str], Optional[str]]


class FatalToolError(RuntimeError):
    """Tool backend failure that must abort the whole run."""


TOOL_502_MAX_RETRIES = 3
TOOL_502_RETRY_BACKOFF_SECONDS = 1.0
LLM_RETRYABLE_STATUS_CODES = frozenset({502, 503})
TOOL_RETRYABLE_STATUS_CODES = frozenset({502, 503, 504})


def _http_status_code(exc: Exception) -> int | None:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    return None


def _llm_http_status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    return _http_status_code(exc)


def is_retryable_tool_error(exc: Exception) -> bool:
    """Return True for transient tool backend errors worth retrying first."""
    return _http_status_code(exc) in TOOL_RETRYABLE_STATUS_CODES


def is_retryable_llm_error(exc: Exception) -> bool:
    """Return True for transient LLM gateway errors worth retrying within one request."""
    if isinstance(exc, openai.APITimeoutError):
        return True
    code = _llm_http_status_code(exc)
    return code in LLM_RETRYABLE_STATUS_CODES


def is_fatal_tool_error(exc: Exception) -> bool:
    """Return True for infrastructure errors that should not become observations."""
    if isinstance(exc, FatalToolError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        # 4xx is model-fixable; persistent 5xx after retries is fatal.
        return exc.response.status_code >= 500
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.PoolTimeout,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
            httpx.WriteError,
            httpx.ReadError,
        ),
    ):
        return True
    return False


@dataclass
class AgentResult:
    """Unified run result wrapper so different tasks can share the same persistence logic."""

    payload: dict[str, Any]


def truncate_tool_messages(
    messages: List[Dict[str, Any]],
    *,
    tool_result_max_chars: Optional[int] = 2048,
    keep_recent_tool_results: int = 1,
    truncation_suffix: str = "...[truncated {removed} chars]",
) -> List[Dict[str, Any]]:
    """Optionally compress historical tool messages so train/infer stay consistent."""
    if tool_result_max_chars is None or tool_result_max_chars <= 0:
        return list(messages)

    result = list(messages)
    tool_seen = 0
    keep_recent = max(0, int(keep_recent_tool_results))
    for i in range(len(result) - 1, -1, -1):
        msg = result[i]
        if msg.get("role") != "tool":
            continue
        tool_seen += 1
        if tool_seen <= keep_recent:
            continue

        content = msg.get("content")
        if not isinstance(content, str) or len(content) <= tool_result_max_chars:
            continue

        suffix = truncation_suffix.format(
            removed=len(content) - tool_result_max_chars,
            original=len(content),
        )
        result[i] = {**msg, "content": content[:tool_result_max_chars] + suffix}
    return result


class BaseToolAgent:
    """Generic ReAct agent focused on multi-turn dialogue and tool-call orchestration."""

    def __init__(
        self,
        *,
        model_name: str,
        qid: str,
        question: str,
        system_prompt: str,
        tools: list[dict[str, Any]],
        tool_executor: ToolExecutor,
        server_url: str | None = None,  # local models require server_url
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        seed: int = 42,
        answer: str | None = None,
        max_round: int = 10,
        max_tool_calls_per_step: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_completion_tokens: int = 512,
        logprobs: bool = False,
        tool_result_max_chars: int | None = 2048,
        keep_recent_tool_results: int = 1,
        truncation_suffix: str = "...[truncated {removed} chars]",
        enable_round_budget_reminder: bool = True,
        round_budget_reminder_ratio: float = 0.8,
        submit_validator: SubmitValidator | None = None,
    ) -> None:
        self.system_prompt = (system_prompt or "").strip()
        self.tools = tools or []
        self.tool_executor = tool_executor
        self.qid = str(qid).strip()
        self.question = str(question).strip()
        self.answer = answer
        self.model_name = model_name
        self.server_url = server_url
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self.seed = int(seed)
        self.max_round = int(max_round)
        self.max_tool_calls_per_step = int(max_tool_calls_per_step)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_completion_tokens = int(max_completion_tokens)
        self.logprobs = bool(logprobs)
        # Tool-history compression settings:
        # - tool_result_max_chars: max chars kept per tool message; None or <=0 disables compression
        # - keep_recent_tool_results: keep the most recent N tool messages uncompressed
        # - truncation_suffix: suffix appended after truncation; supports {removed} / {original}
        self.tool_result_max_chars = (
            int(tool_result_max_chars)
            if tool_result_max_chars is not None and int(tool_result_max_chars) > 0
            else None
        )
        self.keep_recent_tool_results = max(0, int(keep_recent_tool_results))
        self.truncation_suffix = str(truncation_suffix)
        self.enable_round_budget_reminder = bool(enable_round_budget_reminder)
        # Submit validator: set only for tasks that require structured output
        # (e.g. Markdown tables). On failure, feed the error back so the model
        # can resubmit instead of ending the episode immediately.
        self.submit_validator = submit_validator
        # Clamp to (0, 1); ratios outside this range disable mid-run reminder.
        ratio = float(round_budget_reminder_ratio)
        self.round_budget_reminder_ratio = ratio if 0.0 < ratio < 1.0 else 0.8

        if not self.qid:
            raise ValueError("qid must not be empty")
        if not self.question:
            raise ValueError("question must not be empty")

        self._client = create_llm_client(
            model_name=self.model_name,
            server_url=self.server_url or "http://127.0.0.1:19000",
            provider=self.provider,
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def run(self, debug: bool = False) -> dict[str, Any]:
        """Run the episode. Persistence is the caller's responsibility."""
        return self._run_impl(debug=debug).payload

    def _maybe_append_budget_reminder(
        self,
        messages: list[dict[str, Any]],
        *,
        round_id: int,
        budget_threshold: int,
        budget_reminder_sent: bool,
    ) -> bool:
        if budget_threshold < 0 or round_id < budget_threshold or budget_reminder_sent:
            return budget_reminder_sent
        remaining = self.max_round - round_id
        reminder = (
            f"[Round budget] You have used {round_id} of "
            f"{self.max_round} tool-calling rounds; only "
            f"{remaining} round(s) remain. If you already have "
            f"enough evidence, call the `submit_answer` tool with your final answer; otherwise "
            "prioritize decisive searches and submit before "
            "the budget runs out."
        )
        # mistral_common rejects role=user immediately after role=tool.
        if (
            is_mistral_family_model(self.model_name)
            and messages
            and messages[-1].get("role") == "tool"
        ):
            prev = str(messages[-1].get("content") or "")
            messages[-1] = {
                **messages[-1],
                "content": (prev + "\n\n" + reminder).strip(),
            }
        else:
            messages.append({"role": "user", "content": reminder})
        return True

    def _request_assistant_turn(
        self,
        messages: list[dict[str, Any]],
        *,
        usage_total: dict[str, int],
        debug: bool,
        round_id: int,
    ) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        """One model call: truncate history, request, append assistant message."""
        request_messages = truncate_tool_messages(
            messages,
            tool_result_max_chars=self.tool_result_max_chars,
            keep_recent_tool_results=self.keep_recent_tool_results,
            truncation_suffix=self.truncation_suffix,
        )
        assistant_begin = time.time()
        response = self._create_chat_completion_with_retries(messages=request_messages)
        assistant_duration_s = round(time.time() - assistant_begin, 3)
        self.seed += 1

        response_dict = json.loads(response.model_dump_json())
        self._accumulate_usage(usage_total, response_dict.get("usage"))
        assistant_message = response_dict["choices"][0]["message"]
        assistant_message["duration_s"] = assistant_duration_s

        tool_calls = list(assistant_message.get("tool_calls") or [])
        if len(tool_calls) > self.max_tool_calls_per_step:
            tool_calls = tool_calls[: self.max_tool_calls_per_step]
            assistant_message["tool_calls"] = tool_calls
        content = str(assistant_message.get("content") or "").strip()
        tool_calls = list(assistant_message.get("tool_calls") or [])

        # Mistral cannot echo illegal tool_calls on the next turn. Drop them and
        # surface the real validator text on the assistant message (not role=user:
        # user after tool_calls / tool breaks mistral_common role order).
        violations: list[str] = []
        if tool_calls and is_mistral_family_model(self.model_name):
            legal_calls: list[dict[str, Any]] = []
            for tool_call in tool_calls:
                violation = mistral_tool_call_violation(tool_call)
                if violation is None:
                    legal_calls.append(tool_call)
                else:
                    violations.append(violation)
            if violations:
                logger.warning(
                    f"[qid={self.qid}] rejected {len(violations)} illegal mistral "
                    f"tool_call(s); keeping {len(legal_calls)} legal call(s)"
                )
                tool_calls = legal_calls
                if legal_calls:
                    assistant_message["tool_calls"] = legal_calls
                else:
                    assistant_message.pop("tool_calls", None)
                notice = (
                    "[Invalid tool call rejected by serving validator]\n"
                    + "\n".join(f"- {item}" for item in violations)
                    + "\nFix the tool call and try again."
                )
                merged = f"{content}\n\n{notice}".strip() if content else notice
                assistant_message["content"] = merged
                content = merged

        messages.append(assistant_message)
        if debug:
            logger.info(
                f"[qid={self.qid}] round={round_id} content={content or '∅'} tool_calls={len(tool_calls)}"
            )
        return assistant_message, content, tool_calls

    def _invoke_one_tool_call(
        self, tool_call: dict[str, Any]
    ) -> tuple[dict[str, Any], str, str, str, float]:
        """Run a single tool call; returns (tool_call, name, args, output, duration_s)."""
        fn = tool_call.get("function") or {}
        tool_name = str(fn.get("name") or "").strip()
        arguments_json = str(fn.get("arguments") or "{}")
        tool_begin = time.time()
        try:
            tool_output = self._execute_tool_with_retries(tool_name, arguments_json)
        except Exception as e:
            if is_fatal_tool_error(e):
                logger.error(
                    f"[qid={self.qid}] fatal tool backend failure: tool={tool_name}, arguments={arguments_json}, error={type(e).__name__}: {e}"
                )
                raise FatalToolError(
                    f"fatal tool backend failure: qid={self.qid}, "
                    f"tool={tool_name}, error={type(e).__name__}: {e}"
                ) from e
            tool_output = json.dumps(
                {
                    "status": "error",
                    "tool": tool_name,
                    "arguments": arguments_json,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "hint": (
                        "Fix tool arguments and call tool again. "
                        "For search, provide non-empty 'query'."
                    ),
                },
                ensure_ascii=False,
            )
            logger.warning(
                f"[qid={self.qid}] tool call failed: tool={tool_name}, arguments={arguments_json}, error={type(e).__name__}: {e}"
            )
        return (
            tool_call,
            tool_name,
            arguments_json,
            tool_output,
            round(time.time() - tool_begin, 3),
        )

    def _handle_tool_calls(
        self,
        messages: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]],
        *,
        debug: bool,
    ) -> tuple[str, str | None]:
        """Execute tool calls in parallel; return (predicted_answer, stop_reason)."""
        predicted_answer = ""
        stop_reason: str | None = None
        if not tool_calls:
            return predicted_answer, stop_reason

        if len(tool_calls) == 1:
            results = [self._invoke_one_tool_call(tool_calls[0])]
        else:
            with ThreadPoolExecutor(max_workers=len(tool_calls)) as pool:
                results = list(pool.map(self._invoke_one_tool_call, tool_calls))

        for tool_call, tool_name, arguments_json, tool_output, duration_s in results:
            if debug:
                logger.info(f"[tool={tool_name}, arguments={arguments_json}]")
                print(tool_output)

            tool_message = {
                "tool_call_id": tool_call.get("id"),
                "role": "tool",
                "name": tool_name,
                "content": tool_output,
                "duration_s": duration_s,
            }
            messages.append(tool_message)

            done_answer = self._extract_done_answer(tool_output)
            if done_answer is None:
                continue
            reject_msg = self._validate_submission(done_answer)
            if reject_msg is None:
                return done_answer, "Tool submitted final answer"
            rejected_output = json.dumps(
                {
                    "status": "error",
                    "tool": "submit_answer",
                    "error": reject_msg,
                    "hint": (
                        "Your answer was NOT accepted. Fix the format "
                        "and call submit_answer again."
                    ),
                },
                ensure_ascii=False,
            )
            tool_message["content"] = rejected_output
            logger.warning(f"[qid={self.qid}] submit rejected: {reject_msg}")
        return predicted_answer, stop_reason

    def _nudge_for_tool_use(
        self, messages: list[dict[str, Any]], *, content: str
    ) -> None:
        if content:
            nudge = (
                "Plain-text responses are ignored. "
                f"Please call the `submit_answer` tool with your final answer, or call another tool to continue "
                "investigating."
            )
        else:
            nudge = (
                "You must use tools to proceed. "
                f"When you are ready to finalize, call the `submit_answer` tool with your final answer. "
                "Do not answer in plain text."
            )
        messages.append({"role": "user", "content": nudge})

    def _build_result_payload(
        self,
        *,
        messages: list[dict[str, Any]],
        start_time: float,
        stop_reason: str,
        call_num: int,
        predicted_answer: str,
        usage_total: dict[str, int],
    ) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "answer": self.answer,
            "model_name": self.model_name,
            "messages": messages,
            "tools": self.tools,
            "time_cost": time.time() - start_time,
            "stop_reason": stop_reason,
            "call_num": call_num,
            "predicted_answer": predicted_answer,
            "usage": usage_total,
        }

    def _run_impl(self, *, debug: bool) -> AgentResult:
        """Main loop: request the model, run tools, append messages, and decide when to stop."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.question},
        ]
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        start_time = time.time()
        call_num = 0
        predicted_answer = ""
        stop_reason: str | None = None

        if self.enable_round_budget_reminder and self.max_round >= 5:
            budget_threshold = int(self.max_round * self.round_budget_reminder_ratio)
        else:
            budget_threshold = -1
        budget_reminder_sent = False

        for round_id in range(self.max_round):
            budget_reminder_sent = self._maybe_append_budget_reminder(
                messages,
                round_id=round_id,
                budget_threshold=budget_threshold,
                budget_reminder_sent=budget_reminder_sent,
            )
            _assistant_message, content, tool_calls = self._request_assistant_turn(
                messages,
                usage_total=usage_total,
                debug=debug,
                round_id=round_id,
            )
            call_num += 1

            if tool_calls:
                predicted_answer, stop_reason = self._handle_tool_calls(
                    messages, tool_calls, debug=debug
                )
                if stop_reason is not None:
                    break
                continue

            self._nudge_for_tool_use(messages, content=content)

        if stop_reason is None:
            stop_reason = "Max round num reached"

        return AgentResult(
            payload=self._build_result_payload(
                messages=messages,
                start_time=start_time,
                stop_reason=stop_reason,
                call_num=call_num,
                predicted_answer=predicted_answer,
                usage_total=usage_total,
            )
        )

    def _create_chat_completion_with_retries(
        self, *, messages: list[dict[str, Any]]
    ) -> Any:
        """Retry transient 502/503 LLM gateway errors before failing the episode."""
        request_messages = prepare_request_messages(self.model_name, messages)
        chat_template_kwargs = default_chat_template_kwargs_for_model(self.model_name)
        extra_body = (
            {"chat_template_kwargs": chat_template_kwargs}
            if chat_template_kwargs
            else None
        )
        for retry_idx in range(TOOL_502_MAX_RETRIES + 1):
            try:
                request_kwargs = {
                    "model": self.model_name,
                    "messages": request_messages,
                    "tools": self.tools,
                    "tool_choice": "auto",
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_completion_tokens": self.max_completion_tokens,
                    "seed": self.seed,
                    "n": 1,
                    "logprobs": self.logprobs,
                }
                if extra_body:
                    request_kwargs["extra_body"] = extra_body
                return self._client.chat.completions.create(**request_kwargs)
            except Exception as exc:
                if not is_retryable_llm_error(exc):
                    raise
                if retry_idx >= TOOL_502_MAX_RETRIES:
                    raise
                logger.warning(
                    f"[qid={self.qid}] LLM backend returned {_llm_http_status_code(exc)}; retry "
                    f"{retry_idx + 1}/{TOOL_502_MAX_RETRIES}: "
                    f"error={type(exc).__name__}: {exc}"
                )
                time.sleep(TOOL_502_RETRY_BACKOFF_SECONDS)

        raise AssertionError("unreachable")

    def _execute_tool_with_retries(self, tool_name: str, arguments_json: str) -> str:
        """Retry transient 502s; persistent backend failures must abort the run."""
        for retry_idx in range(TOOL_502_MAX_RETRIES + 1):
            try:
                return self.tool_executor(tool_name, arguments_json)
            except Exception as exc:
                if not is_retryable_tool_error(exc):
                    raise
                if retry_idx >= TOOL_502_MAX_RETRIES:
                    raise
                logger.warning(
                    f"[qid={self.qid}] tool backend returned {_http_status_code(exc)}; retry "
                    f"{retry_idx + 1}/{TOOL_502_MAX_RETRIES}: "
                    f"tool={tool_name}, arguments={arguments_json}, "
                    f"error={type(exc).__name__}: {exc}"
                )
                time.sleep(TOOL_502_RETRY_BACKOFF_SECONDS)

        raise AssertionError("unreachable")

    def _validate_submission(self, answer: str) -> str | None:
        """Validate a submit_answer payload; None means pass, a string is the reject reason.

        No validator (plain short-answer tasks) always passes. If the validator
        itself raises, allow the submission so a validator bug cannot stall an
        episode that is otherwise ready to end.
        """
        if self.submit_validator is None:
            return None
        try:
            return self.submit_validator(answer)
        except Exception as e:
            logger.error(f"[qid={self.qid}] submit_validator raised: {e!r}")
            return None

    @staticmethod
    def _extract_done_answer(tool_output: str) -> str | None:
        """Early-stop when a tool returns {"status":"done","answer":"..."}."""
        if not tool_output:
            return None
        try:
            payload = json.loads(tool_output)
        except json.JSONDecodeError:
            return None
        if payload.get("status") != "done":
            return None
        answer = payload.get("answer")
        return str(answer) if answer is not None else ""

    @staticmethod
    def _accumulate_usage(usage_total: dict[str, int], usage: Any) -> None:
        """Accumulate token usage, tolerant of varying backend usage shapes."""
        if not isinstance(usage, dict):
            return
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage_total[k] += int(usage.get(k) or 0)
