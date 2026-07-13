from __future__ import annotations

from typing import Any

from agent.base import BaseToolAgent
from mcp_servers.wikipedia_offline_client import execute_tool

WIKIPEDIA_OFFLINE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search offline knowledge base and return markdown hits with URLs "
                'and one-line snippets. method="rrf" gives hybrid recall/balance '
                '(default), method="keywords" is lexical exact-match friendly, and '
                'method="vector" is semantic similarity friendly.'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "top_k": {
                        "type": "integer",
                        "description": "Number of hits to return.",
                        "default": 10,
                        "minimum": 1,
                    },
                    "only_title": {
                        "type": "boolean",
                        "description": "Whether to only search article titles.",
                        "default": False,
                    },
                    "method": {
                        "type": "string",
                        "description": (
                            'Search method: "rrf" (hybrid/balanced, recommended '
                            'default), "keywords" (lexical exact-token matching), '
                            '"vector" (semantic embedding similarity).'
                        ),
                        "enum": ["rrf", "keywords", "vector"],
                        "default": "rrf",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": (
                "Open a URL. If URL has #chunk-N suffix, returns that chunk; "
                "remove the #chunk-N suffix to fetch the full article."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to open.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": (
                "Submit the final answer and end the episode: a short exact string, "
                "or one Markdown table when the question requires it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": (
                            "Final answer only (short string or one Markdown table), "
                            "no explanation."
                        ),
                    },
                },
                "required": ["answer"],
            },
        },
    },
]

_SHORT_SCRATCHPAD_SYSTEM_APPEND = """
- Treat `think` as a short scratchpad, not a full chain-of-thought transcript.
- Prefer at most three compact lines in this order: `facts`, `gap`, `next`.
- Keep only durable facts, disambiguation decisions, and the immediate next action.
- Do not restate the question, do not copy long observations, and target roughly 100 tokens.
""".strip()

WIKIPEDIA_OFFLINE_SYSTEM_PROMPT_TEMPLATE = """
You are a research assistant. Answer the user question using provided tools.

Rules:
- Tool observation history is compressed: only the latest {keep_recent_tool_results} tool result(s) stay in full; older tool messages are truncated to at most {tool_result_max_chars} characters each and may no longer be visible in later rounds.
- Before each tool call, write brief reasoning in the assistant message (the text before tool calls). Record durable facts there: extracted field values, disambiguation decisions, entity-page mappings, and your next-step plan. Reasoning stays in history; truncated tool dumps do not—do not rely on long-ago tool text remaining available.
- Keep pre-tool reasoning compact. Do not restate the question, do not paste long tool outputs, and do not add unnecessary narration.
- Use focused searches: one query = one reasoning hop; try not to combine multiple hops in one query.
- You may call multiple independent tools in one step.
- Avoid low-gain searches. Do not re-search the same entity-attribute pair with minor rewording once you already have overlapping results or a relevant page.
- For time-varying facts (e.g., headquarters, ownership, operator, affiliation, location), match the question's time scope.
- Never fabricate URLs; only use URLs from search results.
- If search results or opened pages already provide enough evidence, stop searching and submit.
- You MUST end by calling `submit_answer` with a short exact answer string, or one Markdown table if the question asks for a table.
""".strip()


def build_wikipedia_offline_system_prompt(
    *,
    keep_recent_tool_results: int = 1,
    tool_result_max_chars: int = 2048,
) -> str:
    base = WIKIPEDIA_OFFLINE_SYSTEM_PROMPT_TEMPLATE.format(
        keep_recent_tool_results=keep_recent_tool_results,
        tool_result_max_chars=tool_result_max_chars,
    )
    if _SHORT_SCRATCHPAD_SYSTEM_APPEND in base:
        return base
    return f"{base}\n\n{_SHORT_SCRATCHPAD_SYSTEM_APPEND}"


WIKIPEDIA_OFFLINE_SYSTEM_PROMPT = build_wikipedia_offline_system_prompt()


class WikipediaOfflineAgent(BaseToolAgent):
    def __init__(
        self,
        *,
        qid: str,
        question: str,
        model_name: str,
        server_url: str,
        keep_recent_tool_results: int = 1,
        tool_result_max_chars: int = 2048,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            server_url=server_url,
            qid=qid,
            question=question,
            system_prompt=build_wikipedia_offline_system_prompt(
                keep_recent_tool_results=keep_recent_tool_results,
                tool_result_max_chars=tool_result_max_chars,
            ),
            tools=WIKIPEDIA_OFFLINE_TOOLS,
            tool_executor=self._execute_tool,
            keep_recent_tool_results=keep_recent_tool_results,
            tool_result_max_chars=tool_result_max_chars,
            **kwargs,
        )

    @staticmethod
    def _execute_tool(name: str, arguments_json: str) -> str:
        return execute_tool(name, arguments_json)
