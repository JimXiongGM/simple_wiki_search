from __future__ import annotations

import re
from collections.abc import Iterable

_UNICODE_SPACES_TO_ASCII_SPACE = re.compile(
    r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]"
)
_ZERO_WIDTH_AND_BOM = re.compile(r"[\u200b\u200c\u200d\ufeff\u2060]")
_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")


def normalize_spaces(text: str) -> str:
    s = (text or "").replace("\r\n", "\n")
    s = _ZERO_WIDTH_AND_BOM.sub("", s)
    s = _UNICODE_SPACES_TO_ASCII_SPACE.sub(" ", s)
    return s


def collapse_newlines(text: str) -> str:
    return _MULTIPLE_NEWLINES.sub("\n\n", text or "")


def drop_empty_md_table_rows(lines: Iterable[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        compact = line.strip().replace(" ", "").replace("\t", "")
        if compact and len(set(compact)) == 1:
            continue
        out.append(line)
    return out


def strip_wikipedia_edit_prefix(lines: Iterable[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        i = 0
        while i < len(line) and line[i] in (" ", "\t"):
            i += 1
        if line[i:].startswith("[edit]"):
            rest = line[i + len("[edit]") :]
            if rest.startswith(" "):
                rest = rest[1:]
            out.append(line[:i] + rest)
        else:
            out.append(line)
    return out


def drop_after_heading(lines: list[str], heading: str) -> list[str]:
    target = (heading or "").strip()
    for idx, line in enumerate(lines):
        if line.strip() == target:
            return lines[:idx]
    return lines
