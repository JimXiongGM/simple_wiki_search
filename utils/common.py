"""Common utility functions"""

from __future__ import annotations

import re


def colorful(text: str, color: str = "yellow") -> str:
    """Add color to text for terminal output"""
    if color == "yellow":
        text = "\033[1;33m" + str(text) + "\033[0m"
    elif color == "grey":
        text = "\033[1;30m" + str(text) + "\033[0m"
    elif color == "green":
        text = "\033[1;32m" + str(text) + "\033[0m"
    elif color == "orange":
        # 256-color orange
        text = "\033[38;5;208m" + str(text) + "\033[0m"
    elif color == "red":
        text = "\033[1;31m" + str(text) + "\033[0m"
    elif color == "blue":
        text = "\033[1;94m" + str(text) + "\033[0m"
    return text


def extract_toc_from_md(md: str) -> list[dict]:
    """
    Extract markdown table of contents structure, return structured data

    Args:
        md: markdown text

    Returns:
        TOC list, each item contains: number, title, level, start_line, end_line
    """
    lines = md.replace("\r\n", "\n").split("\n")
    counters = [0] * 7
    headings = []

    for i, line in enumerate(lines):
        match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            line_no = i + 1
            headings.append((level, title, line_no))

    if not headings:
        return []

    min_level = min(h[0] for h in headings)
    toc_items = []

    # If there's content before first heading, add "Lead section" item
    if headings and headings[0][2] > 1:
        first_heading_line = headings[0][2]
        has_content = False
        for i in range(first_heading_line - 1):
            if lines[i].strip():
                has_content = True
                break

        if has_content:
            toc_items.append(
                {
                    "number": "0",
                    "title": "Lead section",
                    "level": 0,
                    "start_line": 1,
                    "end_line": first_heading_line - 1,
                }
            )

    for idx, (level, title, line_no) in enumerate(headings):
        counters[level] += 1
        for deeper_level in range(level + 1, 7):
            counters[deeper_level] = 0

        if idx + 1 < len(headings):
            next_level = headings[idx + 1][2]
            end_line = next_level - 1
        else:
            end_line = len(lines)

        number_parts = []
        for lv in range(min_level, level + 1):
            number_parts.append(str(counters[lv]))
        number = ".".join(number_parts)

        indent_level = level - min_level

        toc_items.append(
            {
                "number": number,
                "title": title,
                "level": indent_level,
                "start_line": line_no,
                "end_line": end_line,
            }
        )

    return toc_items
