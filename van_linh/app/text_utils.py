from __future__ import annotations

import re

OPTION_PATTERN = re.compile(
    r"(?ims)(?:^|\n)\s*([ABCD])[\.\):\-]\s*(.+?)(?=(?:\n\s*[ABCD][\.\):\-]\s)|\Z)"
)


def extract_options(question: str) -> dict[str, str]:
    options = {
        label.upper(): " ".join(content.split())
        for label, content in OPTION_PATTERN.findall(question)
    }
    if {"A", "B", "C", "D"}.issubset(options):
        return options
    return {}


def strip_options(question: str) -> str:
    match = OPTION_PATTERN.search(question)
    if not match:
        return question.strip()
    return question[: match.start()].strip()
