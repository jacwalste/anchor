"""Shared parsing of raw judge responses, used by extract and verify.

The only tolerated deviation from bare JSON is a markdown code fence —
a deterministic, unambiguous transform. Anything else fails loudly.
"""

import json

from anchor.errors import JudgeResponseError


def parse_json(raw: str) -> object:
    try:
        return json.loads(strip_code_fence(raw))
    except json.JSONDecodeError as exc:
        raise JudgeResponseError("judge response is not valid JSON", raw) from exc


def strip_code_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
    return text
