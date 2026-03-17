from __future__ import annotations

import json
from typing import Any, Dict


def append_jsonl(log_path: str, payload: Dict[str, Any]) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def append_text_trace(trace_path: str, text: str) -> None:
    with open(trace_path, "a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")