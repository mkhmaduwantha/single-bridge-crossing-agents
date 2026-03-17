from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

GRID_W = 7
GRID_H = 5
MAX_STEPS = 35
CORRIDOR_Y = 2
CORRIDOR_XS = {2, 3, 4}
VALID_MOVES = {"UP", "DOWN", "LEFT", "RIGHT", "WAIT"}

Scenario = Literal["one_off_selfish", "repeated_selfish", "group_priority"]

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

LOG_DIR = Path("simulation_logs")
TRACE_DIR = LOG_DIR / "llm_traces"
JSONL_DIR = LOG_DIR / "jsonl"

LOG_DIR.mkdir(exist_ok=True)
TRACE_DIR.mkdir(exist_ok=True)
JSONL_DIR.mkdir(exist_ok=True)

EPISODES_PER_SCENARIO = 2