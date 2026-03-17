from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict

from schemas import EpisodeSummary

class RelationalMemoryStore:
    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Any]] = {}

    def get_pair_memory(self, pair_id: str) -> Dict[str, Any]:
        return self._data.get(
            pair_id,
            {
                "episodes_played": 0,
                "successful_episodes": 0,
                "deadlocks": 0,
                "convention_notes": [],
                "recent_summaries": [],
            },
        )

    def update_pair_memory(self, pair_id: str, summary: EpisodeSummary) -> None:
        memory = self.get_pair_memory(pair_id)
        memory["episodes_played"] += 1
        if summary.success:
            memory["successful_episodes"] += 1
        if summary.deadlock:
            memory["deadlocks"] += 1

        memory["recent_summaries"].append(asdict(summary))
        memory["recent_summaries"] = memory["recent_summaries"][-5:]

        note = self._infer_convention_note(summary)
        if note:
            memory["convention_notes"].append(note)
            memory["convention_notes"] = memory["convention_notes"][-5:]

        self._data[pair_id] = memory

    @staticmethod
    def _infer_convention_note(summary: EpisodeSummary) -> str:
        if summary.deadlock:
            return "Previous episode deadlocked."
        if summary.success and summary.total_steps <= 8:
            return "Previous episode completed efficiently."
        if summary.collisions > 0:
            return "Previous episode had movement conflicts."
        return ""


MEMORY_STORE = RelationalMemoryStore()