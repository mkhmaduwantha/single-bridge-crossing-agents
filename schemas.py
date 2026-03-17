from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, TypedDict

Position = tuple[int, int]
Move = Literal["UP", "DOWN", "LEFT", "RIGHT", "WAIT"]
AgentName = Literal["A1", "A2"]


@dataclass
class AgentDecision:
    move: Move
    message: str
    justification: str


@dataclass
class LLMTrace:
    prompt: str
    raw_output: str
    parsed_output: Dict[str, Any]


@dataclass
class EpisodeSummary:
    episode_id: str
    scenario: str
    success: bool
    total_steps: int
    deadlock: bool
    collisions: int
    joint_reward: int
    notes: str


class StepRecord(TypedDict):
    step: int
    positions_before: Dict[str, List[int]]
    messages_inbox: Dict[str, List[str]]
    llm_traces: Dict[str, Dict[str, Any]]
    decisions: Dict[str, Dict[str, Any]]
    outcome: Dict[str, Any]
    positions_after: Dict[str, List[int]]
    rewards: Dict[str, int]
    memory_snapshot: Dict[str, Any]


class SimulationState(TypedDict, total=False):
    episode_id: str
    scenario: str
    step: int
    max_steps: int
    done: bool
    deadlock: bool

    positions: Dict[str, Position]
    goals: Dict[str, Position]
    inboxes: Dict[str, List[str]]
    decisions: Dict[str, AgentDecision]
    llm_traces: Dict[str, LLMTrace]

    history: List[StepRecord]
    recent_outcomes: List[str]
    collisions: int
    invalid_moves: int

    rewards: Dict[str, int]
    joint_reward: int
    outcome_text: str

    pair_id: str
    long_term_memory: Dict[str, Any]
    log_path: str
    trace_path: str