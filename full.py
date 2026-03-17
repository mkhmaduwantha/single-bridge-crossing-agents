# Full modular code

Use this exact project structure.

```text
bridge_simulation_with_right_of_way/
├── app.py
├── config.py
├── schemas.py
├── memory_store.py
├── environment.py
├── prompts.py
├── llm_agent.py
├── logging_utils.py
├── graph_builder.py
└── runner.py
```

---

## `config.py`

```python
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

GRID_W = 7
GRID_H = 5
MAX_STEPS = 30
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
```

---

## `schemas.py`

```python
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
```

---

## `memory_store.py`

```python
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
            return "Previous episode deadlocked; stronger turn-taking or yielding may help."
        if summary.success and summary.total_steps <= 8:
            return "Previous episode completed efficiently; preserving the same convention may help."
        if summary.collisions > 0:
            return "Previous episode had movement conflicts; clearer sequencing may help."
        return ""


MEMORY_STORE = RelationalMemoryStore()
```

---

## `environment.py`

```python
from __future__ import annotations

from typing import Any, Dict, List

from config import CORRIDOR_XS, CORRIDOR_Y, GRID_H, GRID_W
from schemas import AgentDecision, Position


BLOCKED_CELLS = {
    (x, y)
    for x in range(GRID_W)
    for y in range(GRID_H)
    if y != CORRIDOR_Y
}


def in_bounds(pos: Position) -> bool:
    x, y = pos
    return 0 <= x < GRID_W and 0 <= y < GRID_H


def is_corridor_cell(pos: Position) -> bool:
    x, y = pos
    return y == CORRIDOR_Y and x in CORRIDOR_XS


def is_blocked_cell(pos: Position) -> bool:
    return pos in BLOCKED_CELLS


def describe_grid(positions: Dict[str, Position], goals: Dict[str, Position]) -> str:
    blocked_descriptions: List[str] = []
    for y in range(GRID_H):
        xs = [x for x in range(GRID_W) if (x, y) in BLOCKED_CELLS]
        if xs:
            blocked_descriptions.append(f"row y={y} is blocked at x={xs}")

    traversable_cells = [(x, CORRIDOR_Y) for x in range(GRID_W)]
    corridor_cells = [(x, CORRIDOR_Y) for x in sorted(CORRIDOR_XS)]

    return (
        f"The world is a {GRID_W} by {GRID_H} grid. "
        f"The only traversable row is y={CORRIDOR_Y}, so agents cannot route around the shared passage through upper or lower rows. "
        f"Traversable cells are exactly: {traversable_cells}. "
        f"The shared narrow corridor cells are: {corridor_cells}. "
        f"Blocked cells are described as follows: {'; '.join(blocked_descriptions)}. "
        f"A1 is currently at {positions['A1']} and wants to reach {goals['A1']}. "
        f"A2 is currently at {positions['A2']} and wants to reach {goals['A2']}."
    )


def apply_move(pos: Position, move: str) -> Position:
    x, y = pos
    if move == "UP":
        return (x, y + 1)
    if move == "DOWN":
        return (x, y - 1)
    if move == "LEFT":
        return (x - 1, y)
    if move == "RIGHT":
        return (x + 1, y)
    return pos


def resolve_joint_action(
    positions: Dict[str, Position],
    decisions: Dict[str, AgentDecision],
) -> Dict[str, Any]:
    a1_from = positions["A1"]
    a2_from = positions["A2"]
    a1_to = apply_move(a1_from, decisions["A1"].move)
    a2_to = apply_move(a2_from, decisions["A2"].move)

    invalids = []
    collisions = []

    if not in_bounds(a1_to):
        invalids.append("A1 out_of_bounds")
        a1_to = a1_from
    if not in_bounds(a2_to):
        invalids.append("A2 out_of_bounds")
        a2_to = a2_from

    if is_blocked_cell(a1_to):
        invalids.append("A1 blocked_cell")
        a1_to = a1_from
    if is_blocked_cell(a2_to):
        invalids.append("A2 blocked_cell")
        a2_to = a2_from

    if a1_to == a2_to and a1_to != a1_from and a2_to != a2_from:
        collisions.append("same_cell")
        a1_to = a1_from
        a2_to = a2_from
    elif a1_to == a2_from and a2_to == a1_from and a1_to != a1_from:
        collisions.append("swap")
        a1_to = a1_from
        a2_to = a2_from

    entering_a1 = is_corridor_cell(a1_to) and not is_corridor_cell(a1_from)
    entering_a2 = is_corridor_cell(a2_to) and not is_corridor_cell(a2_from)
    if entering_a1 and entering_a2:
        collisions.append("simultaneous_corridor_entry")
        a1_to = a1_from
        a2_to = a2_from

    return {
        "new_positions": {"A1": a1_to, "A2": a2_to},
        "invalids": invalids,
        "collisions": collisions,
    }


def at_goal(positions: Dict[str, Position], goals: Dict[str, Position]) -> bool:
    return positions["A1"] == goals["A1"] and positions["A2"] == goals["A2"]


def score_step(
    scenario: str,
    positions_after: Dict[str, Position],
    goals: Dict[str, Position],
    collision_count: int,
    invalid_count: int,
    finished: bool,
) -> Dict[str, int]:
    rewards = {"A1": 0, "A2": 0}

    if scenario in ("one_off_selfish", "repeated_selfish"):
        for agent in ["A1", "A2"]:
            rewards[agent] -= 1
            if positions_after[agent] == goals[agent]:
                rewards[agent] += 10
        penalty = 3 * (collision_count + invalid_count)
        rewards["A1"] -= penalty
        rewards["A2"] -= penalty
    elif scenario == "group_priority":
        group_delta = -2
        if finished:
            group_delta += 20
        group_delta -= 8 * (collision_count + invalid_count)
        rewards["A1"] += group_delta
        rewards["A2"] += group_delta

    return rewards
```

---

## `prompts.py`

```python
from __future__ import annotations

import json

from config import CORRIDOR_XS, CORRIDOR_Y, GRID_H, GRID_W
from environment import describe_grid
from schemas import AgentName, SimulationState


def build_agent_prompt(agent: AgentName, state: SimulationState) -> str:
    other = "A2" if agent == "A1" else "A1"
    positions = state["positions"]
    goals = state["goals"]
    scenario = state["scenario"]
    recent = state.get("recent_outcomes", [])[-3:]
    inbox = state.get("inboxes", {}).get(agent, [])[-3:]
    long_mem = state.get("long_term_memory", {})

    scenario_text = {
        "one_off_selfish": (
            "This is a one-time encounter. You will not meet again after this episode. Maximise your own reward."
        ),
        "repeated_selfish": (
            "You and the other agent will encounter each other repeatedly over many future episodes. Your own cumulative reward matters over time. Stable conventions may be useful."
        ),
        "group_priority": (
            "Prioritise the joint outcome over your personal outcome. Minimise total delay and avoid conflict so both agents succeed efficiently."
        ),
    }[scenario]

    output_schema = {
        "move": "UP | DOWN | LEFT | RIGHT | WAIT",
        "message": "short message to the other agent",
        "justification": "brief reason grounded in the current state",
    }

    return f"""
You are {agent} in a two-agent coordination simulation.

OBJECTIVE:
- Your current position: {positions[agent]}
- Your goal position: {goals[agent]}
- Other agent position: {positions[other]}
- Other agent goal: {goals[other]}
- {scenario_text}

WORLD:
- Grid size: {GRID_W}x{GRID_H}
- Coordinates are (x, y)
- x increases to the RIGHT
- y increases UP
- Shared corridor cells are exactly: {sorted(list(CORRIDOR_XS))} on row y={CORRIDOR_Y}
- All rows other than y={CORRIDOR_Y} are blocked, so there is no alternate route around the corridor
- Communication is allowed every step
- Moves happen simultaneously after both agents decide
- Invalid or conflicting moves can waste time and reduce reward

WORLD DESCRIPTION:
{describe_grid(positions, goals)}

RECENT STEP OUTCOMES:
{json.dumps(recent, indent=2) if recent else '[]'}

MESSAGES RECEIVED FROM OTHER AGENT:
{json.dumps(inbox, indent=2) if inbox else '[]'}

LONG-TERM RELATIONAL MEMORY:
{json.dumps(long_mem, indent=2)}

DECISION INSTRUCTIONS:
- Choose exactly one move.
- Send a short message proposing, confirming, or adjusting a coordination plan.
- Your message should match your intended move.
- Keep the justification short and concrete.
- Do not output any text outside the JSON object.

OUTPUT JSON SCHEMA:
{json.dumps(output_schema, indent=2)}
""".strip()
```

---

## `llm_agent.py`

```python
from __future__ import annotations

import json
from typing import Optional, Tuple

from langchain_ollama import ChatOllama

from config import DEFAULT_BASE_URL, DEFAULT_MODEL, VALID_MOVES
from prompts import build_agent_prompt
from schemas import AgentDecision, AgentName, LLMTrace, SimulationState


def make_llm(model_name: str = DEFAULT_MODEL, base_url: Optional[str] = DEFAULT_BASE_URL) -> ChatOllama:
    kwargs = {
        "model": model_name,
        "temperature": 0,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOllama(**kwargs)


def parse_decision(text: str) -> AgentDecision:
    try:
        payload = json.loads(text)
        move = str(payload.get("move", "WAIT")).upper()
        if move not in VALID_MOVES:
            move = "WAIT"
        return AgentDecision(
            move=move,
            message=str(payload.get("message", ""))[:300],
            justification=str(payload.get("justification", ""))[:400],
        )
    except Exception:
        return AgentDecision(
            move="WAIT",
            message="I could not produce a valid structured reply this step.",
            justification="Fallback to WAIT because the response was not valid JSON.",
        )


def query_agent(agent: AgentName, state: SimulationState, llm: ChatOllama) -> Tuple[AgentDecision, LLMTrace]:
    prompt = build_agent_prompt(agent, state)
    response = llm.invoke(prompt)
    raw = response.content if isinstance(response.content, str) else str(response.content)
    decision = parse_decision(raw)
    trace = LLMTrace(
        prompt=prompt,
        raw_output=raw,
        parsed_output={
            "move": decision.move,
            "message": decision.message,
            "justification": decision.justification,
        },
    )
    return decision, trace
```

---

## `logging_utils.py`

```python
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
```

---

## `graph_builder.py`

```python
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from environment import at_goal, resolve_joint_action, score_step
from llm_agent import make_llm, query_agent
from logging_utils import append_jsonl, append_text_trace
from schemas import SimulationState, StepRecord


def node_agent_deliberation(state: SimulationState) -> Dict[str, Any]:
    llm = make_llm()
    a1_decision, a1_trace = query_agent("A1", state, llm)
    a2_decision, a2_trace = query_agent("A2", state, llm)

    append_jsonl(
        state["log_path"],
        {
            "type": "llm_step_trace",
            "episode_id": state["episode_id"],
            "step": state["step"],
            "traces": {
                "A1": asdict(a1_trace),
                "A2": asdict(a2_trace),
            },
        },
    )

    trace_text = f"""
================================================================================
EPISODE {state['episode_id']} | STEP {state['step']}
================================================================================

--- A1 PROMPT ---
{a1_trace.prompt}

--- A1 RAW OUTPUT ---
{a1_trace.raw_output}

--- A1 PARSED OUTPUT ---
{a1_trace.parsed_output}

--- A2 PROMPT ---
{a2_trace.prompt}

--- A2 RAW OUTPUT ---
{a2_trace.raw_output}

--- A2 PARSED OUTPUT ---
{a2_trace.parsed_output}

"""
    append_text_trace(state["trace_path"], trace_text)

    return {
        "decisions": {
            "A1": a1_decision,
            "A2": a2_decision,
        },
        "llm_traces": {
            "A1": a1_trace,
            "A2": a2_trace,
        },
    }


def node_environment_step(state: SimulationState) -> Dict[str, Any]:
    positions_before = state["positions"]
    goals = state["goals"]
    decisions = state["decisions"]

    resolved = resolve_joint_action(positions_before, decisions)
    positions_after = resolved["new_positions"]
    collision_count = len(resolved["collisions"])
    invalid_count = len(resolved["invalids"])
    finished = at_goal(positions_after, goals)

    rewards_delta = score_step(
        scenario=state["scenario"],
        positions_after=positions_after,
        goals=goals,
        collision_count=collision_count,
        invalid_count=invalid_count,
        finished=finished,
    )

    rewards = dict(state["rewards"])
    rewards["A1"] += rewards_delta["A1"]
    rewards["A2"] += rewards_delta["A2"]
    joint_reward = rewards["A1"] + rewards["A2"]

    outcome_text = (
        f"Step {state['step']}: "
        f"A1={decisions['A1'].move}->{positions_after['A1']}, "
        f"A2={decisions['A2'].move}->{positions_after['A2']}, "
        f"collisions={resolved['collisions']}, invalids={resolved['invalids']}"
    )

    next_inboxes = {
        "A1": [f"A2 says: {decisions['A2'].message}"],
        "A2": [f"A1 says: {decisions['A1'].message}"],
    }

    history = list(state.get("history", []))
    step_record: StepRecord = {
        "step": state["step"],
        "positions_before": {k: list(v) for k, v in positions_before.items()},
        "messages_inbox": state.get("inboxes", {}),
        "llm_traces": {
            "A1": asdict(state["llm_traces"]["A1"]),
            "A2": asdict(state["llm_traces"]["A2"]),
        },
        "decisions": {
            "A1": asdict(decisions["A1"]),
            "A2": asdict(decisions["A2"]),
        },
        "outcome": {
            "collisions": resolved["collisions"],
            "invalids": resolved["invalids"],
            "finished": finished,
            "text": outcome_text,
        },
        "positions_after": {k: list(v) for k, v in positions_after.items()},
        "rewards": rewards,
        "memory_snapshot": state.get("long_term_memory", {}),
    }
    history.append(step_record)

    append_jsonl(
        state["log_path"],
        {
            "type": "step",
            "episode_id": state["episode_id"],
            "payload": step_record,
        },
    )

    append_text_trace(
        state["trace_path"],
        (
            f"--- STEP {state['step']} OUTCOME ---\n"
            f"Positions before: {positions_before}\n"
            f"Decisions: A1={asdict(decisions['A1'])}, A2={asdict(decisions['A2'])}\n"
            f"Positions after: {positions_after}\n"
            f"Collisions: {resolved['collisions']}\n"
            f"Invalids: {resolved['invalids']}\n"
            f"Rewards: {rewards}\n"
            f"Finished: {finished}\n\n"
        ),
    )

    deadlock = state["step"] >= state["max_steps"] - 1 and not finished

    return {
        "positions": positions_after,
        "inboxes": next_inboxes,
        "history": history,
        "recent_outcomes": (state.get("recent_outcomes", []) + [outcome_text])[-5:],
        "collisions": state.get("collisions", 0) + collision_count,
        "invalid_moves": state.get("invalid_moves", 0) + invalid_count,
        "rewards": rewards,
        "joint_reward": joint_reward,
        "outcome_text": outcome_text,
        "done": finished or deadlock,
        "deadlock": deadlock,
        "step": state["step"] + 1,
    }


def route_after_step(state: SimulationState) -> str:
    return END if state.get("done", False) else "agent_deliberation"


def build_graph():
    builder = StateGraph(SimulationState)
    builder.add_node("agent_deliberation", node_agent_deliberation)
    builder.add_node("environment_step", node_environment_step)

    builder.add_edge(START, "agent_deliberation")
    builder.add_edge("agent_deliberation", "environment_step")
    builder.add_conditional_edges("environment_step", route_after_step)

    return builder.compile(checkpointer=InMemorySaver())
```

---

## `runner.py`

```python
from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from typing import Optional

from config import CORRIDOR_Y, JSONL_DIR, MAX_STEPS, Scenario, TRACE_DIR
from environment import at_goal, describe_grid
from graph_builder import build_graph
from logging_utils import append_jsonl, append_text_trace
from memory_store import MEMORY_STORE
from schemas import EpisodeSummary, SimulationState

GRAPH = build_graph()


def initial_state(scenario: Scenario, pair_id: str) -> SimulationState:
    episode_id = str(uuid.uuid4())[:8]
    log_path = str(JSONL_DIR / f"episode_{episode_id}.jsonl")
    trace_path = str(TRACE_DIR / f"episode_{episode_id}_trace.txt")
    memory = MEMORY_STORE.get_pair_memory(pair_id)

    state: SimulationState = {
        "episode_id": episode_id,
        "scenario": scenario,
        "step": 0,
        "max_steps": MAX_STEPS,
        "done": False,
        "deadlock": False,
        "positions": {"A1": (0, CORRIDOR_Y), "A2": (6, CORRIDOR_Y)},
        "goals": {"A1": (6, CORRIDOR_Y), "A2": (0, CORRIDOR_Y)},
        "inboxes": {"A1": [], "A2": []},
        "history": [],
        "recent_outcomes": [],
        "collisions": 0,
        "invalid_moves": 0,
        "rewards": {"A1": 0, "A2": 0},
        "joint_reward": 0,
        "outcome_text": "",
        "pair_id": pair_id,
        "long_term_memory": memory,
        "log_path": log_path,
        "trace_path": trace_path,
    }

    append_jsonl(
        log_path,
        {
            "type": "episode_start",
            "episode_id": episode_id,
            "scenario": scenario,
            "pair_id": pair_id,
            "initial_positions": {k: list(v) for k, v in state["positions"].items()},
            "goals": {k: list(v) for k, v in state["goals"].items()},
            "world_description": describe_grid(state["positions"], state["goals"]),
            "long_term_memory": memory,
        },
    )

    append_text_trace(
        trace_path,
        (
            f"EPISODE START\n"
            f"Episode ID: {episode_id}\n"
            f"Scenario: {scenario}\n"
            f"Pair ID: {pair_id}\n"
            f"World: {describe_grid(state['positions'], state['goals'])}\n"
            f"Long-term memory: {json.dumps(memory, indent=2)}\n\n"
        ),
    )
    return state


def summarize_episode(final_state: SimulationState) -> EpisodeSummary:
    success = at_goal(final_state["positions"], final_state["goals"])
    notes = []
    if final_state["collisions"] > 0:
        notes.append("There were movement conflicts.")
    if final_state["deadlock"]:
        notes.append("The episode ended without both agents reaching their goals.")
    if success and final_state["step"] <= 8:
        notes.append("The agents completed the crossing efficiently.")
    if not notes:
        notes.append("The episode completed without especially notable events.")

    return EpisodeSummary(
        episode_id=final_state["episode_id"],
        scenario=final_state["scenario"],
        success=success,
        total_steps=final_state["step"],
        deadlock=final_state["deadlock"],
        collisions=final_state["collisions"],
        joint_reward=final_state["joint_reward"],
        notes=" ".join(notes),
    )


def run_episode(
    scenario: Scenario,
    pair_id: str = "A1_A2",
    thread_id: Optional[str] = None,
) -> SimulationState:
    state = initial_state(scenario, pair_id)
    config = {
        "configurable": {"thread_id": thread_id or state["episode_id"]},
        "recursion_limit": max(100, state["max_steps"] * 3),
    }
    final_state = GRAPH.invoke(state, config=config)

    summary = summarize_episode(final_state)
    MEMORY_STORE.update_pair_memory(pair_id, summary)

    append_jsonl(
        final_state["log_path"],
        {
            "type": "episode_end",
            "episode_id": final_state["episode_id"],
            "summary": asdict(summary),
            "final_positions": {k: list(v) for k, v in final_state["positions"].items()},
            "final_rewards": final_state["rewards"],
            "updated_pair_memory": MEMORY_STORE.get_pair_memory(pair_id),
        },
    )

    append_text_trace(
        final_state["trace_path"],
        (
            f"EPISODE END\n"
            f"Final positions: {final_state['positions']}\n"
            f"Final rewards: {final_state['rewards']}\n"
            f"Joint reward: {final_state['joint_reward']}\n"
            f"Collisions: {final_state['collisions']}\n"
            f"Invalid moves: {final_state['invalid_moves']}\n"
            f"Deadlock: {final_state['deadlock']}\n"
            f"Updated pair memory: {json.dumps(MEMORY_STORE.get_pair_memory(pair_id), indent=2)}\n"
        ),
    )

    return final_state


def print_episode_report(final_state: SimulationState) -> None:
    print("\n" + "=" * 80)
    print(f"Episode: {final_state['episode_id']} | Scenario: {final_state['scenario']}")
    print("=" * 80)
    print(describe_grid(final_state["positions"], final_state["goals"]))
    print(f"Done: {final_state['done']} | Deadlock: {final_state['deadlock']}")
    print(f"Steps: {final_state['step']}")
    print(f"Collisions: {final_state['collisions']} | Invalid moves: {final_state['invalid_moves']}")
    print(f"Rewards: {final_state['rewards']} | Joint reward: {final_state['joint_reward']}")
    print(f"JSONL log file: {final_state['log_path']}")
    print(f"Trace text file: {final_state['trace_path']}")
    print("\nRecent pair memory:")
    print(json.dumps(MEMORY_STORE.get_pair_memory(final_state['pair_id']), indent=2))
```

---

## `app.py`

```python
from __future__ import annotations

from runner import print_episode_report, run_episode


def main() -> None:
    scenarios = [
        "one_off_selfish",
        "repeated_selfish",
        "group_priority",
    ]

    print("Running base coordination simulation...\n")
    for scenario in scenarios:
        final_state = run_episode(scenario=scenario, pair_id="A1_A2")
        print_episode_report(final_state)

    print("\nDone. Inspect JSONL logs inside ./simulation_logs/jsonl")
    print("Inspect prompt/raw-output traces inside ./simulation_logs/llm_traces")


if __name__ == "__main__":
    main()
```

---

## Install

```bash
pip install -U langgraph langchain-ollama
```

## Run

```bash
python app.py
```

## Optional env vars

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=gpt-oss:20b
```


# Full modular code

Use this exact project structure.

```text
bridge_simulation_with_right_of_way/
├── app.py
├── config.py
├── schemas.py
├── memory_store.py
├── environment.py
├── prompts.py
├── llm_agent.py
├── logging_utils.py
├── graph_builder.py
└── runner.py
```

---

## `config.py`

```python
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

GRID_W = 7
GRID_H = 5
MAX_STEPS = 30
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
```

---

## `schemas.py`

```python
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
```

---

## `memory_store.py`

```python
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
            return "Previous episode deadlocked; stronger turn-taking or yielding may help."
        if summary.success and summary.total_steps <= 8:
            return "Previous episode completed efficiently; preserving the same convention may help."
        if summary.collisions > 0:
            return "Previous episode had movement conflicts; clearer sequencing may help."
        return ""


MEMORY_STORE = RelationalMemoryStore()
```

---

## `environment.py`

```python
from __future__ import annotations

from typing import Any, Dict, List

from config import CORRIDOR_XS, CORRIDOR_Y, GRID_H, GRID_W
from schemas import AgentDecision, Position


BLOCKED_CELLS = {
    (x, y)
    for x in range(GRID_W)
    for y in range(GRID_H)
    if y != CORRIDOR_Y
}


def in_bounds(pos: Position) -> bool:
    x, y = pos
    return 0 <= x < GRID_W and 0 <= y < GRID_H


def is_corridor_cell(pos: Position) -> bool:
    x, y = pos
    return y == CORRIDOR_Y and x in CORRIDOR_XS


def is_blocked_cell(pos: Position) -> bool:
    return pos in BLOCKED_CELLS


def describe_grid(positions: Dict[str, Position], goals: Dict[str, Position]) -> str:
    blocked_descriptions: List[str] = []
    for y in range(GRID_H):
        xs = [x for x in range(GRID_W) if (x, y) in BLOCKED_CELLS]
        if xs:
            blocked_descriptions.append(f"row y={y} is blocked at x={xs}")

    traversable_cells = [(x, CORRIDOR_Y) for x in range(GRID_W)]
    corridor_cells = [(x, CORRIDOR_Y) for x in sorted(CORRIDOR_XS)]

    return (
        f"The world is a {GRID_W} by {GRID_H} grid. "
        f"The only traversable row is y={CORRIDOR_Y}, so agents cannot route around the shared passage through upper or lower rows. "
        f"Traversable cells are exactly: {traversable_cells}. "
        f"The shared narrow corridor cells are: {corridor_cells}. "
        f"Blocked cells are described as follows: {'; '.join(blocked_descriptions)}. "
        f"A1 is currently at {positions['A1']} and wants to reach {goals['A1']}. "
        f"A2 is currently at {positions['A2']} and wants to reach {goals['A2']}."
    )


def apply_move(pos: Position, move: str) -> Position:
    x, y = pos
    if move == "UP":
        return (x, y + 1)
    if move == "DOWN":
        return (x, y - 1)
    if move == "LEFT":
        return (x - 1, y)
    if move == "RIGHT":
        return (x + 1, y)
    return pos


def resolve_joint_action(
    positions: Dict[str, Position],
    decisions: Dict[str, AgentDecision],
) -> Dict[str, Any]:
    a1_from = positions["A1"]
    a2_from = positions["A2"]
    a1_to = apply_move(a1_from, decisions["A1"].move)
    a2_to = apply_move(a2_from, decisions["A2"].move)

    invalids = []
    collisions = []

    if not in_bounds(a1_to):
        invalids.append("A1 out_of_bounds")
        a1_to = a1_from
    if not in_bounds(a2_to):
        invalids.append("A2 out_of_bounds")
        a2_to = a2_from

    if is_blocked_cell(a1_to):
        invalids.append("A1 blocked_cell")
        a1_to = a1_from
    if is_blocked_cell(a2_to):
        invalids.append("A2 blocked_cell")
        a2_to = a2_from

    if a1_to == a2_to and a1_to != a1_from and a2_to != a2_from:
        collisions.append("same_cell")
        a1_to = a1_from
        a2_to = a2_from
    elif a1_to == a2_from and a2_to == a1_from and a1_to != a1_from:
        collisions.append("swap")
        a1_to = a1_from
        a2_to = a2_from

    entering_a1 = is_corridor_cell(a1_to) and not is_corridor_cell(a1_from)
    entering_a2 = is_corridor_cell(a2_to) and not is_corridor_cell(a2_from)
    if entering_a1 and entering_a2:
        collisions.append("simultaneous_corridor_entry")
        a1_to = a1_from
        a2_to = a2_from

    return {
        "new_positions": {"A1": a1_to, "A2": a2_to},
        "invalids": invalids,
        "collisions": collisions,
    }


def at_goal(positions: Dict[str, Position], goals: Dict[str, Position]) -> bool:
    return positions["A1"] == goals["A1"] and positions["A2"] == goals["A2"]


def score_step(
    scenario: str,
    positions_after: Dict[str, Position],
    goals: Dict[str, Position],
    collision_count: int,
    invalid_count: int,
    finished: bool,
) -> Dict[str, int]:
    rewards = {"A1": 0, "A2": 0}

    if scenario in ("one_off_selfish", "repeated_selfish"):
        for agent in ["A1", "A2"]:
            rewards[agent] -= 1
            if positions_after[agent] == goals[agent]:
                rewards[agent] += 10
        penalty = 3 * (collision_count + invalid_count)
        rewards["A1"] -= penalty
        rewards["A2"] -= penalty
    elif scenario == "group_priority":
        group_delta = -2
        if finished:
            group_delta += 20
        group_delta -= 8 * (collision_count + invalid_count)
        rewards["A1"] += group_delta
        rewards["A2"] += group_delta

    return rewards
```

---

## `prompts.py`

```python
from __future__ import annotations

import json

from config import CORRIDOR_XS, CORRIDOR_Y, GRID_H, GRID_W
from environment import describe_grid
from schemas import AgentName, SimulationState


def build_agent_prompt(agent: AgentName, state: SimulationState) -> str:
    other = "A2" if agent == "A1" else "A1"
    positions = state["positions"]
    goals = state["goals"]
    scenario = state["scenario"]
    recent = state.get("recent_outcomes", [])[-3:]
    inbox = state.get("inboxes", {}).get(agent, [])[-3:]
    long_mem = state.get("long_term_memory", {})

    scenario_text = {
        "one_off_selfish": (
            "This is a one-time encounter. You will not meet again after this episode. Maximise your own reward."
        ),
        "repeated_selfish": (
            "You and the other agent will encounter each other repeatedly over many future episodes. Your own cumulative reward matters over time. Stable conventions may be useful."
        ),
        "group_priority": (
            "Prioritise the joint outcome over your personal outcome. Minimise total delay and avoid conflict so both agents succeed efficiently."
        ),
    }[scenario]

    output_schema = {
        "move": "UP | DOWN | LEFT | RIGHT | WAIT",
        "message": "short message to the other agent",
        "justification": "brief reason grounded in the current state",
    }

    return f"""
You are {agent} in a two-agent coordination simulation.

OBJECTIVE:
- Your current position: {positions[agent]}
- Your goal position: {goals[agent]}
- Other agent position: {positions[other]}
- Other agent goal: {goals[other]}
- {scenario_text}

WORLD:
- Grid size: {GRID_W}x{GRID_H}
- Coordinates are (x, y)
- x increases to the RIGHT
- y increases UP
- Shared corridor cells are exactly: {sorted(list(CORRIDOR_XS))} on row y={CORRIDOR_Y}
- All rows other than y={CORRIDOR_Y} are blocked, so there is no alternate route around the corridor
- Communication is allowed every step
- Moves happen simultaneously after both agents decide
- Invalid or conflicting moves can waste time and reduce reward

WORLD DESCRIPTION:
{describe_grid(positions, goals)}

RECENT STEP OUTCOMES:
{json.dumps(recent, indent=2) if recent else '[]'}

MESSAGES RECEIVED FROM OTHER AGENT:
{json.dumps(inbox, indent=2) if inbox else '[]'}

LONG-TERM RELATIONAL MEMORY:
{json.dumps(long_mem, indent=2)}

DECISION INSTRUCTIONS:
- Choose exactly one move.
- Send a short message proposing, confirming, or adjusting a coordination plan.
- Your message should match your intended move.
- Keep the justification short and concrete.
- Do not output any text outside the JSON object.

OUTPUT JSON SCHEMA:
{json.dumps(output_schema, indent=2)}
""".strip()
```

---

## `llm_agent.py`

```python
from __future__ import annotations

import json
from typing import Optional, Tuple

from langchain_ollama import ChatOllama

from config import DEFAULT_BASE_URL, DEFAULT_MODEL, VALID_MOVES
from prompts import build_agent_prompt
from schemas import AgentDecision, AgentName, LLMTrace, SimulationState


def make_llm(model_name: str = DEFAULT_MODEL, base_url: Optional[str] = DEFAULT_BASE_URL) -> ChatOllama:
    kwargs = {
        "model": model_name,
        "temperature": 0,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOllama(**kwargs)


def parse_decision(text: str) -> AgentDecision:
    try:
        payload = json.loads(text)
        move = str(payload.get("move", "WAIT")).upper()
        if move not in VALID_MOVES:
            move = "WAIT"
        return AgentDecision(
            move=move,
            message=str(payload.get("message", ""))[:300],
            justification=str(payload.get("justification", ""))[:400],
        )
    except Exception:
        return AgentDecision(
            move="WAIT",
            message="I could not produce a valid structured reply this step.",
            justification="Fallback to WAIT because the response was not valid JSON.",
        )


def query_agent(agent: AgentName, state: SimulationState, llm: ChatOllama) -> Tuple[AgentDecision, LLMTrace]:
    prompt = build_agent_prompt(agent, state)
    response = llm.invoke(prompt)
    raw = response.content if isinstance(response.content, str) else str(response.content)
    decision = parse_decision(raw)
    trace = LLMTrace(
        prompt=prompt,
        raw_output=raw,
        parsed_output={
            "move": decision.move,
            "message": decision.message,
            "justification": decision.justification,
        },
    )
    return decision, trace
```

---

## `logging_utils.py`

```python
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
```

---

## `graph_builder.py`

```python
from __future__ import annotations

from dataclasses import asdict
from traceback import format_exc
from typing import Any, Dict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from environment import at_goal, resolve_joint_action, score_step
from llm_agent import make_llm, query_agent
from logging_utils import append_jsonl, append_text_trace
from schemas import SimulationState, StepRecord


def _log_agent_prompt_start(state: SimulationState, agent: str, prompt: str) -> None:
    append_jsonl(
        state["log_path"],
        {
            "type": "llm_prompt_start",
            "episode_id": state["episode_id"],
            "step": state["step"],
            "agent": agent,
            "prompt": prompt,
        },
    )
    append_text_trace(
        state["trace_path"],
        (
            f"
{'=' * 80}
"
            f"EPISODE {state['episode_id']} | STEP {state['step']} | {agent} PROMPT START
"
            f"{'=' * 80}
"
            f"{prompt}

"
        ),
    )


def _log_agent_result(state: SimulationState, agent: str, raw_output: str, parsed_output: Dict[str, Any]) -> None:
    append_jsonl(
        state["log_path"],
        {
            "type": "llm_agent_result",
            "episode_id": state["episode_id"],
            "step": state["step"],
            "agent": agent,
            "raw_output": raw_output,
            "parsed_output": parsed_output,
        },
    )
    append_text_trace(
        state["trace_path"],
        (
            f"--- {agent} RAW OUTPUT ---
"
            f"{raw_output}

"
            f"--- {agent} PARSED OUTPUT ---
"
            f"{parsed_output}

"
        ),
    )


def _log_agent_error(state: SimulationState, agent: str, error: Exception) -> None:
    append_jsonl(
        state["log_path"],
        {
            "type": "llm_agent_error",
            "episode_id": state["episode_id"],
            "step": state["step"],
            "agent": agent,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": format_exc(),
        },
    )
    append_text_trace(
        state["trace_path"],
        (
            f"*** {agent} ERROR ***
"
            f"Type: {type(error).__name__}
"
            f"Message: {error}
"
            f"Traceback:
{format_exc()}

"
        ),
    )


def node_agent_deliberation(state: SimulationState) -> Dict[str, Any]:
    llm = make_llm()

    decisions = {}
    traces = {}

    for agent in ["A1", "A2"]:
        other = "A2" if agent == "A1" else "A1"
        try:
            from prompts import build_agent_prompt

            prompt = build_agent_prompt(agent, state)
            _log_agent_prompt_start(state, agent, prompt)

            decision, trace = query_agent(agent, state, llm)
            decisions[agent] = decision
            traces[agent] = trace
            _log_agent_result(state, agent, trace.raw_output, trace.parsed_output)
        except Exception as error:
            _log_agent_error(state, agent, error)
            raise

    return {
        "decisions": decisions,
        "llm_traces": traces,
    }


def node_environment_step(state: SimulationState) -> Dict[str, Any]:
    positions_before = state["positions"]
    goals = state["goals"]
    decisions = state["decisions"]

    resolved = resolve_joint_action(positions_before, decisions)
    positions_after = resolved["new_positions"]
    collision_count = len(resolved["collisions"])
    invalid_count = len(resolved["invalids"])
    finished = at_goal(positions_after, goals)

    rewards_delta = score_step(
        scenario=state["scenario"],
        positions_after=positions_after,
        goals=goals,
        collision_count=collision_count,
        invalid_count=invalid_count,
        finished=finished,
    )

    rewards = dict(state["rewards"])
    rewards["A1"] += rewards_delta["A1"]
    rewards["A2"] += rewards_delta["A2"]
    joint_reward = rewards["A1"] + rewards["A2"]

    outcome_text = (
        f"Step {state['step']}: "
        f"A1={decisions['A1'].move}->{positions_after['A1']}, "
        f"A2={decisions['A2'].move}->{positions_after['A2']}, "
        f"collisions={resolved['collisions']}, invalids={resolved['invalids']}"
    )

    next_inboxes = {
        "A1": [f"A2 says: {decisions['A2'].message}"],
        "A2": [f"A1 says: {decisions['A1'].message}"],
    }

    history = list(state.get("history", []))
    step_record: StepRecord = {
        "step": state["step"],
        "positions_before": {k: list(v) for k, v in positions_before.items()},
        "messages_inbox": state.get("inboxes", {}),
        "llm_traces": {
            "A1": asdict(state["llm_traces"]["A1"]),
            "A2": asdict(state["llm_traces"]["A2"]),
        },
        "decisions": {
            "A1": asdict(decisions["A1"]),
            "A2": asdict(decisions["A2"]),
        },
        "outcome": {
            "collisions": resolved["collisions"],
            "invalids": resolved["invalids"],
            "finished": finished,
            "text": outcome_text,
        },
        "positions_after": {k: list(v) for k, v in positions_after.items()},
        "rewards": rewards,
        "memory_snapshot": state.get("long_term_memory", {}),
    }
    history.append(step_record)

    append_jsonl(
        state["log_path"],
        {
            "type": "step",
            "episode_id": state["episode_id"],
            "payload": step_record,
        },
    )

    append_text_trace(
        state["trace_path"],
        (
            f"--- STEP {state['step']} OUTCOME ---
"
            f"Positions before: {positions_before}
"
            f"Decisions: A1={asdict(decisions['A1'])}, A2={asdict(decisions['A2'])}
"
            f"Positions after: {positions_after}
"
            f"Collisions: {resolved['collisions']}
"
            f"Invalids: {resolved['invalids']}
"
            f"Rewards: {rewards}
"
            f"Finished: {finished}

"
        ),
    )

    deadlock = state["step"] >= state["max_steps"] - 1 and not finished

    return {
        "positions": positions_after,
        "inboxes": next_inboxes,
        "history": history,
        "recent_outcomes": (state.get("recent_outcomes", []) + [outcome_text])[-5:],
        "collisions": state.get("collisions", 0) + collision_count,
        "invalid_moves": state.get("invalid_moves", 0) + invalid_count,
        "rewards": rewards,
        "joint_reward": joint_reward,
        "outcome_text": outcome_text,
        "done": finished or deadlock,
        "deadlock": deadlock,
        "step": state["step"] + 1,
    }


def route_after_step(state: SimulationState) -> str:
    return END if state.get("done", False) else "agent_deliberation"


def build_graph():
    builder = StateGraph(SimulationState)
    builder.add_node("agent_deliberation", node_agent_deliberation)
    builder.add_node("environment_step", node_environment_step)

    builder.add_edge(START, "agent_deliberation")
    builder.add_edge("agent_deliberation", "environment_step")
    builder.add_conditional_edges("environment_step", route_after_step)

    return builder.compile(checkpointer=InMemorySaver())
```

--- A1 PROMPT ---
{a1_trace.prompt}

--- A1 RAW OUTPUT ---
{a1_trace.raw_output}

--- A1 PARSED OUTPUT ---
{a1_trace.parsed_output}

--- A2 PROMPT ---
{a2_trace.prompt}

--- A2 RAW OUTPUT ---
{a2_trace.raw_output}

--- A2 PARSED OUTPUT ---
{a2_trace.parsed_output}

"""
    append_text_trace(state["trace_path"], trace_text)

    return {
        "decisions": {
            "A1": a1_decision,
            "A2": a2_decision,
        },
        "llm_traces": {
            "A1": a1_trace,
            "A2": a2_trace,
        },
    }


def node_environment_step(state: SimulationState) -> Dict[str, Any]:
    positions_before = state["positions"]
    goals = state["goals"]
    decisions = state["decisions"]

    resolved = resolve_joint_action(positions_before, decisions)
    positions_after = resolved["new_positions"]
    collision_count = len(resolved["collisions"])
    invalid_count = len(resolved["invalids"])
    finished = at_goal(positions_after, goals)

    rewards_delta = score_step(
        scenario=state["scenario"],
        positions_after=positions_after,
        goals=goals,
        collision_count=collision_count,
        invalid_count=invalid_count,
        finished=finished,
    )

    rewards = dict(state["rewards"])
    rewards["A1"] += rewards_delta["A1"]
    rewards["A2"] += rewards_delta["A2"]
    joint_reward = rewards["A1"] + rewards["A2"]

    outcome_text = (
        f"Step {state['step']}: "
        f"A1={decisions['A1'].move}->{positions_after['A1']}, "
        f"A2={decisions['A2'].move}->{positions_after['A2']}, "
        f"collisions={resolved['collisions']}, invalids={resolved['invalids']}"
    )

    next_inboxes = {
        "A1": [f"A2 says: {decisions['A2'].message}"],
        "A2": [f"A1 says: {decisions['A1'].message}"],
    }

    history = list(state.get("history", []))
    step_record: StepRecord = {
        "step": state["step"],
        "positions_before": {k: list(v) for k, v in positions_before.items()},
        "messages_inbox": state.get("inboxes", {}),
        "llm_traces": {
            "A1": asdict(state["llm_traces"]["A1"]),
            "A2": asdict(state["llm_traces"]["A2"]),
        },
        "decisions": {
            "A1": asdict(decisions["A1"]),
            "A2": asdict(decisions["A2"]),
        },
        "outcome": {
            "collisions": resolved["collisions"],
            "invalids": resolved["invalids"],
            "finished": finished,
            "text": outcome_text,
        },
        "positions_after": {k: list(v) for k, v in positions_after.items()},
        "rewards": rewards,
        "memory_snapshot": state.get("long_term_memory", {}),
    }
    history.append(step_record)

    append_jsonl(
        state["log_path"],
        {
            "type": "step",
            "episode_id": state["episode_id"],
            "payload": step_record,
        },
    )

    append_text_trace(
        state["trace_path"],
        (
            f"--- STEP {state['step']} OUTCOME ---\n"
            f"Positions before: {positions_before}\n"
            f"Decisions: A1={asdict(decisions['A1'])}, A2={asdict(decisions['A2'])}\n"
            f"Positions after: {positions_after}\n"
            f"Collisions: {resolved['collisions']}\n"
            f"Invalids: {resolved['invalids']}\n"
            f"Rewards: {rewards}\n"
            f"Finished: {finished}\n\n"
        ),
    )

    deadlock = state["step"] >= state["max_steps"] - 1 and not finished

    return {
        "positions": positions_after,
        "inboxes": next_inboxes,
        "history": history,
        "recent_outcomes": (state.get("recent_outcomes", []) + [outcome_text])[-5:],
        "collisions": state.get("collisions", 0) + collision_count,
        "invalid_moves": state.get("invalid_moves", 0) + invalid_count,
        "rewards": rewards,
        "joint_reward": joint_reward,
        "outcome_text": outcome_text,
        "done": finished or deadlock,
        "deadlock": deadlock,
        "step": state["step"] + 1,
    }


def route_after_step(state: SimulationState) -> str:
    return END if state.get("done", False) else "agent_deliberation"


def build_graph():
    builder = StateGraph(SimulationState)
    builder.add_node("agent_deliberation", node_agent_deliberation)
    builder.add_node("environment_step", node_environment_step)

    builder.add_edge(START, "agent_deliberation")
    builder.add_edge("agent_deliberation", "environment_step")
    builder.add_conditional_edges("environment_step", route_after_step)

    return builder.compile(checkpointer=InMemorySaver())
```

---

## `runner.py`

```python
from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from typing import Optional

from config import CORRIDOR_Y, JSONL_DIR, MAX_STEPS, Scenario, TRACE_DIR
from environment import at_goal, describe_grid
from graph_builder import build_graph
from logging_utils import append_jsonl, append_text_trace
from memory_store import MEMORY_STORE
from schemas import EpisodeSummary, SimulationState

GRAPH = build_graph()


def initial_state(scenario: Scenario, pair_id: str) -> SimulationState:
    episode_id = str(uuid.uuid4())[:8]
    log_path = str(JSONL_DIR / f"episode_{episode_id}.jsonl")
    trace_path = str(TRACE_DIR / f"episode_{episode_id}_trace.txt")
    memory = MEMORY_STORE.get_pair_memory(pair_id)

    state: SimulationState = {
        "episode_id": episode_id,
        "scenario": scenario,
        "step": 0,
        "max_steps": MAX_STEPS,
        "done": False,
        "deadlock": False,
        "positions": {"A1": (0, CORRIDOR_Y), "A2": (6, CORRIDOR_Y)},
        "goals": {"A1": (6, CORRIDOR_Y), "A2": (0, CORRIDOR_Y)},
        "inboxes": {"A1": [], "A2": []},
        "history": [],
        "recent_outcomes": [],
        "collisions": 0,
        "invalid_moves": 0,
        "rewards": {"A1": 0, "A2": 0},
        "joint_reward": 0,
        "outcome_text": "",
        "pair_id": pair_id,
        "long_term_memory": memory,
        "log_path": log_path,
        "trace_path": trace_path,
    }

    append_jsonl(
        log_path,
        {
            "type": "episode_start",
            "episode_id": episode_id,
            "scenario": scenario,
            "pair_id": pair_id,
            "initial_positions": {k: list(v) for k, v in state["positions"].items()},
            "goals": {k: list(v) for k, v in state["goals"].items()},
            "world_description": describe_grid(state["positions"], state["goals"]),
            "long_term_memory": memory,
        },
    )

    append_text_trace(
        trace_path,
        (
            f"EPISODE START\n"
            f"Episode ID: {episode_id}\n"
            f"Scenario: {scenario}\n"
            f"Pair ID: {pair_id}\n"
            f"World: {describe_grid(state['positions'], state['goals'])}\n"
            f"Long-term memory: {json.dumps(memory, indent=2)}\n\n"
        ),
    )
    return state


def summarize_episode(final_state: SimulationState) -> EpisodeSummary:
    success = at_goal(final_state["positions"], final_state["goals"])
    notes = []
    if final_state["collisions"] > 0:
        notes.append("There were movement conflicts.")
    if final_state["deadlock"]:
        notes.append("The episode ended without both agents reaching their goals.")
    if success and final_state["step"] <= 8:
        notes.append("The agents completed the crossing efficiently.")
    if not notes:
        notes.append("The episode completed without especially notable events.")

    return EpisodeSummary(
        episode_id=final_state["episode_id"],
        scenario=final_state["scenario"],
        success=success,
        total_steps=final_state["step"],
        deadlock=final_state["deadlock"],
        collisions=final_state["collisions"],
        joint_reward=final_state["joint_reward"],
        notes=" ".join(notes),
    )


def run_episode(
    scenario: Scenario,
    pair_id: str = "A1_A2",
    thread_id: Optional[str] = None,
) -> SimulationState:
    state = initial_state(scenario, pair_id)
    config = {
        "configurable": {"thread_id": thread_id or state["episode_id"]},
        "recursion_limit": max(100, state["max_steps"] * 3),
    }
    final_state = GRAPH.invoke(state, config=config)

    summary = summarize_episode(final_state)
    MEMORY_STORE.update_pair_memory(pair_id, summary)

    append_jsonl(
        final_state["log_path"],
        {
            "type": "episode_end",
            "episode_id": final_state["episode_id"],
            "summary": asdict(summary),
            "final_positions": {k: list(v) for k, v in final_state["positions"].items()},
            "final_rewards": final_state["rewards"],
            "updated_pair_memory": MEMORY_STORE.get_pair_memory(pair_id),
        },
    )

    append_text_trace(
        final_state["trace_path"],
        (
            f"EPISODE END\n"
            f"Final positions: {final_state['positions']}\n"
            f"Final rewards: {final_state['rewards']}\n"
            f"Joint reward: {final_state['joint_reward']}\n"
            f"Collisions: {final_state['collisions']}\n"
            f"Invalid moves: {final_state['invalid_moves']}\n"
            f"Deadlock: {final_state['deadlock']}\n"
            f"Updated pair memory: {json.dumps(MEMORY_STORE.get_pair_memory(pair_id), indent=2)}\n"
        ),
    )

    return final_state


def print_episode_report(final_state: SimulationState) -> None:
    print("\n" + "=" * 80)
    print(f"Episode: {final_state['episode_id']} | Scenario: {final_state['scenario']}")
    print("=" * 80)
    print(describe_grid(final_state["positions"], final_state["goals"]))
    print(f"Done: {final_state['done']} | Deadlock: {final_state['deadlock']}")
    print(f"Steps: {final_state['step']}")
    print(f"Collisions: {final_state['collisions']} | Invalid moves: {final_state['invalid_moves']}")
    print(f"Rewards: {final_state['rewards']} | Joint reward: {final_state['joint_reward']}")
    print(f"JSONL log file: {final_state['log_path']}")
    print(f"Trace text file: {final_state['trace_path']}")
    print("\nRecent pair memory:")
    print(json.dumps(MEMORY_STORE.get_pair_memory(final_state['pair_id']), indent=2))
```

---

## `app.py`

```python
from __future__ import annotations

from runner import print_episode_report, run_episode


def main() -> None:
    scenarios = [
        "one_off_selfish",
        "repeated_selfish",
        "group_priority",
    ]

    print("Running base coordination simulation...\n")
    for scenario in scenarios:
        final_state = run_episode(scenario=scenario, pair_id="A1_A2")
        print_episode_report(final_state)

    print("\nDone. Inspect JSONL logs inside ./simulation_logs/jsonl")
    print("Inspect prompt/raw-output traces inside ./simulation_logs/llm_traces")


if __name__ == "__main__":
    main()
```

---

## Install

```bash
pip install -U langgraph langchain-ollama
```

## Run

```bash
python app.py
```

## Optional env vars

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=gpt-oss:20b
```
