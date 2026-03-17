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


def initial_state(scenario: Scenario, episode_id: int, pair_id: str) -> SimulationState:
    episode_id = f"{episode_id}"
    log_path = str(JSONL_DIR / f"episode_{scenario}_{episode_id}.jsonl")
    trace_path = str(TRACE_DIR / f"episode_{scenario}_{episode_id}_trace.txt")
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
            # "world_description": describe_grid(state["positions"], state["goals"]),
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
    if success:
        notes.append("The agents successfully reached their goals.")
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
    episode_id: int,
    pair_id: str = "A1_A2",
    thread_id: Optional[str] = None,
) -> SimulationState:
    state = initial_state(scenario, episode_id, pair_id)
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