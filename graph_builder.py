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
    # append_jsonl(
    #     state["log_path"],
    #     {
    #         "type": "llm_prompt_start",
    #         "episode_id": state["episode_id"],
    #         "step": state["step"],
    #         "agent": agent,
    #         "prompt": prompt,
    #     },
    # )
    append_text_trace(
        state["trace_path"],
        (
            f"{'=' * 80}"
            f"EPISODE {state['episode_id']} | STEP {state['step']} | {agent} PROMPT START"
            f"{'=' * 80}"
            f"{prompt}"
        ),
    )


def _log_agent_result(state: SimulationState, agent: str, raw_output: str, parsed_output: Dict[str, Any]) -> None:
    # append_jsonl(
    #     state["log_path"],
    #     {
    #         "type": "llm_agent_result",
    #         "episode_id": state["episode_id"],
    #         "step": state["step"],
    #         "agent": agent,
    #         "raw_output": raw_output,
    #         "parsed_output": parsed_output,
    #     },
    # )
    append_text_trace(
        state["trace_path"],
        (
            f"--- {agent} RAW OUTPUT ---"
            f"{raw_output}"
            f"--- {agent} PARSED OUTPUT ---"
            f"{parsed_output}"
        ),
    )


def _log_agent_error(state: SimulationState, agent: str, error: Exception) -> None:
    # append_jsonl(
    #     state["log_path"],
    #     {
    #         "type": "llm_agent_error",
    #         "episode_id": state["episode_id"],
    #         "step": state["step"],
    #         "agent": agent,
    #         "error_type": type(error).__name__,
    #         "error_message": str(error),
    #         "traceback": format_exc(),
    #     },
    # )
    append_text_trace(
        state["trace_path"],
        (
            f"*** {agent} ERROR ***"
            f"Type: {type(error).__name__}"
            f"Message: {error}"
            f"Traceback:{format_exc()}"
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

    

    next_inboxes = {
        "A1": [f"A2 says: {decisions['A2'].message}"],
        "A2": [f"A1 says: {decisions['A1'].message}"],
    }

    outcome_text = (
        f"Step {state['step']}: "
        f"A1={decisions['A1'].move} to {positions_after['A1']}, "
        f"A2={decisions['A2'].move} to {positions_after['A2']}, "
        f"collisions={resolved['collisions']}, invalids={resolved['invalids']}, "
        f"messages: A1='{decisions['A1'].message}', A2='{decisions['A2'].message}'"
    )

    history = list(state.get("history", []))
    step_record: StepRecord = {
        "step": state["step"],
        "positions_before": {k: list(v) for k, v in positions_before.items()},
        "messages_inbox": state.get("inboxes", {}),
        # "llm_traces": {
        #     "A1": asdict(state["llm_traces"]["A1"]),
        #     "A2": asdict(state["llm_traces"]["A2"]),
        # },
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
            f"--- STEP {state['step']} OUTCOME ---"
            f"Positions before: {positions_before}"
            f"Decisions: A1={asdict(decisions['A1'])}, A2={asdict(decisions['A2'])}"
            f"Positions after: {positions_after}"
            f"Collisions: {resolved['collisions']}"
            f"Invalids: {resolved['invalids']}"
            f"Rewards: {rewards}"
            f"Finished: {finished}"
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