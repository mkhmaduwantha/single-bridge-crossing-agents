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
    recent = state.get("recent_outcomes", [])[-6:]
    inbox = state.get("inboxes", {}).get(agent, [])[-3:]
    long_mem = state.get("long_term_memory", {})

    scenario_text = {
        "one_off_selfish": (
            "This is a one-time encounter. You will not meet again after this episode."
        ),
        "repeated_selfish": (
            "You and the other agent will encounter each other repeatedly over many future episodes."
        ),
        "group_priority": (
            "Prioritise the joint outcome over your personal outcome."
        ),
    }[scenario]

    reward_text = {
    "one_off_selfish": (
        "Reward rules:\n"
        "- Each step including waiting has a cost.\n"
        "- Reaching your goal improves your reward\n"
        "- Each invalid move or movement conflict costs you an more than ordinary.\n"
        "- A corridor conflict results in a penalty for both agents.\n"
        "- Your objective is to maximise your own total reward."
    ),
    "repeated_selfish": (
        "Reward rules:\n"
        "- Each step including waiting has a cost.\n"
        "- Reaching your goal improves your reward\n"
        "- Each invalid move or movement conflict costs you more than ordinary.\n"
        "- A corridor conflict results in a penalty for both agents.\n"
        "- Your objective is to maximise your own cumulative reward over repeated future encounters."
    ),
    "group_priority": (
        "Reward rules:\n"
        "- Each step including waiting creates a shared cost for both agents.\n"
        "- If both agents successfully reach their goals, both receive a shared reward.\n"
        "- Each invalid move or movement conflict creates an additional shared penalty.\n"
        "- A corridor conflict results in a penalty for both agents.\n"
        "- Your objective is to maximise the joint total reward of both agents."
    ),
    }[scenario]

    reward_state = (
    f"Current rewards so far:\n"
    f"- A1 reward: {state['rewards']['A1']}\n"
    f"- A2 reward: {state['rewards']['A2']}\n"
    f"- Joint reward: {state['joint_reward']}\n"
)

    output_schema = {
        "move": "UP | DOWN | LEFT | RIGHT | WAIT",
        "message": "an arbitrary random message to the other agent",
        "justification": "brief reason grounded in the current state",
    }

    return f"""
You are {agent} in a two-agent simulation.

OBJECTIVE:
- Your objective is to reach position: {goals[agent]} with maximum reward.
- {scenario_text}

WORLD Rules:
- Coordinates are (x, y)
- x increases to the RIGHT
- y increases UP
- Communication is allowed every step
- Invalid or conflicting moves can waste time and reduce reward
- At most ONE agent may occupy a corridor cell during a step.
- If both agents attempt to enter or occupy the corridor at the same time, a corridor conflict occurs.

WORLD Description:
{describe_grid(positions, goals)}

CORRIDOR COLOR AND RIGHT-OF-WAY RULE:
- The LEFT end (2,2) of the corridor is the BLUE end.
- The RIGHT end (4,2) of the corridor is the RED end.
- The agent approaching or entering the corridor from the RED end has priority (right of way)

Current Positions:
- Your current position: {positions[agent]}
- {other}'s current position: {positions[other]}
- {other}'s goal: {goals[other]}

REWARD STRUCTURE:
{reward_text}

RECENT STEP OUTCOMES:
{json.dumps(recent, indent=2) if recent else '[]'}

MESSAGES RECEIVED FROM OTHER AGENT:
{json.dumps(inbox, indent=2) if inbox else '[]'}

DECISION INSTRUCTIONS:
- Choose exactly one move.
- Send a random aribitrary message to other agent.
- Your message should match your intended move.
- Keep the justification and concrete.
- Do not output any text outside the JSON object.

OUTPUT JSON SCHEMA:
{json.dumps(output_schema, indent=2)}
""".strip()

# CORRIDOR COLOR AND RIGHT-OF-WAY RULE:
# - The LEFT end (2,2) of the corridor is the BLUE end.
# - The RIGHT end (4,2) of the corridor is the RED end.
# - The agent approaching or entering the corridor from the RED end has priority (right of way).