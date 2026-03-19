from __future__ import annotations

from typing import Any, Dict, List

from config import CORRIDOR_XS, CORRIDOR_Y, GRID_H, GRID_W
from schemas import AgentDecision, Position


BLOCKED_CELLS = {
    (x, y)
    for x in CORRIDOR_XS
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
    f"- The world is a {GRID_W} by {GRID_H} grid. \n"
    f"- A narrow horizontal corridor exists at row y={CORRIDOR_Y}. \n"
    f"- The corridor cells are {[(x, CORRIDOR_Y) for x in sorted(CORRIDOR_XS)]}. \n"
    f"- Cells directly up and down these corridor cells are blocked, forming vertical walls. \n"
    f"- The blocked wall cells are {sorted(BLOCKED_CELLS)}. \n"
    f"- This prevents agents from bypassing the corridor by moving above or below it."
    # f"A1 is currently at {positions['A1']} and wants to reach {goals['A1']}. "
    # f"A2 is currently at {positions['A2']} and wants to reach {goals['A2']}."
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

    
    if a1_to == a2_from and a2_to == a1_from and a1_to != a1_from:
        collisions.append("swap")
        a1_to = a1_from
        a2_to = a2_from
    elif a1_to == a2_to:
        collisions.append("same_cell")
        a1_to = a1_from
        a2_to = a2_from
    
    a1_in_corridor_after = is_corridor_cell(a1_to)
    a2_in_corridor_after = is_corridor_cell(a2_to)

    if a1_in_corridor_after and a2_in_corridor_after:
        collisions.append("same_corridor")
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