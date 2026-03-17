import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = "simulation_logs/jsonl"          # folder containing episode log files
OUTPUT_DIR = "visuals"      # folder where images will be written

# Optional fixed grid size. If None, infer from data.
FIXED_GRID_WIDTH = None
FIXED_GRID_HEIGHT = None

# Optional blocked cells if your environment has them.
# Example: {(2, 0), (2, 1), (2, 3)}
BLOCKED_CELLS = set()

# Agent styles
AGENT_STYLES = {
    "A1": {"color": "#2563eb", "goal_color": "#93c5fd"},
    "A2": {"color": "#dc2626", "goal_color": "#fca5a5"},
}

# ============================================================
# PARSING
# ============================================================

def load_json_lines(file_path: Path):
    """
    Reads a file containing either:
    - one JSON object per line
    - or a single JSON array/object
    Returns a list of dicts.
    """
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    items = []

    # Try JSONL first
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    jsonl_ok = True
    for line in lines:
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            jsonl_ok = False
            break

    if jsonl_ok:
        return items

    # Fallback: whole file as JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass

    raise ValueError(f"Could not parse JSON content from {file_path}")


def split_episode_records(records):
    """
    Groups records by episode_id.
    """
    episodes = {}
    for rec in records:
        ep_id = rec.get("episode_id")
        if not ep_id:
            ep_id = rec.get("summary", {}).get("episode_id")
        if not ep_id:
            ep_id = "unknown_episode"

        episodes.setdefault(ep_id, []).append(rec)

    return episodes


def extract_episode_data(records):
    """
    Returns:
      {
        "episode_start": {...} or None,
        "steps": [...],
        "episode_end": {...} or None,
      }
    """
    episode_start = None
    episode_end = None
    steps = []

    for rec in records:
        rtype = rec.get("type")
        if rtype == "episode_start":
            episode_start = rec
        elif rtype == "step":
            steps.append(rec)
        elif rtype == "episode_end":
            episode_end = rec

    steps.sort(key=lambda r: r.get("payload", {}).get("step", 0))

    return {
        "episode_start": episode_start,
        "steps": steps,
        "episode_end": episode_end,
    }


# ============================================================
# GRID HELPERS
# ============================================================

def infer_grid_size(episode_start, steps, episode_end):
    """
    Infer grid bounds from all known positions/goals.
    """
    xs = []
    ys = []

    def add_pos(pos):
        if pos is None:
            return
        xs.append(pos[0])
        ys.append(pos[1])

    if episode_start:
        for pos in episode_start.get("initial_positions", {}).values():
            add_pos(pos)
        for pos in episode_start.get("goals", {}).values():
            add_pos(pos)

    for step in steps:
        payload = step.get("payload", {})
        for pos in payload.get("positions_before", {}).values():
            add_pos(pos)
        for pos in payload.get("positions_after", {}).values():
            add_pos(pos)

    if episode_end:
        for pos in episode_end.get("final_positions", {}).values():
            add_pos(pos)

    for x, y in BLOCKED_CELLS:
        add_pos((x, y))

    if not xs or not ys:
        return 7, 5

    width = max(xs) + 1
    height = max(ys) + 1

    if FIXED_GRID_WIDTH is not None:
        width = FIXED_GRID_WIDTH
    if FIXED_GRID_HEIGHT is not None:
        height = FIXED_GRID_HEIGHT

    return width, height


def move_delta(move):
    move = (move or "").upper()
    return {
        "UP": (0, 1),
        "DOWN": (0, -1),
        "LEFT": (-1, 0),
        "RIGHT": (1, 0),
        "WAIT": (0, 0),
        "STAY": (0, 0),
    }.get(move, (0, 0))


def wrap_text(text, width=50):
    if text is None:
        return ""
    text = str(text)
    words = text.split()
    if not words:
        return ""

    lines = []
    current = words[0]
    for w in words[1:]:
        if len(current) + 1 + len(w) <= width:
            current += " " + w
        else:
            lines.append(current)
            current = w
    lines.append(current)
    return "\n".join(lines)


# ============================================================
# DRAWING
# ============================================================

def draw_grid(ax, width, height):
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")

    # Cells
    for x in range(width):
        for y in range(height):
            face = "white"
            if (x, y) in BLOCKED_CELLS:
                face = "#d1d5db"
            rect = Rectangle((x, y), 1, 1, facecolor=face, edgecolor="black", linewidth=1)
            ax.add_patch(rect)

            ax.text(
                x + 0.5,
                y + 0.08,
                f"({x},{y})",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#6b7280",
            )

    ax.set_xticks(range(width + 1))
    ax.set_yticks(range(height + 1))
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def draw_goals(ax, goals):
    for agent, pos in goals.items():
        if pos is None:
            continue
        x, y = pos
        style = AGENT_STYLES.get(agent, {"goal_color": "#d1d5db"})
        goal_rect = Rectangle(
            (x + 0.12, y + 0.12),
            0.76,
            0.76,
            fill=False,
            linestyle="--",
            linewidth=2,
            edgecolor=style["goal_color"],
        )
        ax.add_patch(goal_rect)
        ax.text(
            x + 0.5,
            y + 0.5,
            f"{agent}\nGOAL",
            ha="center",
            va="center",
            fontsize=8,
            color=style["goal_color"],
            weight="bold",
        )


def draw_agents(ax, positions_after):
    drawn = {}

    # handle overlapping agents
    overlap_groups = {}
    for agent, pos in positions_after.items():
        overlap_groups.setdefault(tuple(pos), []).append(agent)

    for pos, agents in overlap_groups.items():
        x, y = pos
        if len(agents) == 1:
            agent = agents[0]
            style = AGENT_STYLES.get(agent, {"color": "#111827"})
            circ = Circle((x + 0.5, y + 0.5), 0.22, facecolor=style["color"], edgecolor="black")
            ax.add_patch(circ)
            ax.text(x + 0.5, y + 0.5, agent, ha="center", va="center", fontsize=9, color="white", weight="bold")
        else:
            offsets = [(-0.16, 0), (0.16, 0), (0, 0.16), (0, -0.16)]
            for idx, agent in enumerate(agents):
                dx, dy = offsets[idx % len(offsets)]
                style = AGENT_STYLES.get(agent, {"color": "#111827"})
                circ = Circle((x + 0.5 + dx, y + 0.5 + dy), 0.18, facecolor=style["color"], edgecolor="black")
                ax.add_patch(circ)
                ax.text(
                    x + 0.5 + dx,
                    y + 0.5 + dy,
                    agent,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    weight="bold",
                )


def draw_move_arrows(ax, positions_before, decisions):
    for agent, pos in positions_before.items():
        if pos is None:
            continue
        x, y = pos
        move = decisions.get(agent, {}).get("move", "WAIT")
        dx, dy = move_delta(move)

        if dx == 0 and dy == 0:
            ax.text(
                x + 0.5,
                y + 0.84,
                "WAIT",
                ha="center",
                va="center",
                fontsize=8,
                weight="bold",
                color="#111827",
            )
            continue

        ax.arrow(
            x + 0.5,
            y + 0.5,
            0.32 * dx,
            0.32 * dy,
            head_width=0.10,
            head_length=0.10,
            length_includes_head=True,
            linewidth=2,
            color=AGENT_STYLES.get(agent, {}).get("color", "#111827"),
        )


def build_text_panel(step_payload):
    step_num = step_payload.get("step")
    decisions = step_payload.get("decisions", {})
    rewards = step_payload.get("rewards", {})
    outcome = step_payload.get("outcome", {})
    collisions = outcome.get("collisions", [])
    invalids = outcome.get("invalids", [])

    lines = []
    lines.append(f"Step {step_num}")
    lines.append(f"Collisions: {collisions if collisions else 'None'}")
    lines.append(f"Invalids: {invalids if invalids else 'None'}")
    lines.append("")

    for agent in sorted(decisions.keys()):
        move = decisions[agent].get("move", "")
        message = decisions[agent].get("message", "")
        justification = decisions[agent].get("justification", "")
        reward = rewards.get(agent, "")

        lines.append(f"{agent}")
        lines.append(f"  Move: {move}")
        lines.append(f"  Message: {message}")
        lines.append("  Justification:")
        for ln in wrap_text(justification, width=62).splitlines():
            lines.append(f"    {ln}")
        lines.append(f"  Current reward: {reward}")
        lines.append("")

    return "\n".join(lines)


def render_step_image(
    output_path: Path,
    width: int,
    height: int,
    goals: dict,
    step_payload: dict,
    episode_id: str,
    scenario: str = "",
):
    positions_before = step_payload.get("positions_before", {})
    positions_after = step_payload.get("positions_after", {})
    decisions = step_payload.get("decisions", {})
    outcome = step_payload.get("outcome", {})

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.3, 1.7])

    ax_grid = fig.add_subplot(gs[0])
    ax_text = fig.add_subplot(gs[1])

    draw_grid(ax_grid, width, height)
    draw_goals(ax_grid, goals)
    draw_move_arrows(ax_grid, positions_before, decisions)
    draw_agents(ax_grid, positions_after)

    step_num = step_payload.get("step", "?")
    title = f"Episode: {episode_id} | Scenario: {scenario or 'unknown'} | Step {step_num}"
    subtitle = outcome.get("text", "")
    ax_grid.set_title(f"{title}\n{subtitle}", fontsize=13, pad=14)

    ax_text.axis("off")
    panel_text = build_text_panel(step_payload)
    ax_text.text(
        0.01,
        0.98,
        panel_text,
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def process_file(file_path: Path, output_root: Path):
    print(f"Processing {file_path} ...")
    records = load_json_lines(file_path)
    episode_groups = split_episode_records(records)

    for episode_id, ep_records in episode_groups.items():
        episode_data = extract_episode_data(ep_records)
        episode_start = episode_data["episode_start"]
        episode_end = episode_data["episode_end"]
        steps = episode_data["steps"]

        if not steps:
            print(f"  Skipping {episode_id}: no step records found.")
            continue

        width, height = infer_grid_size(episode_start, steps, episode_end)

        goals = {}
        scenario = ""
        if episode_start:
            goals = episode_start.get("goals", {})
            scenario = episode_start.get("scenario", "")

        episode_folder = output_root / file_path.stem / episode_id
        episode_folder.mkdir(parents=True, exist_ok=True)

        for step_record in steps:
            payload = step_record.get("payload", {})
            step_idx = payload.get("step", 0)
            out_file = episode_folder / f"step_{step_idx:03d}.png"

            render_step_image(
                output_path=out_file,
                width=width,
                height=height,
                goals=goals,
                step_payload=payload,
                episode_id=episode_id,
                scenario=scenario,
            )

        print(f"  Wrote {len(steps)} images to {episode_folder}")


def main():
    input_root = Path(INPUT_DIR)
    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    files = []
    for ext in ("*.json", "*.jsonl", "*.txt"):
        files.extend(input_root.glob(ext))

    if not files:
        print(f"No input files found in: {input_root.resolve()}")
        return

    for file_path in sorted(files):
        try:
            process_file(file_path, output_root)
        except Exception as e:
            print(f"Failed on {file_path.name}: {e}")


if __name__ == "__main__":
    main()