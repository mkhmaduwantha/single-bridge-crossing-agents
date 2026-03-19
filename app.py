from __future__ import annotations

from runner import print_episode_report, run_episode
from config import EPISODES_PER_SCENARIO


def main() -> None:
    scenarios = [
        "repeated_selfish",
        "group_priority",
        "one_off_selfish",
    ]

    print("Running base coordination simulation...\n")
    for scenario in scenarios:
        for episode in range(50):
            run_episode(scenario=scenario, episode_id=f"{scenario}_{episode}", pair_id="A1_A2")
        # final_state = run_episode(scenario=scenario, pair_id="A1_A2")
        # print_episode_report(final_state)

    print("\nDone. Inspect JSONL logs inside ./simulation_logs/jsonl")
    print("Inspect prompt/raw-output traces inside ./simulation_logs/llm_traces")


if __name__ == "__main__":
    main()