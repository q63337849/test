#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HybridTrajectoryPlanner 冒烟测试。"""

from hybrid_path_planner import HybridConfig, HybridTrajectoryPlanner
from environment import NavigationEnv


def main() -> None:
    env = NavigationEnv()
    cfg = HybridConfig(
        midbo_population=8,
        midbo_iterations=20,
        waypoint_count=6,
        max_episode_steps=80,
    )
    planner = HybridTrajectoryPlanner(env=env, config=cfg, random_state=0)
    result = planner.run_episode(reset=True)

    assert result["steps"] > 0
    assert len(result["trajectory"]) == result["steps"] + 1
    print("Hybrid smoke test passed.")
    print("success=", result["success"], "reason=", result["reason"], "steps=", result["steps"])


if __name__ == "__main__":
    main()
