from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dgdrl import SequentialTerminalEnv


def terminal_objective(design: np.ndarray) -> float:
    """A toy evaluator standing in for an expensive physical simulation."""
    return float(np.sum(design))


def main() -> None:
    # These toy domains are only for illustrating the interaction sequence.
    domains = ([0, 1], [0, 1, 2], [0, 1])
    env = SequentialTerminalEnv(domains, terminal_objective)
    observation, _ = env.reset(seed=0)
    done = False
    while not done:
        action = int(np.flatnonzero(observation["action_mask"])[0])
        observation, reward, done, _, info = env.step(action)
    print({"terminal_reward": reward, "info": info})


if __name__ == "__main__":
    main()
