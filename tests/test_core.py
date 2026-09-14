from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dgdrl import SequentialTerminalEnv, masked_argmax, masked_epsilon_greedy


def test_terminal_reward_is_deferred_and_called_once():
    calls = []

    def evaluator(design):
        calls.append(np.asarray(design).copy())
        return float(np.sum(design))

    env = SequentialTerminalEnv(([0, 1], [0, 1, 2], [0, 1]), evaluator)
    observation, _ = env.reset(seed=4)
    actions = [1, 2, 1]
    for index, action in enumerate(actions):
        observation, reward, terminated, truncated, info = env.step(action)
        assert truncated is False
        assert terminated is (index == len(actions) - 1)
        assert reward == (4.0 if terminated else 0.0)
        assert info["terminal_evaluation"] is terminated
        if not terminated:
            assert np.all(observation["action_mask"] >= 0)
    assert len(calls) == 1
    assert calls[0].tolist() == actions
    env.close()


def test_dynamic_action_mask_restricts_current_step():
    def restrict(design, step):
        if step == 0:
            return [0]
        return [1]

    env = SequentialTerminalEnv(
        ([0, 1], [0, 1]),
        lambda design: 0.0,
        dynamic_feasibility=restrict,
    )
    observation, _ = env.reset()
    assert observation["action_mask"].tolist() == [1, 0]
    observation, _, _, _, _ = env.step(0)
    assert observation["action_mask"].tolist() == [0, 1]
    env.close()


def test_masked_selection_never_uses_invalid_action():
    values = np.asarray([100.0, 1.0, 2.0])
    mask = np.asarray([0, 1, 1])
    assert masked_argmax(values, mask) == 2
    rng = np.random.default_rng(7)
    choices = {masked_epsilon_greedy(values, mask, 1.0, rng) for _ in range(40)}
    assert choices.issubset({1, 2})
