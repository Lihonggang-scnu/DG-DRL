from __future__ import annotations

import numpy as np


def make_action_mask(valid_actions, action_count: int) -> np.ndarray:
    """Build a binary action mask from a state-dependent feasible set."""
    count = int(action_count)
    mask = np.zeros(count, dtype=np.int8)
    for action in valid_actions:
        action = int(action)
        if action < 0 or action >= count:
            raise ValueError(f"action {action} is outside 0..{count - 1}")
        mask[action] = 1
    if not np.any(mask):
        raise ValueError("a feasible action mask cannot be empty")
    return mask


def masked_argmax(values, mask) -> int:
    """Select the best valid action without allowing invalid logits through."""
    scores = np.asarray(values, dtype=float).ravel()
    valid = np.asarray(mask, dtype=bool).ravel()
    if scores.shape != valid.shape:
        raise ValueError("values and mask must have the same shape")
    if not np.any(valid):
        raise ValueError("a feasible action mask cannot be empty")
    return int(np.argmax(np.where(valid, scores, -np.inf)))


def masked_epsilon_greedy(
    values,
    mask,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """Sample uniformly among feasible actions or choose masked_argmax."""
    valid = np.flatnonzero(np.asarray(mask, dtype=bool))
    if valid.size == 0:
        raise ValueError("a feasible action mask cannot be empty")
    if float(rng.random()) < float(epsilon):
        return int(rng.choice(valid))
    return masked_argmax(values, mask)
