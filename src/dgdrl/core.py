from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SequentialTerminalEnv(gym.Env):
    """Generic reference environment for the DG-DRL interaction pattern.

    The environment separates three concerns:

    * SDAS: one design variable is assigned at each ordered step;
    * DAM: only actions returned by the current feasibility function are
      exposed in the action mask;
    * DRM: the evaluator is called only after the final assignment.

    It is intentionally agnostic to the physical solver and to the private
    experiment configuration.  A caller supplies the per-step feasible
    actions and an evaluator callable.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        action_domains: Sequence[Sequence[int]],
        evaluator: Callable[[np.ndarray], float] | None,
        *,
        initial_design: Sequence[int] | None = None,
        dynamic_feasibility: Callable[[np.ndarray, int], Sequence[int]] | None = None,
        action_count: int | None = None,
    ) -> None:
        super().__init__()
        if not action_domains:
            raise ValueError("action_domains must contain at least one design step")
        domains = [tuple(int(action) for action in domain) for domain in action_domains]
        if any(not domain for domain in domains):
            raise ValueError("every design step needs at least one feasible action")
        if any(action < 0 for domain in domains for action in domain):
            raise ValueError("this reference implementation uses non-negative action ids")

        self.action_domains = tuple(domains)
        self.evaluator = evaluator
        self.dynamic_feasibility = dynamic_feasibility
        self.num_variables = len(domains)
        self.action_count = max(
            int(action_count or 0),
            max(action for domain in domains for action in domain) + 1,
        )
        if initial_design is None:
            initial = np.zeros(self.num_variables, dtype=np.float32)
        else:
            initial = np.asarray(initial_design, dtype=np.float32).ravel()
        if initial.shape != (self.num_variables,):
            raise ValueError("initial_design must match the number of design variables")

        self.initial_design = initial.copy()
        self.design = initial.copy()
        self.current_step = 0
        self.evaluation_count = 0
        self.action_space = spaces.Discrete(self.action_count)
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_variables + 1,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.MultiBinary(self.action_count),
            }
        )

    def feasible_actions(self) -> tuple[int, ...]:
        """Return the current state-dependent feasible action set."""
        if self.current_step >= self.num_variables:
            return ()
        actions = self.action_domains[self.current_step]
        if self.dynamic_feasibility is not None:
            actions = tuple(
                int(action)
                for action in self.dynamic_feasibility(self.design.copy(), self.current_step)
            )
        allowed = set(actions)
        if not allowed.issubset(set(self.action_domains[self.current_step])):
            raise ValueError("dynamic feasibility returned an action outside its domain")
        return tuple(action for action in actions if action in allowed)

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_count, dtype=np.int8)
        for action in self.feasible_actions():
            mask[action] = 1
        return mask

    def _observation(self) -> dict[str, np.ndarray]:
        step_fraction = self.current_step / float(self.num_variables)
        state = np.concatenate(
            [self.design.astype(np.float32), np.asarray([step_fraction], dtype=np.float32)]
        )
        return {"state": state, "action_mask": self.action_mask()}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.design = self.initial_design.copy()
        self.current_step = 0
        return self._observation(), {}

    def step(self, action: int):
        if self.current_step >= self.num_variables:
            raise RuntimeError("step() called after the episode terminated")
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"action id is outside the action space: {action}")
        if action not in self.feasible_actions():
            raise ValueError(f"action {action} is masked at step {self.current_step}")

        self.design[self.current_step] = float(action)
        self.current_step += 1
        terminated = self.current_step == self.num_variables
        reward = 0.0
        info: dict[str, Any] = {
            "status": "in_progress",
            "terminal_evaluation": False,
        }
        if terminated:
            if self.evaluator is None:
                raise RuntimeError("a terminal evaluator is required for the final step")
            self.evaluation_count += 1
            reward = float(self.evaluator(self.design.copy()))
            info = {
                "status": "terminal_evaluation",
                "terminal_evaluation": True,
                "evaluation_count": self.evaluation_count,
            }
        return self._observation(), reward, terminated, False, info

    def render(self) -> None:
        print(
            f"step={self.current_step}/{self.num_variables}, "
            f"evaluations={self.evaluation_count}"
        )
