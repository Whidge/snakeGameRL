"""Gym-style snake env: 11-dim vector obs, 3 relative actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from env.config import GRID_COLS, GRID_ROWS
from env.snake_core import (
    Dir,
    LEFT_TURN,
    RIGHT_TURN,
    direction_from_action,
    in_bounds,
    initial_snake_food,
    step_move,
)

OBS_DIM = 11
N_ACTIONS = 3


@dataclass(frozen=True)
class RewardConfig:
    """Per-step shaping: eat bonus, death penalty, living cost."""

    r_eat: float = 10.0
    r_death: float = -10.0
    r_step: float = -0.005

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


# Legacy module constants (match RewardConfig defaults)
R_EAT, R_DEATH, R_STEP = 10.0, -10.0, -0.005


class SnakeEnv:
    def __init__(
        self,
        max_steps_without_food: int | None = None,
        rewards: RewardConfig | None = None,
    ) -> None:
        self._rewards = rewards if rewards is not None else RewardConfig()
        self._max_idle = max_steps_without_food or (GRID_COLS * GRID_ROWS * 2)
        self._snake: list[tuple[int, int]] = []
        self._direction = Dir.RIGHT
        self._food: tuple[int, int] = (0, 0)
        self._score = 0
        self._alive = True
        self._steps_idle = 0

    def reset(self) -> np.ndarray:
        self._snake, self._food = initial_snake_food()
        self._direction = Dir.RIGHT
        self._score = 0
        self._alive = True
        self._steps_idle = 0
        return self._observe()

    def _cell_ahead(self, head: tuple[int, int], d: Dir) -> tuple[int, int]:
        dx, dy = d.value
        return head[0] + dx, head[1] + dy

    def _danger(self, cell: tuple[int, int]) -> float:
        if not in_bounds(cell):
            return 1.0
        if cell in self._snake[:-1]:
            return 1.0
        return 0.0

    def _observe(self) -> np.ndarray:
        head = self._snake[0]
        d = self._direction
        straight = self._cell_ahead(head, d)
        right = self._cell_ahead(head, RIGHT_TURN[d])
        left = self._cell_ahead(head, LEFT_TURN[d])

        obs: list[float] = [
            self._danger(straight),
            self._danger(right),
            self._danger(left),
        ]
        for e in (Dir.UP, Dir.RIGHT, Dir.DOWN, Dir.LEFT):
            obs.append(1.0 if d == e else 0.0)

        fx, fy = self._food
        hx, hy = head
        obs.extend(
            [
                1.0 if fy < hy else 0.0,
                1.0 if fy > hy else 0.0,
                1.0 if fx < hx else 0.0,
                1.0 if fx > hx else 0.0,
            ]
        )
        return np.asarray(obs, dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if not self._alive:
            return self._observe(), 0.0, True, {}

        self._direction = direction_from_action(self._direction, int(action))
        snake, food, score, alive, ate = step_move(
            self._snake, self._direction, self._food, self._score
        )
        self._snake, self._food, self._score, self._alive = snake, food, score, alive

        rw = self._rewards
        reward = rw.r_step
        if ate:
            reward += rw.r_eat
            self._steps_idle = 0
        else:
            self._steps_idle += 1

        done = False
        info: dict = {}

        if not self._alive:
            reward += rw.r_death
            done = True
        elif self._steps_idle >= self._max_idle:
            done = True
            info["timeout"] = True

        return self._observe(), reward, done, info

    @property
    def snake(self) -> list[tuple[int, int]]:
        return self._snake

    @property
    def food(self) -> tuple[int, int]:
        return self._food

    @property
    def score(self) -> int:
        return self._score

    @property
    def alive(self) -> bool:
        return self._alive

    @property
    def direction(self) -> Dir:
        return self._direction

    @property
    def reward_config(self) -> RewardConfig:
        return self._rewards
