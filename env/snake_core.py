"""Headless snake rules: directions, one physics step, spawn."""

from __future__ import annotations

import random
from enum import Enum

from env.config import GRID_COLS, GRID_ROWS


class Dir(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


OPPOSITE: dict[Dir, Dir] = {
    Dir.UP: Dir.DOWN,
    Dir.DOWN: Dir.UP,
    Dir.LEFT: Dir.RIGHT,
    Dir.RIGHT: Dir.LEFT,
}

# Relative turns from current heading (action 1 = right, 2 = left)
RIGHT_TURN: dict[Dir, Dir] = {
    Dir.UP: Dir.RIGHT,
    Dir.RIGHT: Dir.DOWN,
    Dir.DOWN: Dir.LEFT,
    Dir.LEFT: Dir.UP,
}
LEFT_TURN: dict[Dir, Dir] = {v: k for k, v in RIGHT_TURN.items()}


def in_bounds(cell: tuple[int, int]) -> bool:
    x, y = cell
    return 0 <= x < GRID_COLS and 0 <= y < GRID_ROWS


def random_cell(exclude: set[tuple[int, int]]) -> tuple[int, int]:
    while True:
        c = (random.randrange(GRID_COLS), random.randrange(GRID_ROWS))
        if c not in exclude:
            return c


def initial_snake_food() -> tuple[list[tuple[int, int]], tuple[int, int]]:
    mid = (GRID_COLS // 2, GRID_ROWS // 2)
    snake = [(mid[0], mid[1]), (mid[0] - 1, mid[1]), (mid[0] - 2, mid[1])]
    food = random_cell(set(snake))
    return snake, food


def step_move(
    snake: list[tuple[int, int]],
    direction: Dir,
    food: tuple[int, int],
    score: int,
) -> tuple[list[tuple[int, int]], tuple[int, int], int, bool, bool]:
    """One move. Returns (snake, food, score, alive, ate_food)."""
    dx, dy = direction.value
    hx, hy = snake[0]
    new_head = (hx + dx, hy + dy)

    if not in_bounds(new_head):
        return snake, food, score, False, False
    if new_head in snake[:-1]:
        return snake, food, score, False, False

    ate = new_head == food
    new_snake = [new_head, *snake]
    if not ate:
        new_snake.pop()
    new_food = food
    new_score = score
    if ate:
        new_score += 1
        new_food = random_cell(set(new_snake))
    return new_snake, new_food, new_score, True, ate


def direction_from_action(current: Dir, action: int) -> Dir:
    """0 straight, 1 right, 2 left."""
    if action == 1:
        return RIGHT_TURN[current]
    if action == 2:
        return LEFT_TURN[current]
    return current
