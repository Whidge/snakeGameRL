"""Minimal snake game: pygame, arrow keys, score, game over + restart."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import pygame

from env.config import GRID_COLS, GRID_ROWS
from env.snake_core import Dir, OPPOSITE, initial_snake_food, step_move


# --- pygame-only config ---
CELL = 20
WINDOW_W = GRID_COLS * CELL
WINDOW_H = GRID_ROWS * CELL
FPS = 10
BG_COLOR = (20, 24, 32)
GRID_COLOR = (35, 42, 55)
SNAKE_COLOR = (80, 200, 120)
HEAD_COLOR = (120, 255, 160)
FOOD_COLOR = (220, 90, 90)
TEXT_COLOR = (220, 224, 230)


@dataclass
class GameState:
    snake: list[tuple[int, int]]
    direction: Dir
    pending_dir: Dir
    food: tuple[int, int]
    score: int
    alive: bool


def initial_state() -> GameState:
    snake, food = initial_snake_food()
    return GameState(
        snake=snake,
        direction=Dir.RIGHT,
        pending_dir=Dir.RIGHT,
        food=food,
        score=0,
        alive=True,
    )


def reset_state(state: GameState) -> None:
    snake, food = initial_snake_food()
    state.snake = snake
    state.direction = Dir.RIGHT
    state.pending_dir = Dir.RIGHT
    state.food = food
    state.score = 0
    state.alive = True


def step(state: GameState) -> None:
    if not state.alive:
        return

    state.direction = state.pending_dir
    snake, food, score, alive, _ = step_move(
        state.snake, state.direction, state.food, state.score
    )
    state.snake, state.food, state.score, state.alive = snake, food, score, alive


def try_set_direction(state: GameState, new_dir: Dir) -> None:
    if new_dir != OPPOSITE[state.direction]:
        state.pending_dir = new_dir


def draw_grid(surface: pygame.Surface) -> None:
    for x in range(0, WINDOW_W, CELL):
        pygame.draw.line(surface, GRID_COLOR, (x, 0), (x, WINDOW_H))
    for y in range(0, WINDOW_H, CELL):
        pygame.draw.line(surface, GRID_COLOR, (0, y), (WINDOW_W, y))


def draw_cell(surface: pygame.Surface, cell: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = cell
    rect = pygame.Rect(x * CELL, y * CELL, CELL, CELL)
    pygame.draw.rect(surface, color, rect.inflate(-2, -2), border_radius=4)


def draw_state(surface: pygame.Surface, font: pygame.font.Font, state: GameState) -> None:
    surface.fill(BG_COLOR)
    draw_grid(surface)

    for i, seg in enumerate(state.snake):
        draw_cell(surface, seg, HEAD_COLOR if i == 0 else SNAKE_COLOR)
    draw_cell(surface, state.food, FOOD_COLOR)

    score_surf = font.render(f"Score: {state.score}", True, TEXT_COLOR)
    surface.blit(score_surf, (8, 8))

    if not state.alive:
        msg = font.render("Game Over — SPACE restart / ESC quit", True, TEXT_COLOR)
        surface.blit(msg, (8, WINDOW_H - 32))


def handle_events(state: GameState) -> bool:
    """Return False if quit."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            if not state.alive and event.key == pygame.K_SPACE:
                reset_state(state)
                continue
            if event.key == pygame.K_UP:
                try_set_direction(state, Dir.UP)
            elif event.key == pygame.K_DOWN:
                try_set_direction(state, Dir.DOWN)
            elif event.key == pygame.K_LEFT:
                try_set_direction(state, Dir.LEFT)
            elif event.key == pygame.K_RIGHT:
                try_set_direction(state, Dir.RIGHT)
    return True


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Snake")
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 22)
    state = initial_state()

    running = True
    while running:
        running = handle_events(state)
        step(state)
        draw_state(screen, font, state)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
