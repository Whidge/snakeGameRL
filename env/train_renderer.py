"""Pygame panel: grid + HUD for live training metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pygame

from env.config import GRID_COLS, GRID_ROWS

if TYPE_CHECKING:
    from env.snake_env import SnakeEnv

CELL = 20
HUD_W = 260
BG = (20, 24, 32)
GRID = (35, 42, 55)
SNAKE = (80, 200, 120)
HEAD = (120, 255, 160)
FOOD = (220, 90, 90)
HUD_BG = (14, 18, 26)
TEXT = (220, 224, 230)
TEXT_DIM = (140, 150, 170)


class TrainRenderer:
    def __init__(self, title: str = "DQN Snake — training", fps_limit: int = 120) -> None:
        pygame.init()
        pygame.display.set_caption(title)
        self.grid_px_w = GRID_COLS * CELL
        self.grid_px_h = GRID_ROWS * CELL
        self.screen = pygame.display.set_mode((self.grid_px_w + HUD_W, self.grid_px_h))
        self.clock = pygame.time.Clock()
        self.fps_limit = fps_limit
        self.font = pygame.font.SysFont("consolas", 16)
        self.font_s = pygame.font.SysFont("consolas", 13)

    def close(self) -> None:
        pygame.quit()

    def poll_quit(self) -> bool:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return True
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                return True
        return False

    def _cell_rect(self, cx: int, cy: int) -> pygame.Rect:
        return pygame.Rect(cx * CELL, cy * CELL, CELL, CELL).inflate(-2, -2)

    def _draw_grid(self) -> None:
        for x in range(0, self.grid_px_w + 1, CELL):
            pygame.draw.line(self.screen, GRID, (x, 0), (x, self.grid_px_h))
        for y in range(0, self.grid_px_h + 1, CELL):
            pygame.draw.line(self.screen, GRID, (0, y), (self.grid_px_w, y))

    def _blit_line(self, x: int, y: int, text: str, dim: bool = False) -> int:
        surf = self.font.render(text, True, TEXT_DIM if dim else TEXT)
        self.screen.blit(surf, (x, y))
        return y + surf.get_height() + 2

    def _blit_small(self, x: int, y: int, text: str) -> int:
        surf = self.font_s.render(text, True, TEXT_DIM)
        self.screen.blit(surf, (x, y))
        return y + surf.get_height() + 1

    def draw(
        self,
        env: SnakeEnv,
        *,
        episode: int,
        step_in_ep: int,
        ep_return: float,
        last_reward: float,
        last_action: int,
        epsilon: float,
        last_loss: float | None,
        buf_len: int,
        mean_score: float | None,
        mean_ep_return: float | None,
        fps_actual: float,
    ) -> None:
        self.screen.fill(BG, (0, 0, self.grid_px_w, self.grid_px_h))
        self._draw_grid()

        for i, seg in enumerate(env.snake):
            pygame.draw.rect(self.screen, HEAD if i == 0 else SNAKE, self._cell_rect(*seg), border_radius=4)
        pygame.draw.rect(self.screen, FOOD, self._cell_rect(*env.food), border_radius=4)

        hx = self.grid_px_w
        self.screen.fill(HUD_BG, (hx, 0, HUD_W, self.grid_px_h))
        x = hx + 10
        y = 10

        y = self._blit_line(x, y, f"Episode    {episode}")
        y = self._blit_line(x, y, f"Step (ep)  {step_in_ep}")
        y = self._blit_line(x, y, f"Score      {env.score}")
        y = self._blit_line(x, y, f"Alive      {env.alive}")
        y = self._blit_line(x, y, f"Ep return  {ep_return:.2f}")
        y = self._blit_line(x, y, f"Last r     {last_reward:+.3f}")
        act = ("straight", "right", "left")[last_action] if 0 <= last_action <= 2 else str(last_action)
        y = self._blit_line(x, y, f"Action     {act}")
        y = self._blit_line(x, y, f"Epsilon    {epsilon:.4f}")
        lo = f"{last_loss:.5f}" if last_loss is not None else "—"
        y = self._blit_line(x, y, f"Loss       {lo}")
        y = self._blit_line(x, y, f"Buffer     {buf_len}")
        ms = f"{mean_score:.2f}" if mean_score is not None else "—"
        y = self._blit_line(x, y, f"Avg score  {ms} (30 ep)")
        mer = f"{mean_ep_return:.2f}" if mean_ep_return is not None else "—"
        y = self._blit_line(x, y, f"Avg ret    {mer} (30 ep)")
        y += 6
        y = self._blit_small(x, y, f"FPS ~{fps_actual:.0f}  ESC quit")
        pygame.draw.line(self.screen, GRID, (hx, 0), (hx, self.grid_px_h), 2)

    def flip(self) -> float:
        pygame.display.flip()
        self.clock.tick(self.fps_limit)
        return float(self.clock.get_fps())
