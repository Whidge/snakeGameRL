"""Training loop in a worker thread (pause / stop safe)."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

from agent.dqn import DQNAgent, DQNConfig
from env.snake_env import RewardConfig, SnakeEnv


class TrainWorker(QObject):
    """Runs on QThread; connect thread.started to run_training."""

    metrics = pyqtSignal(dict)  # end of episode
    frame = pyqtSignal(dict)  # board + HUD (throttled)
    loss_step = pyqtSignal(dict)  # {"train_step", "loss"} when learn ran
    finished = pyqtSignal(dict)  # summary

    def __init__(
        self,
        cfg: DQNConfig,
        episodes: int,
        render_every: int,
        rewards: RewardConfig | None = None,
    ) -> None:
        super().__init__()
        self._cfg = cfg
        self._episodes = max(1, episodes)
        self._render_every = max(1, render_every)
        self._rewards = rewards if rewards is not None else RewardConfig()
        self._lock = threading.Lock()
        self._pause = False
        self._stop = False

    def set_pause(self, paused: bool) -> None:
        with self._lock:
            self._pause = paused

    def request_stop(self) -> None:
        with self._lock:
            self._stop = True

    def _is_pause(self) -> bool:
        with self._lock:
            return self._pause

    def _is_stop(self) -> bool:
        with self._lock:
            return self._stop

    def _wait_while_paused(self) -> bool:
        """Return True if should abort."""
        while self._is_pause() and not self._is_stop():
            time.sleep(0.05)
        return self._is_stop()

    def run_training(self) -> None:
        t0 = time.perf_counter()
        env = SnakeEnv(rewards=self._rewards)
        agent = DQNAgent(self._cfg)

        scores_hist: deque[int] = deque(maxlen=100)
        returns_hist: deque[float] = deque(maxlen=100)
        peak_score = 0
        total_env_steps = 0
        last_loss: float | None = None
        reason = "complete"

        try:
            for ep in range(1, self._episodes + 1):
                if self._wait_while_paused():
                    reason = "stopped"
                    break

                obs = env.reset()
                done = False
                ep_reward = 0.0
                steps = 0
                env_step_in_ep = 0

                while not done:
                    if self._wait_while_paused():
                        reason = "stopped"
                        break

                    a = agent.act(obs, explore=True)
                    obs2, r, done, _ = env.step(a)
                    agent.frames += 1
                    agent.buf.push(obs, a, r, obs2, done)
                    lo = agent.learn()
                    if lo is not None:
                        last_loss = lo
                        self.loss_step.emit({"train_step": agent.train_steps, "loss": lo})

                    obs = obs2
                    ep_reward += float(r)
                    steps += 1
                    env_step_in_ep += 1
                    total_env_steps += 1

                    if env_step_in_ep % self._render_every == 0:
                        self.frame.emit(
                            {
                                "snake": list(env.snake),
                                "food": tuple(env.food),
                                "alive": env.alive,
                                "score": env.score,
                                "episode": ep,
                                "step_in_ep": env_step_in_ep,
                                "ep_return": ep_reward,
                                "epsilon": agent.epsilon(),
                                "frames": agent.frames,
                                "loss": last_loss,
                                "buffer": len(agent.buf),
                            }
                        )

                    if self._is_stop():
                        reason = "stopped"
                        break

                if reason == "stopped":
                    break

                sc = env.score
                peak_score = max(peak_score, sc)
                scores_hist.append(sc)
                returns_hist.append(ep_reward)

                mean30_s = mean100_s = mean30_r = mean100_r = None
                if scores_hist:
                    tail30 = list(scores_hist)[-30:]
                    mean30_s = sum(tail30) / len(tail30)
                    mean100_s = sum(scores_hist) / len(scores_hist)
                if returns_hist:
                    tr30 = list(returns_hist)[-30:]
                    mean30_r = sum(tr30) / len(tr30)
                    mean100_r = sum(returns_hist) / len(returns_hist)

                self.metrics.emit(
                    {
                        "episode": ep,
                        "score": sc,
                        "return": ep_reward,
                        "steps": steps,
                        "epsilon": agent.epsilon(),
                        "loss": last_loss,
                        "buffer": len(agent.buf),
                        "mean_score_30": mean30_s,
                        "mean_score_100": mean100_s,
                        "mean_return_30": mean30_r,
                        "mean_return_100": mean100_r,
                        "frames": agent.frames,
                        "train_steps": agent.train_steps,
                    }
                )

                self.frame.emit(
                    {
                        "snake": list(env.snake),
                        "food": tuple(env.food),
                        "alive": env.alive,
                        "score": env.score,
                        "episode": ep,
                        "step_in_ep": steps,
                        "ep_return": ep_reward,
                        "epsilon": agent.epsilon(),
                        "frames": agent.frames,
                        "loss": last_loss,
                        "buffer": len(agent.buf),
                    }
                )

        finally:
            elapsed = time.perf_counter() - t0
            tail30 = list(scores_hist)[-30:] if scores_hist else []
            tail100 = list(scores_hist)
            summary: dict[str, Any] = {
                "reason": reason,
                "episodes_run": len(scores_hist),
                "peak_score": peak_score,
                "mean_score_30": sum(tail30) / len(tail30) if tail30 else None,
                "mean_score_100": sum(tail100) / len(tail100) if tail100 else None,
                "total_env_steps": total_env_steps,
                "final_epsilon": agent.epsilon(),
                "seconds": elapsed,
                "rewards": env.reward_config.to_dict(),
                "agent": agent,
            }
            self.finished.emit(summary)
