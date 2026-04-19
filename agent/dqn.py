"""Minimal DQN: MLP, replay buffer, target net."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env.snake_env import N_ACTIONS, OBS_DIM


@dataclass
class DQNConfig:
    lr: float = 5e-3
    gamma: float = 0.95
    batch: int = 64
    buffer_cap: int = 50_000
    warmup: int = 1000
    target_sync: int = 800
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 100_000


# CLI / legacy defaults
GAMMA = 0.95
LR = 5e-3
BATCH = 64
BUFFER_CAP = 50_000
WARMUP = 1000
TARGET_SYNC = 800
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 100_000


def default_config() -> DQNConfig:
    return DQNConfig(
        lr=LR,
        gamma=GAMMA,
        batch=BATCH,
        buffer_cap=BUFFER_CAP,
        warmup=WARMUP,
        target_sync=TARGET_SYNC,
        eps_start=EPS_START,
        eps_end=EPS_END,
        eps_decay_steps=EPS_DECAY_STEPS,
    )


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._buf: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(
        self,
        s: np.ndarray,
        a: int,
        r: float,
        s2: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append((s, a, r, s2, done))

    def sample(self, batch: int) -> tuple[torch.Tensor, ...]:
        batch = min(batch, len(self._buf))
        rows = random.sample(self._buf, batch)
        s = torch.tensor(np.stack([x[0] for x in rows]), dtype=torch.float32)
        a = torch.tensor([x[1] for x in rows], dtype=torch.int64).unsqueeze(1)
        r = torch.tensor([x[2] for x in rows], dtype=torch.float32).unsqueeze(1)
        s2 = torch.tensor(np.stack([x[3] for x in rows]), dtype=torch.float32)
        d = torch.tensor([x[4] for x in rows], dtype=torch.float32).unsqueeze(1)
        return s, a, r, s2, d

    def __len__(self) -> int:
        return len(self._buf)


class DQNAgent:
    def __init__(
        self,
        cfg: DQNConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.cfg = cfg or default_config()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = QNetwork().to(self.device)
        self.target = QNetwork().to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.opt = optim.Adam(self.policy.parameters(), lr=self.cfg.lr)
        self.buf = ReplayBuffer(self.cfg.buffer_cap)
        self.frames = 0
        self.train_steps = 0

    def epsilon(self) -> float:
        c = self.cfg
        frac = min(1.0, self.frames / max(1, c.eps_decay_steps))
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    def act(self, obs: np.ndarray, explore: bool) -> int:
        if explore and random.random() < self.epsilon():
            return random.randrange(N_ACTIONS)
        with torch.no_grad():
            q = self.policy(torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            return int(q.argmax(dim=1).item())

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            q = self.policy(torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            return q.squeeze(0).cpu().numpy()

    def learn(self) -> float | None:
        c = self.cfg
        if len(self.buf) < c.warmup:
            return None
        s, a, r, s2, d = self.buf.sample(c.batch)
        s, a, r, s2, d = s.to(self.device), a.to(self.device), r.to(self.device), s2.to(self.device), d.to(self.device)

        with torch.no_grad():
            y = r + (1.0 - d) * c.gamma * self.target(s2).max(dim=1, keepdim=True).values

        q = self.policy(s).gather(1, a)
        loss = nn.functional.mse_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.train_steps += 1
        if self.train_steps % c.target_sync == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return float(loss.item())

    def save(self, path: str | Path, meta: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "target": self.target.state_dict(),
                "cfg": asdict(self.cfg),
                "frames": self.frames,
                "train_steps": self.train_steps,
                "meta": meta or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: torch.device | None = None) -> DQNAgent:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        cfg = DQNConfig(**ckpt["cfg"])
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent = cls(cfg, device=dev)
        agent.policy.load_state_dict(ckpt["policy"])
        if "target" in ckpt:
            agent.target.load_state_dict(ckpt["target"])
        agent.frames = int(ckpt.get("frames", 0))
        agent.train_steps = int(ckpt.get("train_steps", 0))
        agent.policy.to(dev)
        agent.target.to(dev)
        agent.target.eval()
        agent.opt = optim.Adam(agent.policy.parameters(), lr=agent.cfg.lr)
        return agent
