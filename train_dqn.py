"""Train DQN on SnakeEnv; optional pygame view + HUD."""

from __future__ import annotations

import argparse
import sys
from collections import deque

from agent.dqn import DQNAgent, default_config
from env.snake_env import SnakeEnv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--render", action="store_true", help="pygame: grid + metrics")
    p.add_argument("--fps", type=int, default=120, help="cap redraw rate when --render")
    args = p.parse_args()

    env = SnakeEnv()
    agent = DQNAgent(default_config())

    renderer = None
    if args.render:
        from env.train_renderer import TrainRenderer

        renderer = TrainRenderer(fps_limit=args.fps)

    recent_scores: deque[int] = deque(maxlen=30)
    recent_returns: deque[float] = deque(maxlen=30)
    last_loss: float | None = None
    last_action = 0
    last_r = 0.0
    fps_smooth = 0.0

    try:
        for ep in range(1, args.episodes + 1):
            obs = env.reset()
            done = False
            ep_reward = 0.0
            steps = 0
            while not done:
                if renderer and renderer.poll_quit():
                    return

                a = agent.act(obs, explore=True)
                obs2, r, done, _ = env.step(a)
                agent.frames += 1
                agent.buf.push(obs, a, r, obs2, done)
                lo = agent.learn()
                if lo is not None:
                    last_loss = lo

                last_action = a
                last_r = float(r)
                obs = obs2
                ep_reward += r
                steps += 1

                if renderer:
                    mean_s = sum(recent_scores) / len(recent_scores) if recent_scores else None
                    mean_er = sum(recent_returns) / len(recent_returns) if recent_returns else None
                    renderer.draw(
                        env,
                        episode=ep,
                        step_in_ep=steps,
                        ep_return=ep_reward,
                        last_reward=last_r,
                        last_action=last_action,
                        epsilon=agent.epsilon(),
                        last_loss=last_loss,
                        buf_len=len(agent.buf),
                        mean_score=mean_s,
                        mean_ep_return=mean_er,
                        fps_actual=fps_smooth,
                    )
                    fps_smooth = 0.9 * fps_smooth + 0.1 * renderer.flip()

            recent_scores.append(env.score)
            recent_returns.append(ep_reward)

            if ep % args.log_every == 0:
                print(
                    f"ep {ep}  score={env.score}  steps={steps}  reward={ep_reward:.1f}  eps={agent.epsilon():.3f}"
                )
    finally:
        if renderer:
            renderer.close()


if __name__ == "__main__":
    main()
    sys.exit(0)
