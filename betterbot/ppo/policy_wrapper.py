"""Wrapper around Stable Baselines 3 PPO policy."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

try:  # pragma: no cover - environment specific dependency loading
    from stable_baselines3 import PPO
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "stable-baselines3 is required. Install via 'pip install stable-baselines3'."
    ) from exc

from stable_baselines3.common.vec_env import DummyVecEnv

LOGGER = logging.getLogger(__name__)


class PPOPolicyWrapper:
    """Lightweight wrapper that hides SB3 specifics from the rest of the codebase."""

    def __init__(self, policy: str = "MlpPolicy", **ppo_kwargs: Any) -> None:
        self.policy = policy
        self.ppo_kwargs = ppo_kwargs
        self.model: Optional[PPO] = None

    def train(self, env, timesteps: int) -> None:
        LOGGER.info("Training PPO policy for %s timesteps", timesteps)
        if isinstance(env, DummyVecEnv):
            vec_env = env
        else:
            if callable(env):
                env_fn = env
            else:
                env_instance = env
                env_fn = lambda env_instance=env_instance: env_instance
            vec_env = DummyVecEnv([env_fn])
        self.model = PPO(self.policy, vec_env, verbose=0, **self.ppo_kwargs)
        self.model.learn(total_timesteps=timesteps, progress_bar=False)

    def predict(self, observation, deterministic: bool = True):
        if self.model is None:
            raise RuntimeError("PPO model has not been trained or loaded.")
        action, _state = self.model.predict(observation, deterministic=deterministic)
        return int(action)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save PPO model before training.")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(destination)
        LOGGER.info("Saved PPO policy to %s", destination)

    def load(self, path: str | Path) -> None:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"PPO policy file not found: {source}")
        self.model = PPO.load(source)
        LOGGER.info("Loaded PPO policy from %s", source)


__all__ = ["PPOPolicyWrapper"]
