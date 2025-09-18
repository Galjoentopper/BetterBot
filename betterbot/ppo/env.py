"""Custom OpenAI Gym environment representing a single trading episode."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:  # pragma: no cover - dependency resolution
    import gymnasium as gym
except ImportError:  # pragma: no cover    
    import gym  # type: ignore

from gym import spaces

LOGGER = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    """Internal state of the trading portfolio."""

    cash: float
    position_size: float = 0.0
    entry_price: float = 0.0

    def total_value(self, price: float) -> float:
        return self.cash + self.position_size * price


class TradingEnv(gym.Env):  # type: ignore[misc]
    """A trading environment producing rewards based on portfolio value deltas."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        price_column: str = "close",
        initial_cash: float = 1000.0,
        trading_cost: float = 0.001,
        volatility_penalty: float = 0.0,
        max_position: float = 1.0,
    ) -> None:
        super().__init__()
        if price_column not in data:
            raise KeyError(f"Data frame must include '{price_column}' column")
        self.data = data.reset_index(drop=True)
        self.price_column = price_column
        self.feature_columns = feature_columns or [
            col for col in data.columns if col not in {price_column}
        ]
        if not self.feature_columns:
            raise ValueError("At least one feature column is required for the environment")
        self.initial_cash = initial_cash
        self.trading_cost = trading_cost
        self.volatility_penalty = volatility_penalty
        self.max_position = max_position

        self.action_space = spaces.Discrete(3)
        obs_length = len(self.feature_columns) + 2  # features + position + cash utilisation
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_length,),
            dtype=np.float32,
        )
        self._portfolio = PortfolioState(cash=initial_cash)
        self._current_step = 0
        self._previous_value = initial_cash

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: Optional[Dict[str, float]] = None):  # type: ignore[override]
        super().reset(seed=seed)
        self._current_step = 0
        self._portfolio = PortfolioState(cash=self.initial_cash)
        self._previous_value = self.initial_cash
        observation = self._build_observation(self._current_step)
        LOGGER.debug("TradingEnv reset: cash=%.2f", self.initial_cash)
        return observation, {}

    def step(self, action: int):  # type: ignore[override]
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} outside of action space")

        current_price = float(self.data.at[self._current_step, self.price_column])
        LOGGER.debug(
            "Step %s | action=%s | price=%.2f | position=%.6f",
            self._current_step,
            action,
            current_price,
            self._portfolio.position_size,
        )

        # Execute trade logic
        if action == 1:  # buy / go long
            self._go_long(current_price)
        elif action == 2:  # sell / flatten
            self._go_flat(current_price)
        # action=0: hold

        next_step = self._current_step + 1
        terminated = next_step >= len(self.data) - 1
        next_price = float(self.data.at[next_step, self.price_column]) if not terminated else current_price

        portfolio_value = self._portfolio.total_value(next_price)
        reward = portfolio_value - self._previous_value
        if self.volatility_penalty > 0:
            reward -= self.volatility_penalty * abs(self._portfolio.position_size)

        self._previous_value = portfolio_value
        self._current_step = next_step

        observation = self._build_observation(self._current_step)
        info = {
            "portfolio_value": portfolio_value,
            "cash": self._portfolio.cash,
            "position": self._portfolio.position_size,
            "price": next_price,
        }
        return observation, float(reward), terminated, False, info

    def _go_long(self, price: float) -> None:
        if self._portfolio.position_size > 0:
            return
        available_units = min(self.max_position, self._portfolio.cash / price)
        cost = available_units * price * (1 + self.trading_cost)
        if cost > self._portfolio.cash:
            return
        self._portfolio.cash -= cost
        self._portfolio.position_size = available_units
        self._portfolio.entry_price = price
        LOGGER.debug("Opened long position: size=%.6f cost=%.2f", available_units, cost)

    def _go_flat(self, price: float) -> None:
        if self._portfolio.position_size <= 0:
            return
        proceeds = self._portfolio.position_size * price * (1 - self.trading_cost)
        self._portfolio.cash += proceeds
        LOGGER.debug("Closed position: proceeds=%.2f", proceeds)
        self._portfolio.position_size = 0.0
        self._portfolio.entry_price = 0.0

    def _build_observation(self, step: int) -> np.ndarray:
        row = self.data.loc[step, self.feature_columns]
        cash_utilisation = 1 - (self._portfolio.cash / self.initial_cash)
        observation = np.concatenate(
            [row.values.astype(np.float32), np.array([self._portfolio.position_size, cash_utilisation], dtype=np.float32)]
        )
        return observation.astype(np.float32)

    def render(self):  # pragma: no cover - use logging instead
        LOGGER.info(
            "Step %s | Value %.2f | Cash %.2f | Position %.6f",
            self._current_step,
            self._previous_value,
            self._portfolio.cash,
            self._portfolio.position_size,
        )


__all__ = ["TradingEnv"]
