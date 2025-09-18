"""Generate daily operational reports for BetterBot."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

try:  # pragma: no cover - optional dependency
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore

from betterbot.utils.config import load_config
from betterbot.utils.registry import ModelRegistry
from betterbot.utils.telegram_alerts import send_report

LOGGER = logging.getLogger(__name__)


def _system_metrics() -> Dict[str, float]:
    if psutil is None:
        return {"cpu": 0.0, "memory": 0.0}
    return {
        "cpu": psutil.cpu_percent(interval=1),
        "memory": psutil.virtual_memory().percent,
    }


def _format_report(metadata: Dict, metrics: Dict[str, float]) -> str:
    model_metrics = metadata.get("metrics", {})
    lines = [
        "*BetterBot Daily Report*",
        f"Date: {datetime.utcnow():%Y-%m-%d %H:%M UTC}",
        "",
        "_Model_",
        f"• Accuracy: {model_metrics.get('lightgbm_accuracy', float('nan')):.3f}",
        f"• GRU MSE: {model_metrics.get('gru_mse', float('nan')):.4f}",
        f"• PPO Reward: {model_metrics.get('ppo_reward', float('nan')):.2f}",
        "",
        "_System_",
        f"• CPU Usage: {metrics['cpu']:.1f}%",
        f"• Memory Usage: {metrics['memory']:.1f}%",
    ]
    return "\n".join(lines)


def send_daily_report(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    registry = ModelRegistry(Path(cfg["paths"]["registry_root"]))
    bundle = registry.load()
    system = _system_metrics()
    report = _format_report(bundle["metadata"], system)
    LOGGER.info("Dispatching daily report")
    send_report(report)


__all__ = ["send_daily_report"]
