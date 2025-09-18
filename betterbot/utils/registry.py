"""Utilities for persisting and retrieving BetterBot models."""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from betterbot.forecasters.gru_model import GRUForecaster
from betterbot.forecasters.lightgbm_model import LightGBMForecaster
from betterbot.ppo.policy_wrapper import PPOPolicyWrapper

LOGGER = logging.getLogger(__name__)

LATEST_POINTER = "latest.txt"


class ModelRegistry:
    """Filesystem-based registry storing model artefacts under timestamped folders."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        LOGGER.debug("Model registry initialised at %s", self.root)

    # ------------------------------------------------------------------
    def register(
        self,
        metadata: Dict,
        lightgbm: LightGBMForecaster,
        gru: GRUForecaster,
        ppo: PPOPolicyWrapper,
        model_id: Optional[str] = None,
    ) -> Path:
        """Persist the supplied models and metadata, returning the model folder."""

        model_id = model_id or datetime.utcnow().strftime("%Y%m%d%H%M%S")
        target_dir = self.root / model_id
        if target_dir.exists():
            raise FileExistsError(f"Model id already exists: {model_id}")

        temp_dir = target_dir.with_suffix(".tmp")
        temp_dir.mkdir(parents=True)

        lightgbm_path = temp_dir / "lightgbm.pkl"
        gru_path = temp_dir / "gru.pt"
        ppo_path = temp_dir / "ppo.zip"
        metadata_path = temp_dir / "metadata.json"

        lightgbm.save_model(lightgbm_path)
        gru.save_model(gru_path)
        ppo.save(ppo_path)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

        shutil.move(str(temp_dir), target_dir)
        self._update_latest(model_id)
        LOGGER.info("Registered new model %s", model_id)
        return target_dir

    def load(self, model_id: Optional[str] = None):
        """Load a model bundle. When ``model_id`` is ``None`` the latest model is used."""

        model_id = model_id or self.get_latest_model_id()
        if model_id is None:
            raise FileNotFoundError("No models have been registered yet.")
        folder = self.root / model_id
        if not folder.exists():
            raise FileNotFoundError(f"Model folder not found: {folder}")
        metadata = self._load_metadata(folder)

        lgb = LightGBMForecaster()
        lgb.load_model(folder / "lightgbm.pkl")

        gru_settings = metadata.get("models", {}).get("gru", {})
        input_size = gru_settings.get("input_size")
        if input_size is None:
            raise KeyError("Metadata missing GRU input size, cannot load model.")
        gru = GRUForecaster(
            input_size=input_size,
            hidden_size=gru_settings.get("hidden_size", 64),
            num_layers=gru_settings.get("num_layers", 2),
        )
        gru.load_model(folder / "gru.pt")

        ppo = PPOPolicyWrapper()
        ppo.load(folder / "ppo.zip")

        return {"metadata": metadata, "lightgbm": lgb, "gru": gru, "ppo": ppo}

    def list_models(self) -> Dict[str, Path]:
        """Return a dictionary mapping model ids to their folder paths."""

        models: Dict[str, Path] = {}
        for metadata_file in sorted(self.root.glob("*/metadata.json")):
            models[metadata_file.parent.name] = metadata_file.parent
        return models

    def get_latest_model_id(self) -> Optional[str]:
        pointer = self.root / LATEST_POINTER
        if not pointer.exists():
            return None
        return pointer.read_text(encoding="utf-8").strip() or None

    # ------------------------------------------------------------------
    def _update_latest(self, model_id: str) -> None:
        pointer = self.root / LATEST_POINTER
        pointer.write_text(model_id, encoding="utf-8")

    @staticmethod
    def _load_metadata(folder: Path) -> Dict:
        metadata_path = folder / "metadata.json"
        with metadata_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


__all__ = ["ModelRegistry", "LATEST_POINTER"]
