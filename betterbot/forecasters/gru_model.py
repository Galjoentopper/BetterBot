"""PyTorch GRU forecaster for sequential feature data."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:  # pragma: no cover - environment specific dependency loading
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required. Install via 'pip install torch'.") from exc

LOGGER = logging.getLogger(__name__)


def _to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


class GRUModel(nn.Module):
    """Simple GRU network with a linear output head."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        output, _ = self.gru(x)
        last_hidden = output[:, -1, :]
        return self.head(last_hidden)


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5


class GRUForecaster:
    """Trainable GRU forecaster for sequential input data."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        device: Optional[str] = None,
        training: TrainingConfig | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = GRUModel(input_size, hidden_size, num_layers, dropout).to(self.device)
        self.training_config = training or TrainingConfig()
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        LOGGER.debug("Initialised GRUForecaster on device %s", self.device)

    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Train the model using mini-batch gradient descent."""

        if features.ndim != 3:
            raise ValueError("Features must be a 3D array: (samples, seq_len, features)")
        if targets.ndim == 1:
            targets = targets[:, None]

        dataset = TensorDataset(_to_tensor(features, self.device), _to_tensor(targets, self.device))
        loader = DataLoader(dataset, batch_size=self.training_config.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.training_config.epochs):
            running_loss = 0.0
            for batch_features, batch_targets in loader:
                self.optimizer.zero_grad()
                predictions = self.model(batch_features)
                loss = self.loss_fn(predictions, batch_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                running_loss += loss.item() * batch_features.size(0)
            epoch_loss = running_loss / len(dataset)
            LOGGER.debug("GRU epoch %s | loss=%.6f", epoch + 1, epoch_loss)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Run inference on ``features`` and return predictions as a NumPy array."""

        if features.ndim == 2:
            features = features[None, ...]
        tensor = _to_tensor(features, self.device)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(tensor).cpu().numpy().squeeze(-1)
        return predictions

    def save_model(self, path: str | Path) -> None:
        """Persist the model parameters to ``path``."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), destination)
        LOGGER.info("Saved GRU model to %s", destination)

    def load_model(self, path: str | Path) -> None:
        """Load parameters from file."""

        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"GRU model file not found: {source}")
        state_dict = torch.load(source, map_location=self.device)
        self.model.load_state_dict(state_dict)
        LOGGER.info("Loaded GRU model from %s", source)


__all__ = ["GRUForecaster", "TrainingConfig"]
