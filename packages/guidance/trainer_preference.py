"""Deterministic preference learning trainer."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from packages.guidance.dataset import PreferenceDataset
from packages.guidance.models import RewardModel
from packages.telemetry.logger import get_logger
from packages.utils.config import load_config

__all__ = ["train_preference"]

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "training" / "preference.yaml"
)
_LOGGER = get_logger("praxis.guidance.preference")


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_config(
    config_path: str | Path | None,
    runtime_config: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if runtime_config is not None:
        return dict(runtime_config)
    path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH
    return load_config(path)


def train_preference(
    config_path: str | Path | None = None,
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a deterministic DPO-style optimisation over synthetic pairs."""

    cfg = _resolve_config(config_path, config)
    training_cfg = dict(cfg.get("training", {}))
    seed = int(cfg.get("seed", training_cfg.get("seed", 0)))
    _set_seed(seed)

    dataset = PreferenceDataset(cfg.get("dataset"))
    batch_size = int(training_cfg.get("batch_size", 8))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    model = RewardModel(cfg.get("model"))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training_cfg.get("learning_rate", 1e-2))
    )
    beta = float(training_cfg.get("beta", 0.1))
    epochs = int(training_cfg.get("epochs", 1))
    log_interval = max(1, int(training_cfg.get("log_interval", 1)))

    total_loss = 0.0
    total_batches = 0
    running_accuracy = 0.0

    for epoch in range(epochs):
        for batch_index, batch in enumerate(loader, start=1):
            optimizer.zero_grad()
            chosen_reward = model(batch["chosen"])
            rejected_reward = model(batch["rejected"])
            logits = beta * (chosen_reward - rejected_reward)
            loss = -F.logsigmoid(logits).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                accuracy = (chosen_reward > rejected_reward).float().mean().item()

            total_loss += loss.item()
            total_batches += 1
            running_accuracy += accuracy

            if batch_index % log_interval == 0:
                _LOGGER.info(
                    "epoch=%d batch=%d loss=%.4f accuracy=%.3f margin=%.4f",
                    epoch,
                    batch_index,
                    loss.item(),
                    accuracy,
                    logits.mean().item(),
                )

    checkpoint_dir = Path(
        training_cfg.get("checkpoint_dir", "data/processed/checkpoints/preference")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "preference.pt"
    torch.save(
        {
            "config": cfg,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epochs": epochs,
        },
        checkpoint_path,
    )

    mean_loss = total_loss / max(1, total_batches)
    mean_accuracy = running_accuracy / max(1, total_batches)
    _LOGGER.info(
        "training-complete epochs=%d mean_loss=%.4f accuracy=%.3f", epochs, mean_loss, mean_accuracy
    )
    return {
        "epochs": epochs,
        "mean_loss": mean_loss,
        "mean_accuracy": mean_accuracy,
        "checkpoint": str(checkpoint_path),
    }
