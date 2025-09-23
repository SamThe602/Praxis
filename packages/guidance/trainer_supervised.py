"""Tiny supervised imitation trainer used for unit tests."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from packages.guidance.dataset import SupervisedDataset
from packages.guidance.models import PolicyValueModel
from packages.telemetry.logger import get_logger
from packages.utils.config import load_config

__all__ = ["train_supervised"]

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "training" / "supervised.yaml"
)
_LOGGER = get_logger("praxis.guidance.supervised")


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


def train_supervised(
    config_path: str | Path | None = None,
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a deterministic supervised imitation round.

    The helper returns a small metrics dictionary which makes it convenient for
    tests to assert loss magnitudes without having to inspect log files.
    """

    cfg = _resolve_config(config_path, config)
    training_cfg = dict(cfg.get("training", {}))
    seed = int(cfg.get("seed", training_cfg.get("seed", 0)))
    _set_seed(seed)

    dataset = SupervisedDataset(cfg.get("dataset"))
    batch_size = int(training_cfg.get("batch_size", 8))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    model = PolicyValueModel(cfg.get("model"))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training_cfg.get("learning_rate", 1e-2))
    )
    epochs = int(training_cfg.get("epochs", 1))
    log_interval = max(1, int(training_cfg.get("log_interval", 1)))

    total_loss = 0.0
    total_batches = 0

    for epoch in range(epochs):
        for batch_index, batch in enumerate(loader, start=1):
            optimizer.zero_grad()
            outputs = model(batch["inputs"])
            policy_loss = F.cross_entropy(outputs["policy_logits"], batch["action"])
            value_loss = F.mse_loss(outputs["value"], batch["value_target"])
            guard_loss = F.binary_cross_entropy_with_logits(
                outputs["guard_logits"], batch["guard_label"]
            )
            latency_loss = F.mse_loss(outputs["latency"], batch["latency_target"])
            loss = policy_loss + value_loss + guard_loss + latency_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            if batch_index % log_interval == 0:
                _LOGGER.info(
                    "epoch=%d batch=%d loss=%.4f policy=%.4f value=%.4f guard=%.4f latency=%.4f",
                    epoch,
                    batch_index,
                    loss.item(),
                    policy_loss.item(),
                    value_loss.item(),
                    guard_loss.item(),
                    latency_loss.item(),
                )

    checkpoint_dir = Path(
        training_cfg.get("checkpoint_dir", "data/processed/checkpoints/supervised")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "supervised.pt"
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
    _LOGGER.info("training-complete epochs=%d mean_loss=%.4f", epochs, mean_loss)
    return {
        "epochs": epochs,
        "mean_loss": mean_loss,
        "checkpoint": str(checkpoint_path),
    }
