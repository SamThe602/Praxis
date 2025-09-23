"""Deterministic lightweight neural models used by training loops.

The production system relies on considerably larger architectures, however the
unit tests exercise CPU-only training loops that must finish within seconds.
The models below therefore keep the parameter count intentionally small while
still exposing the same heads as their production counterparts:

``PolicyValueModel``
    Joint policy/value/auxiliary head used for supervised imitation and PPO.

``RewardModel``
    Preference-learning reward head that scores a prompt/response embedding.

Both modules accept an optional seed to guarantee deterministic parameter
initialisation which keeps the tiny training runs reproducible across test
invocations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn

__all__ = [
    "ModelConfig",
    "PreferenceModelConfig",
    "PolicyValueModel",
    "RewardModel",
]


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for :class:`PolicyValueModel`.

    The defaults mirror the dimensions used in the training configs and favour
    CPU friendliness.  Validation happens eagerly to surface misconfigurations
    in the unit tests rather than failing deep inside torch.
    """

    input_dim: int = 32
    hidden_dim: int = 64
    action_dim: int = 8
    seed: int | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ModelConfig":
        if data is None:
            return cls()
        return cls(
            input_dim=int(data.get("input_dim", cls.input_dim)),
            hidden_dim=int(data.get("hidden_dim", cls.hidden_dim)),
            action_dim=int(data.get("action_dim", cls.action_dim)),
            seed=data.get("seed"),
        )

    def __post_init__(self) -> None:
        for field_name in ("input_dim", "hidden_dim", "action_dim"):
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")


@dataclass(frozen=True)
class PreferenceModelConfig:
    """Configuration for :class:`RewardModel`."""

    input_dim: int = 32
    hidden_dim: int = 64
    seed: int | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "PreferenceModelConfig":
        if data is None:
            return cls()
        return cls(
            input_dim=int(data.get("input_dim", cls.input_dim)),
            hidden_dim=int(data.get("hidden_dim", cls.hidden_dim)),
            seed=data.get("seed"),
        )

    def __post_init__(self) -> None:
        for field_name in ("input_dim", "hidden_dim"):
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")


class PolicyValueModel(nn.Module):
    """Compact policy/value network with auxiliary heads."""

    def __init__(self, config: ModelConfig | Mapping[str, Any] | None = None) -> None:
        if isinstance(config, Mapping) or config is None:
            model_config = ModelConfig.from_mapping(config)
        else:
            model_config = config
        if model_config.seed is not None:
            torch.manual_seed(int(model_config.seed))
        super().__init__()
        self._config = model_config
        self.encoder = nn.Sequential(
            nn.Linear(model_config.input_dim, model_config.hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(model_config.hidden_dim, model_config.action_dim)
        self.value_head = nn.Linear(model_config.hidden_dim, 1)
        self.guard_head = nn.Linear(model_config.hidden_dim, 1)
        self.latency_head = nn.Linear(model_config.hidden_dim, 1)

    @property
    def config(self) -> ModelConfig:
        return self._config

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        features = self.encoder(inputs)
        policy_logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        guard_logits = self.guard_head(features).squeeze(-1)
        latency = self.latency_head(features).squeeze(-1)
        return {
            "policy_logits": policy_logits,
            "value": value,
            "guard_logits": guard_logits,
            "latency": latency,
        }


class RewardModel(nn.Module):
    """Tiny feed-forward reward head for preference optimisation."""

    def __init__(self, config: PreferenceModelConfig | Mapping[str, Any] | None = None) -> None:
        if isinstance(config, Mapping) or config is None:
            pref_config = PreferenceModelConfig.from_mapping(config)
        else:
            pref_config = config
        if pref_config.seed is not None:
            torch.manual_seed(int(pref_config.seed))
        super().__init__()
        self._config = pref_config
        self.network = nn.Sequential(
            nn.Linear(pref_config.input_dim, pref_config.hidden_dim),
            nn.ReLU(),
            nn.Linear(pref_config.hidden_dim, 1),
        )

    @property
    def config(self) -> PreferenceModelConfig:
        return self._config

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.network(inputs)
        return output.squeeze(-1)
