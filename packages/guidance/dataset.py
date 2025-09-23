"""Deterministic in-memory datasets powering the test training loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset

__all__ = [
    "SupervisedDataset",
    "SupervisedDatasetConfig",
    "TrajectoryDataset",
    "TrajectoryDatasetConfig",
    "PreferenceDataset",
    "PreferenceDatasetConfig",
]


@dataclass(frozen=True)
class SupervisedDatasetConfig:
    """Configuration describing the synthetic supervised dataset."""

    num_samples: int = 64
    input_dim: int = 32
    action_dim: int = 8
    seed: int = 7_432

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "SupervisedDatasetConfig":
        if data is None:
            return cls()
        return cls(
            num_samples=int(data.get("num_samples", cls.num_samples)),
            input_dim=int(data.get("input_dim", cls.input_dim)),
            action_dim=int(data.get("action_dim", cls.action_dim)),
            seed=int(data.get("seed", cls.seed)),
        )

    def __post_init__(self) -> None:
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        for name in ("input_dim", "action_dim"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer")


class SupervisedDataset(Dataset):
    """Synthetic dataset with policy/value/auxiliary targets."""

    def __init__(self, config: SupervisedDatasetConfig | Mapping[str, Any] | None = None) -> None:
        if isinstance(config, Mapping) or config is None:
            dataset_config = SupervisedDatasetConfig.from_mapping(config)
        else:
            dataset_config = config
        generator = torch.Generator().manual_seed(dataset_config.seed)
        self._config = dataset_config
        self._inputs = torch.randn(
            dataset_config.num_samples, dataset_config.input_dim, generator=generator
        )
        self._actions = torch.randint(
            dataset_config.action_dim, (dataset_config.num_samples,), generator=generator
        )
        self._value_targets = torch.randn(dataset_config.num_samples, generator=generator).tanh()
        guard_raw = torch.rand(dataset_config.num_samples, generator=generator)
        self._guard_labels = (guard_raw > 0.5).float()
        self._latency_targets = torch.rand(dataset_config.num_samples, generator=generator)

    @property
    def config(self) -> SupervisedDatasetConfig:
        return self._config

    def __len__(self) -> int:
        return self._inputs.size(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError("index out of range")
        return {
            "inputs": self._inputs[index],
            "action": self._actions[index],
            "value_target": self._value_targets[index],
            "guard_label": self._guard_labels[index],
            "latency_target": self._latency_targets[index],
        }


@dataclass(frozen=True)
class TrajectoryDatasetConfig:
    """Configuration for a batch of deterministic PPO-style trajectories."""

    num_trajectories: int = 4
    trajectory_length: int = 8
    input_dim: int = 32
    action_dim: int = 8
    seed: int = 12_547

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "TrajectoryDatasetConfig":
        if data is None:
            return cls()
        return cls(
            num_trajectories=int(data.get("num_trajectories", cls.num_trajectories)),
            trajectory_length=int(data.get("trajectory_length", cls.trajectory_length)),
            input_dim=int(data.get("input_dim", cls.input_dim)),
            action_dim=int(data.get("action_dim", cls.action_dim)),
            seed=int(data.get("seed", cls.seed)),
        )

    def __post_init__(self) -> None:
        if self.num_trajectories <= 0:
            raise ValueError("num_trajectories must be positive")
        if self.trajectory_length <= 0:
            raise ValueError("trajectory_length must be positive")
        for name in ("input_dim", "action_dim"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer")


class TrajectoryDataset(Dataset):
    """Offline trajectory buffer used for deterministic PPO updates."""

    def __init__(self, config: TrajectoryDatasetConfig | Mapping[str, Any] | None = None) -> None:
        if isinstance(config, Mapping) or config is None:
            dataset_config = TrajectoryDatasetConfig.from_mapping(config)
        else:
            dataset_config = config
        generator = torch.Generator().manual_seed(dataset_config.seed)
        total_steps = dataset_config.num_trajectories * dataset_config.trajectory_length
        self._config = dataset_config
        self._states = torch.randn(total_steps, dataset_config.input_dim, generator=generator)
        self._actions = torch.randint(
            dataset_config.action_dim, (total_steps,), generator=generator
        )
        self._log_probs = torch.randn(total_steps, generator=generator) * 0.1
        self._advantages = torch.randn(total_steps, generator=generator)
        rewards = torch.sigmoid(torch.randn(total_steps, generator=generator))
        self._returns = rewards + self._advantages * 0.1
        dones = torch.zeros(total_steps, dtype=torch.bool)
        for t in range(dataset_config.num_trajectories):
            dones[(t + 1) * dataset_config.trajectory_length - 1] = True
        self._dones = dones

    @property
    def config(self) -> TrajectoryDatasetConfig:
        return self._config

    def __len__(self) -> int:
        return self._states.size(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError("index out of range")
        return {
            "state": self._states[index],
            "action": self._actions[index],
            "log_prob": self._log_probs[index],
            "advantage": self._advantages[index],
            "returns": self._returns[index],
            "done": self._dones[index],
        }


@dataclass(frozen=True)
class PreferenceDatasetConfig:
    """Configuration for the synthetic preference dataset."""

    num_samples: int = 24
    input_dim: int = 32
    seed: int = 23_991

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "PreferenceDatasetConfig":
        if data is None:
            return cls()
        return cls(
            num_samples=int(data.get("num_samples", cls.num_samples)),
            input_dim=int(data.get("input_dim", cls.input_dim)),
            seed=int(data.get("seed", cls.seed)),
        )

    def __post_init__(self) -> None:
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if self.input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")


class PreferenceDataset(Dataset):
    """Pairs of responses with a deterministic preferred candidate."""

    def __init__(self, config: PreferenceDatasetConfig | Mapping[str, Any] | None = None) -> None:
        if isinstance(config, Mapping) or config is None:
            dataset_config = PreferenceDatasetConfig.from_mapping(config)
        else:
            dataset_config = config
        generator = torch.Generator().manual_seed(dataset_config.seed)
        self._config = dataset_config
        prompts = torch.randn(
            dataset_config.num_samples, dataset_config.input_dim, generator=generator
        )
        deltas = (
            torch.randn(
                dataset_config.num_samples, 2, dataset_config.input_dim, generator=generator
            )
            * 0.1
        )
        self._prompts = prompts
        # Encourage the first candidate to be marginally better than the second.
        self._chosen = prompts + deltas[:, 0]
        self._rejected = prompts + deltas[:, 1] - 0.05
        self._rewards = torch.full((dataset_config.num_samples,), 1.0)

    @property
    def config(self) -> PreferenceDatasetConfig:
        return self._config

    def __len__(self) -> int:
        return self._prompts.size(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError("index out of range")
        return {
            "prompt": self._prompts[index],
            "chosen": self._chosen[index],
            "rejected": self._rejected[index],
            "label": self._rewards[index],
        }
