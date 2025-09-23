"""Model definitions scaffold."""

from __future__ import annotations

import torch.nn as nn


class PolicyValueModel(nn.Module):
    """Stub policy/value network."""

    def __init__(self) -> None:  # pragma: no cover - scaffold
        super().__init__()

    def forward(self, *args, **kwargs):  # pragma: no cover - scaffold
        raise NotImplementedError("Model scaffold.")
