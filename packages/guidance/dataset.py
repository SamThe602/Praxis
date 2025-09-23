"""Dataset helpers scaffold."""

from __future__ import annotations

from torch.utils.data import Dataset


class SupervisedDataset(Dataset):  # pragma: no cover - scaffold
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError
