"""Common orchestrator type aliases (scaffold)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SolutionReport:
    """Placeholder solution report."""
    status: str = "pending"
