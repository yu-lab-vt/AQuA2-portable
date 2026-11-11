"""Minimal scientific result contract, independent of viewer state."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class StageResult:
    stage: str
    shape_hwlt: tuple[int, int, int, int]
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.shape_hwlt) != 4 or any(
            isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0
            for v in self.shape_hwlt
        ):
            raise ValueError("shape_hwlt must contain four positive integers")
