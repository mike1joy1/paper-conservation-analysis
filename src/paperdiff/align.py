from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["AlignResult", "AlignmentError", "align"]


class AlignmentError(RuntimeError):
    """Compatibility exception retained for API stability."""


@dataclass
class AlignResult:
    aligned: np.ndarray
    valid_mask: np.ndarray
    warp: np.ndarray
    score: float


def align(template: np.ndarray, moving: np.ndarray) -> AlignResult:
    """Compatibility shim: the app assumes alignment is already complete before analysis."""
    template_arr = np.asarray(template, dtype=np.float32)
    moving_arr = np.asarray(moving, dtype=np.float32)
    if template_arr.shape != moving_arr.shape:
        raise ValueError(f"Template and moving images must match: {template_arr.shape} != {moving_arr.shape}")

    valid_mask = np.ones(template_arr.shape[:2], dtype=bool)
    return AlignResult(
        aligned=moving_arr.copy(),
        valid_mask=valid_mask,
        warp=np.eye(3, dtype=np.float32),
        score=1.0,
    )
