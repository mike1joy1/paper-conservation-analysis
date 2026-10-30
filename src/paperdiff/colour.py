from __future__ import annotations

import numpy as np
from skimage.color import rgb2lab

__all__ = ["rgb_to_lab", "compute_colour_deltas"]


def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return rgb2lab(np.clip(array, 0.0, 1.0))


def compute_colour_deltas(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float32)
    cand = np.asarray(candidate, dtype=np.float32)
    if ref.shape != cand.shape:
        raise ValueError(f"Image shapes do not match: {ref.shape} vs {cand.shape}")

    ref_lab = rgb_to_lab(ref)
    cand_lab = rgb_to_lab(cand)
    delta = np.sqrt(
        (ref_lab[..., 0] - cand_lab[..., 0]) ** 2
        + (ref_lab[..., 1] - cand_lab[..., 1]) ** 2
        + (ref_lab[..., 2] - cand_lab[..., 2]) ** 2
    )
    return delta.astype(np.float32)
