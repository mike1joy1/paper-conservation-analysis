from __future__ import annotations

import numpy as np
from skimage.color import rgb2gray

__all__ = ["make_paper_ink_masks"]


def make_paper_ink_masks(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        gray = array
    else:
        gray = rgb2gray(np.clip(array, 0.0, 1.0))

    paper_mask = (gray > 0.08) & (gray < 0.92)
    ink_mask = gray < 0.42
    return paper_mask.astype(bool), ink_mask.astype(bool)
