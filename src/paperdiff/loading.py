from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage import io

__all__ = ["load_image"]


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk and normalize it to float32 in the 0..1 range."""
    image = io.imread(path)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]

    image = np.asarray(image, dtype=np.float32)
    if image.dtype.kind in {"u", "i"}:
        max_value = np.iinfo(image.dtype).max
        if max_value > 1:
            image = image / float(max_value)
    elif image.max() > 1.0:
        image = image / 255.0

    return image.astype(np.float32)
