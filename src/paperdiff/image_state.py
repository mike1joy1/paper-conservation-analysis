from __future__ import annotations

import numpy as np
from PySide6.QtGui import QColor
from scipy.ndimage import binary_erosion

from .loading import load_image

__all__ = ["analysis_options", "image_data_colour"]


class analysis_options:
    """Container for all analysis option settings used by the UI."""

    def __init__(self):
        self.rgb_comparison = True
        self.lab_comparison = True
        self.use_masks = True
        self.intersection_mask = True
        self.images_aligned = False
        self.mean_comparison = True
        self.per_pixel_comparison = False
        self.heatmap_grid_size = 50

        self.int_paper_mask = None
        self.int_ink_mask = None

    def __str__(self) -> str:
        details = {
            "rgb_comparison": self.rgb_comparison,
            "lab_comparison": self.lab_comparison,
            "use_masks": self.use_masks,
            "intersection_mask": self.intersection_mask,
            "images_aligned": self.images_aligned,
            "mean_comparison": self.mean_comparison,
            "per_pixel_comparison": self.per_pixel_comparison,
            "heatmap_grid_size": self.heatmap_grid_size,
        }
        return f"analysis_options({details})"

    def sync_from_ui(self, **values):
        for key, value in values.items():
            if hasattr(self, key):
                setattr(self, key, value)


class image_data_colour:
    """Container for a single image and the related pixel and mask state."""

    def __init__(self, key: str, file_path: str | None = None):
        self.key = key
        self.file_path = file_path
        self.pixels: dict[str, np.ndarray] = {}
        self.masks: dict[str, np.ndarray] = {}
        self.mask_border: np.ndarray | None = None
        self.masks_exists: dict[str, bool] = {"paper": False, "ink": False}
        self.reference_colours = {
            "paper": QColor("white"),
            "ink": QColor("black"),
        }
        self._reset_masks_to_full()

    def __str__(self) -> str:
        pixels_summary = {name: np.asarray(value).shape if value is not None else None for name, value in self.pixels.items()}
        masks_summary = {name: np.asarray(value).shape if value is not None else None for name, value in self.masks.items()}
        details = {
            "key": self.key,
            "file_path": self.file_path,
            "pixels": pixels_summary,
            "masks": masks_summary,
            "mask_border": None if self.mask_border is None else np.asarray(self.mask_border).shape,
            "masks_exists": self.masks_exists,
            "reference_colours": {name: colour.name() for name, colour in self.reference_colours.items()},
        }
        return f"image_data_colour({details})"

    @staticmethod
    def _mask_to_rgb(mask: np.ndarray | None) -> np.ndarray | None:
        if mask is None:
            return None
        boolean_mask = np.asarray(mask, dtype=bool)
        if boolean_mask.ndim == 2:
            return np.stack([boolean_mask.astype(np.float32)] * 3, axis=-1)
        if boolean_mask.ndim == 3 and boolean_mask.shape[-1] == 1:
            return np.repeat(boolean_mask.astype(np.float32), 3, axis=-1)
        return boolean_mask.astype(np.float32)

    def _reset_masks_to_full(self):
        if "rgb" in self.pixels:
            shape = np.asarray(self.pixels["rgb"]).shape[:2]
        else:
            shape = (0, 0)

        if shape[0] == 0 or shape[1] == 0:
            self.masks = {"paper": None, "ink": None}
            self.mask_border = np.zeros((0, 0), dtype=bool)
            self.masks_exists = {"paper": False, "ink": False}
            return

        full_mask = np.ones(shape, dtype=bool)
        self.masks = {
            "paper": self._mask_to_rgb(full_mask),
            "ink": self._mask_to_rgb(full_mask),
        }
        self.mask_border = self._mask_to_rgb(np.zeros(shape, dtype=bool))
        self.masks_exists = {"paper": False, "ink": False}

    def clear_masks(self):
        self._reset_masks_to_full()

    def get_pixels(self, colourspace: str):
        if colourspace in self.pixels:
            return np.asarray(self.pixels[colourspace], dtype=np.float32)
        if colourspace == "lab":
            return self.generate_lab_pixels()
        raise KeyError(f"Unknown colourspace '{colourspace}'")

    def set_mask(self, name: str, mask: np.ndarray):
        if name not in {"paper", "ink"}:
            raise ValueError(f"Unsupported mask '{name}'")
        array = self._mask_to_rgb(mask)
        self.masks[name] = array
        self.masks_exists[name] = True

    def load_from_path(self, path: str):
        self.file_path = path
        image = load_image(path)
        rgb = np.asarray(image, dtype=np.float32)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=-1)
        if rgb.shape[-1] == 1:
            rgb = np.repeat(rgb, 3, axis=-1)
        if rgb.size > 0 and rgb.max() > 1.0:
            rgb = rgb / 255.0
        self.pixels["rgb"] = np.clip(rgb, 0.0, 1.0)
        self.generate_lab_pixels()
        self._reset_masks_to_full()
        return self.pixels["rgb"]

    def generate_lab_pixels(self):
        if "rgb" not in self.pixels:
            return None

        rgb = np.asarray(self.pixels["rgb"], dtype=np.float32)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=-1)
        if rgb.shape[-1] == 1:
            rgb = np.repeat(rgb, 3, axis=-1)
        if rgb.size == 0:
            self.pixels["lab"] = rgb.copy()
            return self.pixels["lab"]

        rgb = np.clip(rgb, 0.0, 1.0)
        rgb_linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        r, g, b = rgb_linear[..., 0], rgb_linear[..., 1], rgb_linear[..., 2]
        x = 0.4124 * r + 0.3576 * g + 0.1805 * b
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        z = 0.0193 * r + 0.1192 * g + 0.9505 * b

        white_ref = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
        xyz = np.stack([x, y, z], axis=-1)
        xyz = xyz / white_ref
        f_xyz = np.where(xyz > 0.008856, xyz ** (1.0 / 3.0), (7.787 * xyz) + (16.0 / 116.0))

        l_channel = 116.0 * f_xyz[..., 1] - 16.0
        a_channel = 500.0 * (f_xyz[..., 0] - f_xyz[..., 1])
        b_channel = 200.0 * (f_xyz[..., 1] - f_xyz[..., 2])
        lab = np.stack([l_channel, a_channel, b_channel], axis=-1)
        self.pixels["lab"] = lab
        return lab

    def generate_masks(self, mode: str, threshold: int = 140, reference_weight: float = 0.5, border_pixels: int = 2):
        rgb_array = self.get_pixels("rgb")
        if rgb_array is None:
            return None, None, None

        if mode == "luminance":
            brightness = np.mean(rgb_array, axis=-1)
            threshold_value = threshold / 255.0
            ink_mask = brightness <= threshold_value
        else:
            paper_colour = np.array(self.reference_colours["paper"].getRgb()[:3], dtype=np.float32) / 255.0
            ink_colour = np.array(self.reference_colours["ink"].getRgb()[:3], dtype=np.float32) / 255.0
            paper_distance = np.linalg.norm(rgb_array - paper_colour, axis=-1)
            ink_distance = np.linalg.norm(rgb_array - ink_colour, axis=-1)
            max_distance = np.sqrt(3.0)
            paper_weight = float(reference_weight)
            adjusted_paper_distance = paper_weight * paper_distance + (1.0 - paper_weight) * max_distance
            adjusted_ink_distance = (1.0 - paper_weight) * ink_distance + paper_weight * max_distance
            ink_mask = adjusted_paper_distance > adjusted_ink_distance

        original_ink_mask = np.asarray(ink_mask, dtype=bool).copy()
        original_paper_mask = (~original_ink_mask).astype(bool)

        if border_pixels > 0:
            paper_mask = binary_erosion(original_paper_mask, iterations=border_pixels)
            ink_mask = binary_erosion(original_ink_mask, iterations=border_pixels)
            border_mask = (original_ink_mask ^ binary_erosion(original_ink_mask, iterations=1)) | (
                original_paper_mask ^ binary_erosion(original_paper_mask, iterations=1)
            )
        else:
            paper_mask = original_paper_mask
            ink_mask = original_ink_mask
            border_mask = np.zeros_like(original_ink_mask, dtype=bool)

        self.masks["paper"] = self._mask_to_rgb(np.asarray(paper_mask, dtype=bool))
        self.masks["ink"] = self._mask_to_rgb(np.asarray(ink_mask, dtype=bool))
        self.mask_border = self._mask_to_rgb(np.asarray(border_mask, dtype=bool))
        self.masks_exists["paper"] = True
        self.masks_exists["ink"] = True
        return self.masks["paper"], self.masks["ink"], self.mask_border
