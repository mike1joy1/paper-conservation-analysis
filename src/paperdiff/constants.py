from __future__ import annotations

DEFAULT_GAUSSIAN_SIGMA = 2.0
INK_SIGMA = 1.0
MASK_ERODE_PX = 3
MIN_INK_SPECK_PX = 5
EDGE_BAND_WIDTH = 2
HALO_MAX_DISTANCE = 30
TILE_SIZE = 128
PIXEL_SIZE_MM = None

NOISE_WORDING = {
    "below_2": "slightly",
    "2_to_5": "noticeably",
    "above_5": "strongly",
}

DETECTION_THRESHOLDS = (1.0, 2.0, 5.0)
EDGE_SOFTNESS_FACTOR = 2.0
EDGE_SOFTNESS_PERCENT = 5.0
MAGNITUDE_WORDS = {"below_2": "slightly", "2_to_5": "noticeably", "above_5": "strongly"}

INK_THRESHOLD_OVERRIDE = None
FIXED_INK_THRESHOLD = None
