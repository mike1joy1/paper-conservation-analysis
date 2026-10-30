from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage
from skimage.color import deltaE_ciede2000, rgb2gray, rgb2lab
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects

from .constants import (
    DEFAULT_GAUSSIAN_SIGMA,
    EDGE_BAND_WIDTH,
    FIXED_INK_THRESHOLD,
    HALO_MAX_DISTANCE,
    INK_SIGMA,
    INK_THRESHOLD_OVERRIDE,
    MASK_ERODE_PX,
    MIN_INK_SPECK_PX,
)

__all__ = [
    "compute_deltas",
    "compute_ink_mask",
    "evaluate_reference_pair",
    "compute_ink_area_metrics",
    "compute_edge_softness",
    "compute_halo_profile",
]


def _as_rgb_float(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return np.clip(array, 0.0, 1.0).astype(np.float32)


def _smooth_rgb(image: np.ndarray, sigma: float = DEFAULT_GAUSSIAN_SIGMA) -> np.ndarray:
    rgb = _as_rgb_float(image)
    if sigma is None or sigma <= 0:
        return rgb
    return ndimage.gaussian_filter(rgb, sigma=(sigma, sigma, 0), mode="reflect")


def compute_deltas(before: np.ndarray, after: np.ndarray, sigma: float = DEFAULT_GAUSSIAN_SIGMA) -> dict[str, np.ndarray]:
    before_rgb = _smooth_rgb(before, sigma)
    after_rgb = _smooth_rgb(after, sigma)

    before_lab = rgb2lab(before_rgb)
    after_lab = rgb2lab(after_rgb)

    dL = after_lab[..., 0] - before_lab[..., 0]
    da = after_lab[..., 1] - before_lab[..., 1]
    db = after_lab[..., 2] - before_lab[..., 2]
    dC = np.hypot(after_lab[..., 1], after_lab[..., 2]) - np.hypot(before_lab[..., 1], before_lab[..., 2])
    dE00 = deltaE_ciede2000(before_lab, after_lab)

    return {
        "dE00": np.asarray(dE00, dtype=np.float32),
        "dL": np.asarray(dL, dtype=np.float32),
        "da": np.asarray(da, dtype=np.float32),
        "db": np.asarray(db, dtype=np.float32),
        "dC": np.asarray(dC, dtype=np.float32),
    }


def _threshold_for_mask(luma: np.ndarray, threshold: float | None = None, use_otsu: bool = False) -> float:
    if threshold is not None:
        return float(threshold)
    if use_otsu:
        return float(threshold_otsu(luma))
    if FIXED_INK_THRESHOLD is not None:
        return float(FIXED_INK_THRESHOLD)
    return float(threshold_otsu(luma))


def compute_ink_mask(image: np.ndarray, threshold: float | None = None, sigma_ink: float = INK_SIGMA, use_otsu: bool = False) -> np.ndarray:
    rgb = _as_rgb_float(image)
    gray = rgb2gray(rgb)
    luma = ndimage.gaussian_filter(gray, sigma=sigma_ink, mode="reflect")
    cutoff = _threshold_for_mask(luma, threshold=threshold if threshold is not None else INK_THRESHOLD_OVERRIDE, use_otsu=use_otsu)
    mask = luma < cutoff
    mask = ndimage.binary_erosion(mask, iterations=MASK_ERODE_PX, border_value=0)
    mask = remove_small_objects(mask.astype(bool), min_size=MIN_INK_SPECK_PX)
    mask = remove_small_holes(mask.astype(bool), area_threshold=MIN_INK_SPECK_PX)
    return mask.astype(bool)


def _summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "n": 0.0}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "n": float(values.size),
    }


def _region_stats(deltas: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    dE = deltas["dE00"][mask]
    dL = deltas["dL"][mask]
    da = deltas["da"][mask]
    db = deltas["db"][mask]
    dC = deltas["dC"][mask]

    return {
        "mean": float(np.mean(dE)) if dE.size else 0.0,
        "std": float(np.std(dE)) if dE.size else 0.0,
        "max": float(np.max(dE)) if dE.size else 0.0,
        "median": float(np.median(dE)) if dE.size else 0.0,
        "p95": float(np.percentile(dE, 95)) if dE.size else 0.0,
        "p99": float(np.percentile(dE, 99)) if dE.size else 0.0,
        "dL_median": float(np.median(dL)) if dL.size else 0.0,
        "da_median": float(np.median(da)) if da.size else 0.0,
        "db_median": float(np.median(db)) if db.size else 0.0,
        "dC_median": float(np.median(dC)) if dC.size else 0.0,
        "pct_dE00_gt_1": 100.0 * np.mean(dE > 1.0) if dE.size else 0.0,
        "pct_dE00_gt_2": 100.0 * np.mean(dE > 2.0) if dE.size else 0.0,
        "pct_dE00_gt_5": 100.0 * np.mean(dE > 5.0) if dE.size else 0.0,
        "n_pixels": float(dE.size),
    }


def _verdict_for_region(median_dE: float, baseline_p95: float) -> str:
    if median_dE > baseline_p95:
        return "above noise"
    return "within noise"


def evaluate_reference_pair(b1: np.ndarray, b2: np.ndarray, a: np.ndarray, sigma: float = DEFAULT_GAUSSIAN_SIGMA) -> dict[str, dict[str, Any]]:
    b1_rgb = _as_rgb_float(b1)
    b2_rgb = _as_rgb_float(b2)
    a_rgb = _as_rgb_float(a)
    if b1_rgb.shape != b2_rgb.shape or b1_rgb.shape != a_rgb.shape:
        raise ValueError("B1, B2 and A must have the same shape.")

    ink_b1 = compute_ink_mask(b1_rgb, threshold=FIXED_INK_THRESHOLD, use_otsu=True)
    paper_mask = ~ink_b1

    b1_vs_b2 = compute_deltas(b1_rgb, b2_rgb, sigma=sigma)
    b1_vs_a = compute_deltas(b1_rgb, a_rgb, sigma=sigma)

    base_paper = _region_stats(b1_vs_b2, paper_mask)
    base_ink = _region_stats(b1_vs_b2, ink_b1)
    test_paper = _region_stats(b1_vs_a, paper_mask)
    test_ink = _region_stats(b1_vs_a, ink_b1)

    base_p95_paper = float(np.percentile(b1_vs_b2["dE00"][paper_mask], 95)) if np.any(paper_mask) else 0.0
    base_p95_ink = float(np.percentile(b1_vs_b2["dE00"][ink_b1], 95)) if np.any(ink_b1) else 0.0

    def _die(region: dict[str, Any], baseline_p95: float) -> dict[str, Any]:
        return {
            **region,
            "verdict": _verdict_for_region(region["median"], baseline_p95),
            "baseline_p95": baseline_p95,
            "excess": region["median"] - baseline_p95,
            "direction": {
                "dL": "lighter" if region["dL_median"] > 0 else "darker",
                "db": "yellower" if region["db_median"] > 0 else "bluer",
                "dC": "more saturated" if region["dC_median"] > 0 else "less saturated",
            },
        }

    return {
        "base": {
            "reference": b1_rgb,
            "aligned": b2_rgb,
            "score": float(np.mean(b1_vs_b2["dE00"][paper_mask | ink_b1])) if np.any(paper_mask | ink_b1) else 0.0,
            "paper": _die(base_paper, base_p95_paper),
            "ink": _die(base_ink, base_p95_ink),
        },
        "test": {
            "reference": b1_rgb,
            "aligned": a_rgb,
            "score": float(np.mean(b1_vs_a["dE00"][paper_mask | ink_b1])) if np.any(paper_mask | ink_b1) else 0.0,
            "paper": _die(test_paper, base_p95_paper),
            "ink": _die(test_ink, base_p95_ink),
        },
    }


def compute_ink_area_metrics(b1: np.ndarray, b2: np.ndarray, a: np.ndarray, threshold: float | None = None, pixel_size_mm: float | None = None, use_otsu: bool = False) -> dict[str, Any]:
    b1_rgb = _as_rgb_float(b1)
    b2_rgb = _as_rgb_float(b2)
    a_rgb = _as_rgb_float(a)

    b1_mask = compute_ink_mask(b1_rgb, threshold=threshold, use_otsu=use_otsu)
    b2_mask = compute_ink_mask(b2_rgb, threshold=threshold, use_otsu=use_otsu)
    a_mask = compute_ink_mask(a_rgb, threshold=threshold, use_otsu=use_otsu)

    def _area(mask: np.ndarray) -> float:
        return float(np.count_nonzero(mask))

    def _mm(area_px: float) -> float:
        if pixel_size_mm is None:
            return float(area_px)
        return float(area_px * (pixel_size_mm ** 2))

    b1_area = _area(b1_mask)
    def _pct_change(current: float, ref: float) -> float:
        return ((current - ref) / ref) * 100.0 if ref else 0.0

    new_b2 = np.logical_and(b2_mask, ~ndimage.binary_dilation(b1_mask, iterations=1))
    lost_b2 = np.logical_and(b1_mask, ~ndimage.binary_dilation(b2_mask, iterations=1))
    new_a = np.logical_and(a_mask, ~ndimage.binary_dilation(b1_mask, iterations=1))
    lost_a = np.logical_and(b1_mask, ~ndimage.binary_dilation(a_mask, iterations=1))

    return {
        "B1": {"area_px": b1_area, "area_mm2": _mm(b1_area), "pct_change_vs_B1": 0.0},
        "B2": {
            "area_px": _area(b2_mask),
            "area_mm2": _mm(_area(b2_mask)),
            "pct_change_vs_B1": _pct_change(_area(b2_mask), b1_area),
            "new_ink_pct_B1": 100.0 * np.count_nonzero(new_b2) / b1_area if b1_area else 0.0,
            "lost_ink_pct_B1": 100.0 * np.count_nonzero(lost_b2) / b1_area if b1_area else 0.0,
        },
        "A": {
            "area_px": _area(a_mask),
            "area_mm2": _mm(_area(a_mask)),
            "pct_change_vs_B1": _pct_change(_area(a_mask), b1_area),
            "new_ink_pct_B1": 100.0 * np.count_nonzero(new_a) / b1_area if b1_area else 0.0,
            "lost_ink_pct_B1": 100.0 * np.count_nonzero(lost_a) / b1_area if b1_area else 0.0,
        },
        "fixed_threshold": {"B1": b1_mask, "B2": b2_mask, "A": a_mask},
    }


def compute_edge_softness(b1: np.ndarray, b2: np.ndarray, a: np.ndarray, threshold: float | None = None, pixel_size_mm: float | None = None) -> dict[str, Any]:
    b1_rgb = _as_rgb_float(b1)
    b2_rgb = _as_rgb_float(b2)
    a_rgb = _as_rgb_float(a)

    def _measure(image: np.ndarray) -> dict[str, float]:
        ink = compute_ink_mask(image, threshold=threshold, use_otsu=True)
        lab = rgb2lab(_as_rgb_float(image))
        L = lab[..., 0]
        paper = ~ink
        if not np.any(ink) or not np.any(paper):
            return {"edge_sharpness": 0.0, "contrast": 0.0, "width_px": 0.0, "width_mm": 0.0}
        boundary = ndimage.binary_dilation(ink, iterations=1) & ~ink
        band = np.zeros_like(ink, dtype=bool)
        for offset in range(-EDGE_BAND_WIDTH, EDGE_BAND_WIDTH + 1):
            band |= ndimage.shift(boundary, shift=(offset, 0), order=0, mode="nearest")
        grad = ndimage.gaussian_gradient_magnitude(L, sigma=1.0)
        sharpness = float(np.mean(grad[band])) if np.any(band) else 0.0
        contrast = float(np.median(L[paper]) - np.median(L[ink]))
        width_px = contrast / sharpness if sharpness > 0 else 0.0
        width_mm = width_px * pixel_size_mm if pixel_size_mm is not None else width_px
        return {"edge_sharpness": sharpness, "contrast": contrast, "width_px": width_px, "width_mm": width_mm}

    b1_measure = _measure(b1_rgb)
    b2_measure = _measure(b2_rgb)
    a_measure = _measure(a_rgb)
    return {"B1": b1_measure, "B2": b2_measure, "A": a_measure}


def compute_halo_profile(before: np.ndarray, after: np.ndarray, threshold: float | None = None, sigma: float = DEFAULT_GAUSSIAN_SIGMA, pixel_size_mm: float | None = None) -> dict[str, Any]:
    before_rgb = _as_rgb_float(before)
    after_rgb = _as_rgb_float(after)
    before_ink = compute_ink_mask(before_rgb, threshold=threshold, use_otsu=True)
    deltas = compute_deltas(before_rgb, after_rgb, sigma=sigma)
    paper_mask = ~before_ink
    dist = ndimage.distance_transform_edt(paper_mask)

    rows: list[dict[str, Any]] = []
    baseline_95_by_ring = []
    for d in range(HALO_MAX_DISTANCE + 1):
        ring = (dist >= d) & (dist < d + 1) & paper_mask
        if not np.any(ring):
            continue
        values = deltas["dE00"][ring]
        rows.append({
            "distance": d,
            "pixels": int(np.count_nonzero(ring)),
            "median_dE00": float(np.median(values)) if values.size else 0.0,
            "median_dL": float(np.median(deltas["dL"][ring])) if values.size else 0.0,
            "median_db": float(np.median(deltas["db"][ring])) if values.size else 0.0,
            "median_dC": float(np.median(deltas["dC"][ring])) if values.size else 0.0,
            "distance_mm": float(d * pixel_size_mm) if pixel_size_mm is not None else float(d),
        })
        baseline_95_by_ring.append(float(np.percentile(values, 95)))

    halo_extent = 0
    for idx, row in enumerate(rows):
        if row["median_dE00"] <= (baseline_95_by_ring[idx] if idx < len(baseline_95_by_ring) else 0.0):
            halo_extent = max(0, idx - 1)
            break
    else:
        halo_extent = rows[-1]["distance"] if rows else 0

    return {
        "rings": rows,
        "bleed_extent_px": halo_extent,
        "bleed_extent_mm": float(halo_extent * pixel_size_mm) if pixel_size_mm is not None else float(halo_extent),
        "halo_strength": float(np.mean(deltas["dL"][dist <= 5])) if np.any(dist <= 5) else 0.0,
    }
