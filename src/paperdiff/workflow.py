"""Simple analysis workflow for already-aligned paper images.

This module is intentionally small and explicit. The research assumption is that
B1, B2 and A have already been aligned and cropped before they reach this app.
The code therefore focuses only on measuring the visual difference between the
reference image and the comparison images.
"""

from __future__ import annotations

import numpy as np

from .analysis import compute_deltas, compute_ink_mask

__all__ = ["evaluate_reference_pair", "describe_region_change"]


def _as_rgb(image: np.ndarray) -> np.ndarray:
    """Convert any image-like input to a float RGB array in the 0..1 range."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return array.astype(np.float32)


def _summarize_metric_map(values: np.ndarray) -> dict[str, float]:
    """Return the raw statistical summary for a region. This is kept visible so
    the output remains verifiable and audit-friendly.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "median": 0.0, "n_pixels": 0.0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "n_pixels": float(values.size),
    }


def describe_region_change(metrics: dict[str, float], region_name: str = "region") -> str:
    """Translate the numerical metrics into plain-language conservation notes.

    The app reports both the science metric (ΔE00) and the underlying Lab-direction
    components: lightness (L*) and yellow-blue shift (b*). This makes the result
    understandable to a new user without requiring them to decode the raw arrays.
    """
    mean_delta = float(metrics.get("mean", 0.0))
    dL = float(metrics.get("dL_median", metrics.get("dL_mean", 0.0)))
    db = float(metrics.get("db_median", metrics.get("db_mean", 0.0)))

    brightness = "unchanged"
    if dL > 0.5:
        brightness = "lighter"
    elif dL < -0.5:
        brightness = "darker"

    yellowing = "unchanged"
    if db > 0.5:
        yellowing = "more yellow"
    elif db < -0.5:
        yellowing = "more blue"

    return (
        f"{region_name.title()}: mean colour change {mean_delta:.2f} ΔE00; "
        f"brightness change {dL:.2f} L* ({brightness}); "
        f"yellow-blue shift {db:.2f} b* ({yellowing})."
    )


def evaluate_reference_pair(b1: np.ndarray, b2: np.ndarray, a: np.ndarray) -> dict[str, dict[str, object]]:
    """Compare B1 with B2 and A using the already-aligned input images.

    The workflow is intentionally simple:
    1. Build a paper/ink mask from B1.
    2. Compute per-pixel colour deltas between B1 and each comparison image.
    3. Summarise the change in the paper and ink regions.
    4. Return the raw numbers and a simple score for each comparison.
    """
    ref = _as_rgb(b1)
    base_img = _as_rgb(b2)
    test_img = _as_rgb(a)

    if ref.shape != base_img.shape or ref.shape != test_img.shape:
        raise ValueError("B1, B2, and A must all have matching widths, heights, and channels.")

    # Build the mask once from the reference image and reuse it for both comparisons.
    ink_mask = compute_ink_mask(ref)
    paper_mask = ~ink_mask

    # B1 vs B2 is the baseline reference noise; B1 vs A is the treatment change.
    base_diff = compute_deltas(ref, base_img)
    test_diff = compute_deltas(ref, test_img)

    # Keep the raw numbers visible, but also attach the Lab-direction metrics used in
    # the human-readable report so the meaning of the result is clear.
    base_paper = _summarize_metric_map(base_diff["dE00"][paper_mask])
    base_paper["dL_median"] = float(np.median(base_diff["dL"][paper_mask])) if np.any(paper_mask) else 0.0
    base_paper["db_median"] = float(np.median(base_diff["db"][paper_mask])) if np.any(paper_mask) else 0.0
    base_ink = _summarize_metric_map(base_diff["dE00"][ink_mask])
    base_ink["dL_median"] = float(np.median(base_diff["dL"][ink_mask])) if np.any(ink_mask) else 0.0
    base_ink["db_median"] = float(np.median(base_diff["db"][ink_mask])) if np.any(ink_mask) else 0.0

    test_paper = _summarize_metric_map(test_diff["dE00"][paper_mask])
    test_paper["dL_median"] = float(np.median(test_diff["dL"][paper_mask])) if np.any(paper_mask) else 0.0
    test_paper["db_median"] = float(np.median(test_diff["db"][paper_mask])) if np.any(paper_mask) else 0.0
    test_ink = _summarize_metric_map(test_diff["dE00"][ink_mask])
    test_ink["dL_median"] = float(np.median(test_diff["dL"][ink_mask])) if np.any(ink_mask) else 0.0
    test_ink["db_median"] = float(np.median(test_diff["db"][ink_mask])) if np.any(ink_mask) else 0.0

    valid_mask = paper_mask | ink_mask
    base_score = float(np.mean(base_diff["dE00"][valid_mask])) if np.any(valid_mask) else 0.0
    test_score = float(np.mean(test_diff["dE00"][valid_mask])) if np.any(valid_mask) else 0.0

    return {
        "base": {
            "reference": ref,
            "aligned": base_img,
            "score": base_score,
            "paper": base_paper,
            "ink": base_ink,
        },
        "test": {
            "reference": ref,
            "aligned": test_img,
            "score": test_score,
            "paper": test_paper,
            "ink": test_ink,
        },
    }
