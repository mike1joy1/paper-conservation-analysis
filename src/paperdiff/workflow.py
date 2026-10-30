from __future__ import annotations

import numpy as np

from .analysis import compute_deltas, compute_ink_mask

__all__ = ["evaluate_reference_pair", "register_image_to_reference", "describe_region_change"]


def _as_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return array.astype(np.float32)


def register_image_to_reference(reference: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Compatibility shim for already-aligned images. The app assumes alignment has already been done externally."""
    ref = _as_rgb(reference)
    mov = _as_rgb(moving)
    if ref.shape != mov.shape:
        raise ValueError(f"Reference and moving images must have matching shapes: {ref.shape} != {mov.shape}")
    return mov.copy(), {"transform": np.eye(3, dtype=np.float32), "transform_type": "identity", "score": 1.0}


def _summarize_metric_map(values: np.ndarray) -> dict[str, float]:
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
    """Evaluate the difference between B1 and each comparison image. Images are expected to be already aligned and cropped."""
    ref = _as_rgb(b1)
    base_img = _as_rgb(b2)
    test_img = _as_rgb(a)

    if ref.shape != base_img.shape or ref.shape != test_img.shape:
        raise ValueError("B1, B2, and A must all have matching widths, heights, and channels.")

    ink_mask = compute_ink_mask(ref)
    paper_mask = ~ink_mask
    base_diff = compute_deltas(ref, base_img)
    test_diff = compute_deltas(ref, test_img)

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
