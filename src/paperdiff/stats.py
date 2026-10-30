from __future__ import annotations

import numpy as np

__all__ = ["compute_region_metrics", "verdict_for_region"]


def compute_region_metrics(delta_map: np.ndarray, paper_mask: np.ndarray, ink_mask: np.ndarray) -> dict[str, dict[str, float]]:
    paper_values = np.asarray(delta_map[paper_mask], dtype=np.float32)
    ink_values = np.asarray(delta_map[ink_mask], dtype=np.float32)

    def _summarize(values: np.ndarray) -> dict[str, float]:
        if values.size == 0:
            return {"mean": 0.0, "std": 0.0, "max": 0.0, "median": 0.0, "n_pixels": 0.0}
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "n_pixels": float(values.size),
        }

    return {"paper": _summarize(paper_values), "ink": _summarize(ink_values)}


def verdict_for_region(metrics: dict[str, float]) -> str:
    mean = metrics.get("mean", 0.0)
    if mean < 2.0:
        return "No material change detected"
    if mean < 5.0:
        return "Minor change"
    if mean < 10.0:
        return "Moderate change"
    return "Major change"
