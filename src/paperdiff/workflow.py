"""Simple analysis workflow for already-aligned paper images.

This module is intentionally small and explicit. The research assumption is that
B1, B2 and A have already been aligned and cropped before they reach this app.
The code therefore focuses only on measuring the visual difference between the
reference image and the comparison images.
"""

from __future__ import annotations

import numpy as np

from .analysis import compute_deltas, compute_ink_mask
from .image_state import analysis_options, image_data_colour

__all__ = ["compare_images", "describe_region_change"]


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


def compare_images(b1: image_data_colour, b2: image_data_colour, a: image_data_colour, op: analysis_options) -> dict:
    """Compare B1 with B2 and A using the already-aligned input images.

    The workflow:
    1. Build a paper/ink mask from B1.
    2. Compute per-pixel colour deltas between B1 and each comparison image.
    3. Summarise the change in the paper and ink regions.
    4. Return the raw numbers and a simple score for each comparison.
    """
    
    # Get masks
    individual_paper_mask = (not op.intersection_mask) or (op.int_paper_mask is None) or (not op.images_aligned)
    individual_ink_mask = (not op.intersection_mask) or (op.int_ink_mask is None) or (not op.images_aligned)
    
    if individual_paper_mask:
        paper_mask = {
            "b1": b1.masks["paper"],
            "b2": b2.masks["paper"],
            "a": a.masks["paper"]
        }
    else:
        paper_mask = {
                "b1": op.int_paper_mask,
                "b2": op.int_paper_mask,
                "a": op.int_paper_mask
        }

    if individual_ink_mask:
        ink_mask = {
            "b1": b1.masks["ink"],
            "b2": b2.masks["ink"],
            "a": a.masks["ink"]
        }
    else:
        ink_mask = {
                "b1": op.int_ink_mask,
                "b2": op.int_ink_mask,
                "a": op.int_ink_mask
        }
    
    results = dict()
    
    # RGB comparisons
    if op.rgb_comparison:
        rgb = {
            "b1": b1.get_pixels("rgb"),
            "b2": b2.get_pixels("rgb"),
            "a": a.get_pixels("rgb")
        }
        
        # mean comparison
        if op.mean_comparison:
            
            rgb_paper_avg = {
                "b1": np.sum(rgb["b1"]*paper_mask["b1"], axis=(0, 1))/np.sum(paper_mask["b1"], axis=(0, 1)),
                "b2": np.sum(rgb["b2"]*paper_mask["b2"], axis=(0, 1))/np.sum(paper_mask["b2"], axis=(0, 1)),
                "a": np.sum(rgb["a"]*paper_mask["a"], axis=(0, 1))/np.sum(paper_mask["a"], axis=(0, 1))
            }

            rgb_paper_avg["b_avg"] = (rgb_paper_avg["b1"] + rgb_paper_avg["b2"]) / 2.0
            rgb_paper_avg["b_diff"] = np.abs(rgb_paper_avg["b1"] - rgb_paper_avg["b2"])

            rgb_ink_avg = {
                "b1": np.sum(rgb["b1"]*ink_mask["b1"], axis=(0, 1))/np.sum(ink_mask["b1"], axis=(0, 1)),
                "b2": np.sum(rgb["b2"]*ink_mask["b2"], axis=(0, 1))/np.sum(ink_mask["b2"], axis=(0, 1)),
                "a": np.sum(rgb["a"]*ink_mask["a"], axis=(0, 1))/np.sum(ink_mask["a"], axis=(0, 1))
            }

            rgb_ink_avg["b_avg"] = (rgb_ink_avg["b1"] + rgb_ink_avg["b2"]) / 2.0
            rgb_ink_avg["b_diff"] = np.abs(rgb_ink_avg["b1"] - rgb_ink_avg["b2"])

            rgb_raw_results = {
                "b1": {
                    "paper": {
                        "r": rgb_paper_avg["b1"][0],
                        "g": rgb_paper_avg["b1"][1],
                        "b": rgb_paper_avg["b1"][2]
                    },
                    "ink": {
                        "r": rgb_ink_avg["b1"][0],
                        "g": rgb_ink_avg["b1"][1],
                        "b": rgb_ink_avg["b1"][2]
                    }
                },
                "b2": {
                    "paper": {
                        "r": rgb_paper_avg["b2"][0],
                        "g": rgb_paper_avg["b2"][1],
                        "b": rgb_paper_avg["b2"][2]
                    },
                    "ink": {
                        "r": rgb_ink_avg["b2"][0],
                        "g": rgb_ink_avg["b2"][1],
                        "b": rgb_ink_avg["b2"][2]
                    }
                },
                "a": {
                    "paper": {
                        "r": rgb_paper_avg["a"][0],
                        "g": rgb_paper_avg["a"][1],
                        "b": rgb_paper_avg["a"][2]
                    },
                    "ink": {
                        "r": rgb_ink_avg["a"][0],
                        "g": rgb_ink_avg["a"][1],
                        "b": rgb_ink_avg["a"][2]
                    }
                }
            }
            
            results["RGB Average"] = {
                "Paper": {
                    "Before Average": {
                        "R": str(rgb_paper_avg["b_avg"][0])+" +-"+str(rgb_paper_avg["b_diff"][0]/2.0),
                        "G": str(rgb_paper_avg["b_avg"][1])+" +-"+str(rgb_paper_avg["b_diff"][1]/2.0),
                        "B": str(rgb_paper_avg["b_avg"][2])+" +-"+str(rgb_paper_avg["b_diff"][2]/2.0)
                    },
                    "After Average": {
                        "R": str(rgb_paper_avg["a"][0]),
                        "G": str(rgb_paper_avg["a"][1]),
                        "B": str(rgb_paper_avg["a"][2])
                    },
                    "Difference": {
                        "R": str(rgb_paper_avg["a"][0]-rgb_paper_avg["b_avg"][0])+" +-"+str(rgb_paper_avg["b_diff"][0]/2.0),
                        "G": str(rgb_paper_avg["a"][1]-rgb_paper_avg["b_avg"][1])+" +-"+str(rgb_paper_avg["b_diff"][1]/2.0),
                        "B": str(rgb_paper_avg["a"][2]-rgb_paper_avg["b_avg"][2])+" +-"+str(rgb_paper_avg["b_diff"][2]/2.0)
                    }
                },
                "Ink": {
                    "Before Average": {
                        "R": str(rgb_ink_avg["b_avg"][0])+" +-"+str(rgb_ink_avg["b_diff"][0]/2.0),
                        "G": str(rgb_ink_avg["b_avg"][1])+" +-"+str(rgb_ink_avg["b_diff"][1]/2.0),
                        "B": str(rgb_ink_avg["b_avg"][2])+" +-"+str(rgb_ink_avg["b_diff"][2]/2.0)
                    },
                    "After Average": {
                        "R": str(rgb_ink_avg["a"][0]),
                        "G": str(rgb_ink_avg["a"][1]),
                        "B": str(rgb_ink_avg["a"][2])
                    },
                    "Difference": {
                        "R": str(rgb_ink_avg["a"][0]-rgb_ink_avg["b_avg"][0])+" +-"+str(rgb_ink_avg["b_diff"][0]/2.0),
                        "G": str(rgb_ink_avg["a"][1]-rgb_ink_avg["b_avg"][1])+" +-"+str(rgb_ink_avg["b_diff"][1]/2.0),
                        "B": str(rgb_ink_avg["a"][2]-rgb_ink_avg["b_avg"][2])+" +-"+str(rgb_ink_avg["b_diff"][2]/2.0)
                    }
                }
            }
            

        # Per-pixel comparison (todo)

    # LAB comparisons
    
    return results

    # ref = _as_rgb(b1)
    # base_img = _as_rgb(b2)
    # test_img = _as_rgb(a)

    # if ref.shape != base_img.shape or ref.shape != test_img.shape:
    #     raise ValueError("B1, B2, and A must all have matching widths, heights, and channels.")

    # # Build the mask once from the reference image and reuse it for both comparisons.
    # ink_mask = compute_ink_mask(ref)
    # paper_mask = ~ink_mask

    # # B1 vs B2 is the baseline reference noise; B1 vs A is the treatment change.
    # base_diff = compute_deltas(ref, base_img)
    # test_diff = compute_deltas(ref, test_img)

    # # Keep the raw numbers visible, but also attach the Lab-direction metrics used in
    # # the human-readable report so the meaning of the result is clear.
    # base_paper = _summarize_metric_map(base_diff["dE00"][paper_mask])
    # base_paper["dL_median"] = float(np.median(base_diff["dL"][paper_mask])) if np.any(paper_mask) else 0.0
    # base_paper["db_median"] = float(np.median(base_diff["db"][paper_mask])) if np.any(paper_mask) else 0.0
    # base_ink = _summarize_metric_map(base_diff["dE00"][ink_mask])
    # base_ink["dL_median"] = float(np.median(base_diff["dL"][ink_mask])) if np.any(ink_mask) else 0.0
    # base_ink["db_median"] = float(np.median(base_diff["db"][ink_mask])) if np.any(ink_mask) else 0.0

    # test_paper = _summarize_metric_map(test_diff["dE00"][paper_mask])
    # test_paper["dL_median"] = float(np.median(test_diff["dL"][paper_mask])) if np.any(paper_mask) else 0.0
    # test_paper["db_median"] = float(np.median(test_diff["db"][paper_mask])) if np.any(paper_mask) else 0.0
    # test_ink = _summarize_metric_map(test_diff["dE00"][ink_mask])
    # test_ink["dL_median"] = float(np.median(test_diff["dL"][ink_mask])) if np.any(ink_mask) else 0.0
    # test_ink["db_median"] = float(np.median(test_diff["db"][ink_mask])) if np.any(ink_mask) else 0.0

    # valid_mask = paper_mask | ink_mask
    # base_score = float(np.mean(base_diff["dE00"][valid_mask])) if np.any(valid_mask) else 0.0
    # test_score = float(np.mean(test_diff["dE00"][valid_mask])) if np.any(valid_mask) else 0.0
    
    # return {
    #     "base": {
    #         "reference": ref,
    #         "aligned": base_img,
    #         "score": base_score,
    #         "paper": base_paper,
    #         "ink": base_ink,
    #     },
    #     "test": {
    #         "reference": ref,
    #         "aligned": test_img,
    #         "score": test_score,
    #         "paper": test_paper,
    #         "ink": test_ink,
    #     },
    
    
