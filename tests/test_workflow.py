from pathlib import Path

import numpy as np

from paperdiff.loading import load_image
from paperdiff.workflow import describe_region_change, evaluate_reference_pair


def test_evaluate_reference_pair_reports_base_and_test_metrics():
    root = Path(__file__).resolve().parent.parent
    b1 = load_image(root / "test_images" / "B1.png")
    b2 = load_image(root / "test_images" / "B2.png")
    a = load_image(root / "test_images" / "A.png")

    results = evaluate_reference_pair(b1, b2, a)

    assert set(results) == {"base", "test"}
    assert results["base"]["score"] >= 0.0
    assert results["test"]["score"] >= 0.0
    assert set(results["base"]["paper"]) >= {"mean", "std", "max", "n_pixels"}
    assert set(results["base"]["ink"]) >= {"mean", "std", "max", "n_pixels"}
    assert set(results["test"]["paper"]) >= {"mean", "std", "max", "n_pixels"}
    assert set(results["test"]["ink"]) >= {"mean", "std", "max", "n_pixels"}


def test_evaluate_reference_pair_accepts_already_aligned_images():
    root = Path(__file__).resolve().parent.parent
    b1 = load_image(root / "test_images" / "B1.png")
    b2 = load_image(root / "test_images" / "B2.png")
    a = load_image(root / "test_images" / "A.png")

    results = evaluate_reference_pair(b1, b2, a)

    assert results["base"]["reference"].shape == b1.shape
    assert results["test"]["reference"].shape == b1.shape
    assert np.isfinite(results["base"]["score"])
    assert np.isfinite(results["test"]["score"])


def test_describe_region_change_mentions_measured_properties():
    metrics = {"mean": 1.8, "dL_median": 1.4, "db_median": 2.3}
    text = describe_region_change(metrics, "paper")

    assert "ΔE00" in text
    assert "brightness" in text.lower()
    assert "yellow" in text.lower() or "blue" in text.lower()
