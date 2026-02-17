"""Tests for src/features/build_features.py (evaluation module)"""

import csv
import os
import pytest
import numpy as np

from src.features.build_features import (
    load_results,
    group_by_method,
    compute_convergence,
    sample_efficiency,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _write_csv(path, rows):
    """Write a results CSV with the standard header."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "lat", "lng", "temp", "search_method"])
        for r in rows:
            writer.writerow(r)


# ── load_results ─────────────────────────────────────────────────────────────

class TestLoadResults:
    def test_loads_valid_csv(self, tmp_path):
        csv_path = tmp_path / "results.csv"
        _write_csv(csv_path, [
            ["2026-01-01T00:00:00", 25.0, 15.0, 22.9, "random_search"],
            ["2026-01-01T00:01:00", 10.0, 20.0, 18.5, "manual_search"],
        ])
        results = load_results(str(csv_path))
        assert len(results) == 2
        assert results[0]["temp"] == 22.9
        assert results[1]["search_method"] == "manual_search"

    def test_returns_empty_for_missing_file(self, tmp_path):
        results = load_results(str(tmp_path / "nonexistent.csv"))
        assert results == []

    def test_skips_malformed_rows(self, tmp_path):
        csv_path = tmp_path / "results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "lat", "lng", "temp", "search_method"])
            writer.writerow(["2026-01-01", 25.0, 15.0, 22.9, "ok"])
            writer.writerow(["2026-01-01", "bad", "bad", "bad", "broken"])
        results = load_results(str(csv_path))
        assert len(results) == 1


# ── group_by_method ──────────────────────────────────────────────────────────

class TestGroupByMethod:
    def test_groups_correctly(self):
        rows = [
            {"search_method": "random_search", "temp": 20},
            {"search_method": "manual_search", "temp": 25},
            {"search_method": "random_search", "temp": 30},
        ]
        groups = group_by_method(rows)
        assert len(groups) == 2
        assert len(groups["random_search"]) == 2
        assert len(groups["manual_search"]) == 1

    def test_unknown_method_fallback(self):
        rows = [{"temp": 20}]  # no search_method key
        groups = group_by_method(rows)
        assert "unknown" in groups


# ── compute_convergence ──────────────────────────────────────────────────────

class TestComputeConvergence:
    def test_running_max(self):
        group = [
            {"temp": 10},
            {"temp": 30},
            {"temp": 20},
            {"temp": 40},
        ]
        iters, bests = compute_convergence(group)
        assert iters == [1, 2, 3, 4]
        assert bests == [10, 30, 30, 40]

    def test_handles_none_temp(self):
        group = [
            {"temp": None},
            {"temp": 20},
        ]
        iters, bests = compute_convergence(group)
        assert len(iters) == 2
        # First is NaN since no valid temp yet, second is 20
        assert np.isnan(bests[0])
        assert bests[1] == 20

    def test_single_element(self):
        group = [{"temp": 42}]
        iters, bests = compute_convergence(group)
        assert iters == [1]
        assert bests == [42]


# ── sample_efficiency ────────────────────────────────────────────────────────

class TestSampleEfficiency:
    def test_immediate_reach(self):
        group = [{"temp": 100}]
        # 95% of 100 = 95; 100 >= 95 → iteration 1
        assert sample_efficiency(group, threshold_pct=0.95) == 1

    def test_reaches_at_later_iteration(self):
        group = [
            {"temp": 10},
            {"temp": 20},
            {"temp": 50},
            {"temp": 50},
        ]
        # best within group = 50; threshold = 47.5
        # running max: 10, 20, 50 → reached at iter 3
        assert sample_efficiency(group, threshold_pct=0.95) == 3

    def test_never_reached(self):
        group = [{"temp": 10}, {"temp": 12}]
        # global_best = 100 → threshold = 95 → never reached
        assert sample_efficiency(group, threshold_pct=0.95, global_best=100) is None

    def test_with_global_best(self):
        group = [{"temp": 40}, {"temp": 50}]
        # global_best=50 → threshold=47.5 → reached at iter 2
        assert sample_efficiency(group, threshold_pct=0.95, global_best=50) == 2

    def test_negative_global_best_returns_none(self):
        group = [{"temp": -10}]
        assert sample_efficiency(group, global_best=-5) is None
