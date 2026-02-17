"""Tests for src/data/data_manager.py"""

import csv
import os
import tempfile
import pytest

from src.data.data_manager import is_valid_coordinate, save_result, load_results, get_cached_result


# ── is_valid_coordinate ──────────────────────────────────────────────────────

class TestIsValidCoordinate:
    def test_valid_coordinates(self):
        assert is_valid_coordinate(0, 0)
        assert is_valid_coordinate(55.67, 12.56)
        assert is_valid_coordinate(-33.86, 151.21)

    def test_boundary_values(self):
        assert is_valid_coordinate(90, 180)
        assert is_valid_coordinate(-90, -180)
        assert is_valid_coordinate(90, -180)
        assert is_valid_coordinate(-90, 180)

    def test_invalid_lat(self):
        assert not is_valid_coordinate(91, 0)
        assert not is_valid_coordinate(-91, 0)

    def test_invalid_lng(self):
        assert not is_valid_coordinate(0, 181)
        assert not is_valid_coordinate(0, -181)

    def test_both_invalid(self):
        assert not is_valid_coordinate(100, 200)


# ── save_result / load_results ───────────────────────────────────────────────

class TestSaveAndLoad:
    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        """save_result → load_results should return the saved data."""
        fake_csv = tmp_path / "results.csv"
        monkeypatch.setattr("src.data.data_manager.RESULTS_FILE", str(fake_csv))

        save_result(25.0, 15.0, 22.9, "test_method")
        save_result(55.67, 12.56, 5.3, "test_method")

        results = load_results()
        assert len(results) == 2
        assert float(results[0]["lat"]) == 25.0
        assert float(results[0]["temp"]) == 22.9
        assert results[0]["search_method"] == "test_method"
        assert float(results[1]["lat"]) == 55.67

    def test_load_returns_empty_when_no_file(self, tmp_path, monkeypatch):
        fake_csv = tmp_path / "nonexistent.csv"
        monkeypatch.setattr("src.data.data_manager.RESULTS_FILE", str(fake_csv))
        assert load_results() == []

    def test_save_creates_header(self, tmp_path, monkeypatch):
        fake_csv = tmp_path / "results.csv"
        monkeypatch.setattr("src.data.data_manager.RESULTS_FILE", str(fake_csv))

        save_result(10, 20, 30.0, "header_test")

        with open(fake_csv) as f:
            header = f.readline().strip()
        assert header == "timestamp,lat,lng,temp,search_method"


# ── get_cached_result ────────────────────────────────────────────────────────

class TestGetCachedResult:
    def _seed_csv(self, path, rows):
        """Helper to create a CSV with given rows."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "lat", "lng", "temp", "search_method"])
            for r in rows:
                writer.writerow(r)

    def test_cache_hit_exact(self, tmp_path, monkeypatch):
        fake_csv = tmp_path / "results.csv"
        self._seed_csv(fake_csv, [
            ["2026-01-01T00:00:00", 25.0, 15.0, 22.9, "test"],
        ])
        monkeypatch.setattr("src.data.data_manager.RESULTS_FILE", str(fake_csv))

        assert get_cached_result(25.0, 15.0) == 22.9

    def test_cache_hit_within_tolerance(self, tmp_path, monkeypatch):
        fake_csv = tmp_path / "results.csv"
        self._seed_csv(fake_csv, [
            ["2026-01-01T00:00:00", 25.0, 15.0, 22.9, "test"],
        ])
        monkeypatch.setattr("src.data.data_manager.RESULTS_FILE", str(fake_csv))

        assert get_cached_result(25.00005, 15.00005) == 22.9

    def test_cache_miss(self, tmp_path, monkeypatch):
        fake_csv = tmp_path / "results.csv"
        self._seed_csv(fake_csv, [
            ["2026-01-01T00:00:00", 25.0, 15.0, 22.9, "test"],
        ])
        monkeypatch.setattr("src.data.data_manager.RESULTS_FILE", str(fake_csv))

        assert get_cached_result(0.0, 0.0) is None

    def test_cache_miss_empty_file(self, tmp_path, monkeypatch):
        fake_csv = tmp_path / "nonexistent.csv"
        monkeypatch.setattr("src.data.data_manager.RESULTS_FILE", str(fake_csv))

        assert get_cached_result(25.0, 15.0) is None
