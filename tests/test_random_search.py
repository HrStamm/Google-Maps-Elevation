"""Tests for src/models/random_search.py"""

from unittest.mock import patch
import numpy as np

from src.models.random_search import random_search


class TestRandomSearch:
    @patch("src.models.random_search.fetch_temperature")
    def test_returns_correct_number_of_results(self, mock_fetch):
        mock_fetch.return_value = 25.0
        results = random_search(n_iterations=5, seed=42)
        assert len(results) == 5

    @patch("src.models.random_search.fetch_temperature")
    def test_iterations_are_sequential(self, mock_fetch):
        mock_fetch.return_value = 20.0
        results = random_search(n_iterations=3, seed=1)
        assert [r["iteration"] for r in results] == [1, 2, 3]

    @patch("src.models.random_search.fetch_temperature")
    def test_coordinates_within_bounds(self, mock_fetch):
        mock_fetch.return_value = 15.0
        results = random_search(n_iterations=100, seed=42)
        for r in results:
            assert -90 <= r["lat"] <= 90
            assert -180 <= r["lng"] <= 180

    @patch("src.models.random_search.fetch_temperature")
    def test_best_temp_is_running_max(self, mock_fetch):
        temps = [10.0, 30.0, 20.0, 40.0, 5.0]
        mock_fetch.side_effect = temps
        results = random_search(n_iterations=5, seed=0)

        expected_bests = [10.0, 30.0, 30.0, 40.0, 40.0]
        for r, expected in zip(results, expected_bests):
            assert r["best_temp"] == expected

    @patch("src.models.random_search.fetch_temperature")
    def test_handles_none_temperature(self, mock_fetch):
        mock_fetch.side_effect = [None, 20.0, None]
        results = random_search(n_iterations=3, seed=0)

        assert results[0]["temp"] is None
        assert results[0]["best_temp"] is None
        assert results[1]["best_temp"] == 20.0
        assert results[2]["best_temp"] == 20.0

    @patch("src.models.random_search.fetch_temperature")
    def test_seed_reproducibility(self, mock_fetch):
        mock_fetch.return_value = 25.0
        r1 = random_search(n_iterations=5, seed=42)
        r2 = random_search(n_iterations=5, seed=42)
        for a, b in zip(r1, r2):
            assert a["lat"] == b["lat"]
            assert a["lng"] == b["lng"]

    @patch("src.models.random_search.fetch_temperature")
    def test_calls_fetch_with_random_search_method(self, mock_fetch):
        mock_fetch.return_value = 25.0
        random_search(n_iterations=1, seed=0)
        _, kwargs = mock_fetch.call_args
        assert kwargs["search_method"] == "random_search"
        assert kwargs["use_cache"] is False
