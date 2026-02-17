"""Tests for src/models/manual_search.py"""

from unittest.mock import patch
from src.models.manual_search import manual_search


class TestManualSearch:
    @patch("src.models.manual_search.fetch_temperature")
    @patch("builtins.input")
    def test_single_valid_guess(self, mock_input, mock_fetch):
        mock_input.side_effect = ["25.0, 15.0", "q"]
        mock_fetch.return_value = 30.0

        results = manual_search()

        assert len(results) == 1
        assert results[0]["lat"] == 25.0
        assert results[0]["lng"] == 15.0
        assert results[0]["temp"] == 30.0

    @patch("src.models.manual_search.fetch_temperature")
    @patch("builtins.input")
    def test_quit_immediately(self, mock_input, mock_fetch):
        mock_input.return_value = "q"
        results = manual_search()
        assert len(results) == 0
        mock_fetch.assert_not_called()

    @patch("src.models.manual_search.fetch_temperature")
    @patch("builtins.input")
    def test_invalid_input_skipped(self, mock_input, mock_fetch):
        mock_input.side_effect = ["not_a_number", "25.0 15.0", "q"]
        mock_fetch.return_value = 22.0

        results = manual_search()

        # Only the valid "25.0 15.0" should produce a result
        assert len(results) == 1

    @patch("src.models.manual_search.fetch_temperature")
    @patch("builtins.input")
    def test_out_of_bounds_skipped(self, mock_input, mock_fetch):
        mock_input.side_effect = ["100, 200", "25.0, 15.0", "q"]
        mock_fetch.return_value = 20.0

        results = manual_search()
        assert len(results) == 1
        assert results[0]["lat"] == 25.0

    @patch("src.models.manual_search.fetch_temperature")
    @patch("builtins.input")
    def test_best_temp_tracking(self, mock_input, mock_fetch):
        mock_input.side_effect = ["10, 10", "20, 20", "q"]
        mock_fetch.side_effect = [15.0, 35.0]

        results = manual_search()

        assert results[0]["best_temp"] == 15.0
        assert results[1]["best_temp"] == 35.0

    @patch("src.models.manual_search.fetch_temperature")
    @patch("builtins.input")
    def test_max_guesses_enforced(self, mock_input, mock_fetch):
        mock_input.side_effect = ["10, 10", "20, 20", "30, 30"]
        mock_fetch.return_value = 25.0

        results = manual_search(max_guesses=2)
        assert len(results) == 2

    @patch("src.models.manual_search.fetch_temperature")
    @patch("builtins.input")
    def test_space_separated_input(self, mock_input, mock_fetch):
        mock_input.side_effect = ["55.67 12.56", "q"]
        mock_fetch.return_value = 5.0

        results = manual_search()
        assert results[0]["lat"] == 55.67
        assert results[0]["lng"] == 12.56

    @patch("src.models.manual_search.fetch_temperature")
    @patch("builtins.input")
    def test_calls_fetch_with_manual_method(self, mock_input, mock_fetch):
        mock_input.side_effect = ["25, 15", "q"]
        mock_fetch.return_value = 30.0

        manual_search()
        _, kwargs = mock_fetch.call_args
        assert kwargs["search_method"] == "manual_search"
        assert kwargs["use_cache"] is False
