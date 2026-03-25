"""Tests for episode filter parsing and application."""

import pytest

from easi.evaluation.episode_filter import filter_episodes, parse_episodes_flag


EPISODES = [
    {"episode_id": "10"},
    {"episode_id": "20"},
    {"episode_id": "30"},
    {"episode_id": "40"},
    {"episode_id": "50"},
]


class TestParseEpisodesFlag:
    def test_single_id(self):
        slices, ids = parse_episodes_flag("10")
        assert slices == []
        assert ids == ["10"]

    def test_multiple_ids(self):
        slices, ids = parse_episodes_flag("10,20,30")
        assert slices == []
        assert ids == ["10", "20", "30"]

    def test_range(self):
        slices, ids = parse_episodes_flag("1:3")
        assert slices == [(1, 3)]
        assert ids == []

    def test_open_start(self):
        slices, ids = parse_episodes_flag(":3")
        assert slices == [(None, 3)]
        assert ids == []

    def test_open_end(self):
        slices, ids = parse_episodes_flag("2:")
        assert slices == [(2, None)]
        assert ids == []

    def test_mixed(self):
        slices, ids = parse_episodes_flag("10,1:3,50")
        assert slices == [(1, 3)]
        assert ids == ["10", "50"]

    def test_inverted_range_error(self):
        with pytest.raises(ValueError, match="start.*must be less than stop"):
            parse_episodes_flag("5:3")

    def test_equal_range_error(self):
        with pytest.raises(ValueError, match="start.*must be less than stop"):
            parse_episodes_flag("3:3")

    def test_negative_index_error(self):
        with pytest.raises(ValueError, match="negative"):
            parse_episodes_flag("-1:3")


class TestFilterEpisodes:
    def test_first_n(self):
        result = filter_episodes(EPISODES, ":2")
        assert len(result) == 2
        assert result[0]["episode_id"] == "10"
        assert result[1]["episode_id"] == "20"

    def test_range(self):
        result = filter_episodes(EPISODES, "1:3")
        assert len(result) == 2
        assert result[0]["episode_id"] == "20"
        assert result[1]["episode_id"] == "30"

    def test_open_end(self):
        result = filter_episodes(EPISODES, "3:")
        assert len(result) == 2
        assert result[0]["episode_id"] == "40"
        assert result[1]["episode_id"] == "50"

    def test_episode_ids(self):
        result = filter_episodes(EPISODES, "20,40")
        assert len(result) == 2
        assert result[0]["episode_id"] == "20"
        assert result[1]["episode_id"] == "40"

    def test_mixed_ids_and_range(self):
        result = filter_episodes(EPISODES, "50,0:2")
        assert len(result) == 3
        # Original order: index 0, index 1, then index 4
        assert result[0]["episode_id"] == "10"
        assert result[1]["episode_id"] == "20"
        assert result[2]["episode_id"] == "50"

    def test_dedup_id_in_range(self):
        """Episode ID that also falls within a range is not duplicated."""
        result = filter_episodes(EPISODES, "20,0:3")
        assert len(result) == 3
        assert [r["episode_id"] for r in result] == ["10", "20", "30"]

    def test_overlapping_ranges(self):
        result = filter_episodes(EPISODES, "0:3,2:5")
        assert len(result) == 5
        assert [r["episode_id"] for r in result] == ["10", "20", "30", "40", "50"]

    def test_preserves_original_order(self):
        """IDs specified in reverse order still come out in dataset order."""
        result = filter_episodes(EPISODES, "50,30,10")
        assert [r["episode_id"] for r in result] == ["10", "30", "50"]

    def test_missing_id_errors(self):
        with pytest.raises(ValueError, match="not found.*999"):
            filter_episodes(EPISODES, "999")

    def test_multiple_missing_ids_errors(self):
        with pytest.raises(ValueError, match="not found"):
            filter_episodes(EPISODES, "999,888")

    def test_empty_value_errors(self):
        with pytest.raises(ValueError, match="Empty"):
            filter_episodes(EPISODES, "")

    def test_range_beyond_bounds_clamps(self):
        """Range extending past the list is clamped, not an error."""
        result = filter_episodes(EPISODES, "3:100")
        assert len(result) == 2
        assert result[0]["episode_id"] == "40"
        assert result[1]["episode_id"] == "50"

    def test_single_episode_by_id(self):
        result = filter_episodes(EPISODES, "30")
        assert len(result) == 1
        assert result[0]["episode_id"] == "30"
