import numpy as np

from Numba_Engine.output.statistics import store_base_game
from Numba_Engine.serialization.pretty_print import (
    format_base_game_statistics,
    format_payload,
)


def _write_base_result(path):
    overlays = np.full((3, 3, 5), -1, dtype=np.int8)
    overlays[1, 0, 0] = 0
    overlays[1, 0, 1] = 2
    overlays[2, 1, 0] = 2
    overlays[2, 1, 1] = 3
    values_before = np.array(
        [
            [2.0, 10.0, 100.0, 10_000.0],
            [2.0, 10.0, 100.0, 10_000.0],
            [2.1, 10.0, 105.0, 10_000.0],
        ],
        dtype=np.float64,
    )
    values_after = np.array(
        [
            [2.0, 10.0, 100.0, 10_000.0],
            [2.1, 10.0, 105.0, 10_000.0],
            [2.1, 10.0, 110.0, 10_050.0],
        ],
        dtype=np.float64,
    )
    increment_counts = np.array(
        [[0, 0, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1]],
        dtype=np.int16,
    )
    np.savez_compressed(
        path,
        spin_count=np.array(3, dtype=np.int64),
        round_count=np.array(1, dtype=np.int64),
        round_spin_offsets=np.array([0, 3], dtype=np.int64),
        wins=np.array([0.0, 1.0, 0.5], dtype=np.float64),
        line_wins=np.array([0.0, 1.0, 0.5], dtype=np.float64),
        collect_wins=np.zeros(3, dtype=np.float64),
        collector_counts=np.zeros(3, dtype=np.int16),
        spin_triggers=np.zeros(3, dtype=np.bool_),
        jackpot_overlay_boards=overlays,
        jackpot_values_before=values_before,
        jackpot_values_after=values_after,
        jackpot_increment_counts=increment_counts,
    )


def test_base_statistics_include_jackpot_overlay_activity(tmp_path):
    result_path = tmp_path / "base_results.npz"
    _write_base_result(result_path)

    statistics = store_base_game(result_path, print_result=False)

    assert statistics.jackpot_overlay_spin_count == 2
    assert np.isclose(statistics.jackpot_overlay_spin_rate, 2 / 3)
    assert statistics.total_jackpot_overlays == 4
    assert np.isclose(statistics.average_jackpot_overlays_per_spin, 4 / 3)
    assert statistics.maximum_jackpot_overlays_on_spin == 2
    np.testing.assert_array_equal(
        statistics.jackpot_overlay_counts,
        np.array([1, 0, 2, 1]),
    )
    np.testing.assert_array_equal(
        statistics.jackpot_overlay_count_histogram,
        np.array([1, 0, 2]),
    )
    np.testing.assert_allclose(
        statistics.jackpot_value_increments,
        np.array([0.1, 0.0, 10.0, 50.0]),
    )
    assert np.isclose(statistics.total_jackpot_value_increment, 60.1)

    report = format_base_game_statistics(statistics)
    assert "JACKPOT OVERLAYS BY TYPE" in report
    assert "JACKPOT OVERLAY COUNT HISTOGRAM" in report
    assert "Grand" in report


def test_base_spin_pretty_print_uses_named_jackpot_overlays():
    payload = {
        "storage_type": "base_game",
        "round_count": 1,
        "spin_count": 1,
        "rounds": [
            {
                "win": 0.0,
                "free_game_triggers": 0,
                "spins": [
                    {
                        "win": 0.0,
                        "line_win": 0.0,
                        "collect_win": 0.0,
                        "collector_count": 0,
                        "free_game_trigger": False,
                        "board": [[0, 1]],
                        "coin_value_board": [[0, 0]],
                        "jackpot_overlay_board": [[0, 2]],
                        "jackpot_values_before": [2.0, 10.0, 100.0, 10_000.0],
                        "jackpot_values_after": [2.1, 10.0, 105.0, 10_000.0],
                        "jackpot_increment_counts": [1, 0, 1, 0],
                        "winning_lines": [],
                    }
                ],
            }
        ],
    }

    report = format_payload(payload)

    assert "Mini" in report
    assert "Major" in report
    assert "Jackpot overlay counts: Mini=1, Major=1" in report

