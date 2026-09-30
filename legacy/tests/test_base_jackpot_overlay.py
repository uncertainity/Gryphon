import numpy as np

from Numba_Engine import (
    BASE_JACKPOT_OVERLAY_CONFIG,
    JACKPOT_CONFIG,
    REEL_DICT,
    BaseJackpotOverlayConfig,
    apply_jackpot_overlay,
    make_base_jackpot_overlay,
)


def _forced_overlay_rules(count, jackpot_type):
    count_probabilities = np.zeros(5, dtype=np.float64)
    count_probabilities[count] = 1.0
    type_probabilities = np.zeros(4, dtype=np.float64)
    type_probabilities[jackpot_type] = 1.0
    return BaseJackpotOverlayConfig(
        count_values=np.array([0, 1, 2, 3, 4], dtype=np.int8),
        count_probabilities=count_probabilities,
        jackpot_type_probabilities=type_probabilities,
        eligible_symbols=np.arange(12, dtype=np.int16),
    )


def test_overlay_count_values_are_configurable_zero_through_four():
    np.testing.assert_array_equal(
        BASE_JACKPOT_OVERLAY_CONFIG.count_values,
        np.array([0, 1, 2, 3, 4], dtype=np.int8),
    )
    assert np.isclose(
        BASE_JACKPOT_OVERLAY_CONFIG.count_probabilities.sum(),
        1.0,
    )


def test_four_overlays_use_unique_eligible_positions_without_changing_symbols():
    board = np.full((3, 5), REEL_DICT["H1"], dtype=np.int16)
    original_board = board.copy()

    overlay = make_base_jackpot_overlay(
        board,
        _forced_overlay_rules(count=4, jackpot_type=2),
        JACKPOT_CONFIG.jackpot_types,
    )

    np.testing.assert_array_equal(board, original_board)
    assert np.count_nonzero(overlay >= 0) == 4
    assert np.all(overlay[overlay >= 0] == 2)


def test_overlay_count_is_limited_by_eligible_positions():
    board = np.full((3, 5), REEL_DICT["COIN"], dtype=np.int16)
    board[1, 2] = REEL_DICT["WD"]

    overlay = make_base_jackpot_overlay(
        board,
        _forced_overlay_rules(count=4, jackpot_type=0),
        JACKPOT_CONFIG.jackpot_types,
    )

    assert np.count_nonzero(overlay >= 0) == 1
    assert overlay[1, 2] == 0


def test_overlay_increments_matching_jackpot_and_stops_at_twice_seed():
    overlay = np.full((3, 5), -1, dtype=np.int8)
    overlay[0, 0] = 2
    overlay[1, 1] = 2
    major = 2
    jackpot_values = JACKPOT_CONFIG.seed_values.copy()
    jackpot_values[major] = (
        JACKPOT_CONFIG.seed_values[major]
        * JACKPOT_CONFIG.cap_multiplier
        - JACKPOT_CONFIG.increment_values[major] / 2.0
    )

    values_before, increment_counts = apply_jackpot_overlay(
        overlay,
        JACKPOT_CONFIG,
        jackpot_values,
    )

    assert increment_counts[major] == 2
    assert values_before[major] < jackpot_values[major]
    assert jackpot_values[major] == (
        JACKPOT_CONFIG.seed_values[major]
        * JACKPOT_CONFIG.cap_multiplier
    )
