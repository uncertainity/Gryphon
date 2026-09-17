import numpy as np

from Numba_Engine import (
    HOLD_AND_SPIN_CONFIG,
    booster_bag,
    collect_bag_positions,
    collector_bag,
    expansion_bag,
    grower_bag,
    hold_and_spin,
    multiplier_bag,
)
from Numba_Engine.core.storage import HoldAndSpinStorage


def _new_hold_and_spin_storage(config):
    return HoldAndSpinStorage(
        1,
        1,
        1,
        config.num_rows,
        config.num_reels,
    )


def _certain(value, dtype=np.int16):
    return np.array([value], dtype=dtype), np.array([1.0])


def test_bag_probability_tables_favor_lower_rtp_outcomes():
    config = HOLD_AND_SPIN_CONFIG
    tables = (
        (
            config.splitter_source_counts,
            config.splitter_source_count_probabilities,
        ),
        (
            config.splitter_copy_counts,
            config.splitter_copy_count_probabilities,
        ),
        (config.grower_coin_counts, config.grower_coin_count_probabilities),
        (
            config.grower_increment_values,
            config.grower_increment_probabilities,
        ),
        (
            config.booster_increment_values,
            config.booster_increment_probabilities,
        ),
        (config.multiplier_values, config.multiplier_probabilities),
    )

    for outcomes, probabilities in tables:
        assert len(outcomes) == len(probabilities)
        assert np.all(np.diff(outcomes) > 0)
        assert np.all(np.diff(probabilities) < 0)
        assert np.isclose(probabilities.sum(), 1.0)

    bag_landing_probabilities = config.coin_type_probabilities[1:]
    assert len(np.unique(bag_landing_probabilities)) == len(
        bag_landing_probabilities
    )
    assert np.isclose(config.coin_type_probabilities.sum(), 1.0)
    np.testing.assert_array_equal(
        config.bag_resolution_order,
        np.array([5, 0, 2, 1, 3, 4]),
    )
    np.testing.assert_array_equal(
        config.bag_symbol_actions,
        np.array([2, 0, 1, 1, 1, 2]),
    )


def test_grower_selects_unique_coins_and_applies_weighted_increment():
    board = np.zeros((6, 5), dtype=np.int16)
    board[3, 0] = 2
    board[3, 1] = 5
    board[3, 2] = 10
    counts, count_probabilities = _certain(2, np.int8)
    increments, increment_probabilities = _certain(3)

    grower_bag(
        board,
        np.array([15, 16, 17], dtype=np.int32),
        2,
        count_probabilities,
        counts,
        increment_probabilities,
        increments,
        100,
    )

    assert np.count_nonzero(board[3, :3] == np.array([2, 5, 10])) == 1
    assert board[3, :3].sum() == 23


def test_booster_and_multiplier_apply_to_every_supplied_coin_with_cap():
    board = np.zeros((3, 5), dtype=np.int16)
    board[0, :3] = np.array([2, 5, 40])
    positions = np.array([0, 1, 2], dtype=np.int32)
    increments, increment_probabilities = _certain(5)
    multipliers, multiplier_probabilities = _certain(3, np.int8)

    booster_bag(
        board,
        positions,
        increment_probabilities,
        increments,
        50,
    )
    multiplier_bag(
        board,
        positions,
        multiplier_probabilities,
        multipliers,
        50,
    )

    np.testing.assert_array_equal(board[0, :3], np.array([21, 30, 50]))


def test_collector_does_not_remove_coins_and_expansion_unlocks_one_row():
    board = np.zeros((6, 5), dtype=np.int16)
    board[3, :3] = np.array([2, 5, 10])
    original = board.copy()

    meter = collector_bag(
        board,
        np.array([15, 16, 17], dtype=np.int32),
        4,
    )

    assert meter == 21
    np.testing.assert_array_equal(board, original)
    assert expansion_bag(2, 1) == 1
    assert expansion_bag(0, 1) == -1
    assert expansion_bag(-1, 1) == -1


def test_bag_scan_uses_coin_mask_when_coin_value_matches_symbol_id():
    config = HOLD_AND_SPIN_CONFIG
    board = np.array([[config.splitter_symbol, config.splitter_symbol]])
    coin_mask = np.array([True, False])

    positions, counts = collect_bag_positions(
        board,
        coin_mask,
        config.bag_symbols,
    )

    assert counts[0] == 1
    assert positions[0, 0] == 1


def test_hold_and_spin_resolves_all_bags_in_configured_order():
    config = HOLD_AND_SPIN_CONFIG._replace(
        p_coin_locked=0.0,
        p_coin_unlocked=0.0,
        starting_coin_counts=np.array([1], dtype=np.int8),
        starting_coin_count_probabilities=np.array([1.0]),
        coin_values=np.array([2], dtype=np.int16),
        coin_value_probabilities=np.array([1.0]),
        splitter_source_counts=np.array([1], dtype=np.int8),
        splitter_source_count_probabilities=np.array([1.0]),
        splitter_copy_counts=np.array([1], dtype=np.int8),
        splitter_copy_count_probabilities=np.array([1.0]),
        grower_coin_counts=np.array([1], dtype=np.int8),
        grower_coin_count_probabilities=np.array([1.0]),
        grower_increment_values=np.array([1], dtype=np.int16),
        grower_increment_probabilities=np.array([1.0]),
        booster_increment_values=np.array([1], dtype=np.int16),
        booster_increment_probabilities=np.array([1.0]),
        multiplier_values=np.array([2], dtype=np.int8),
        multiplier_probabilities=np.array([1.0]),
    )

    result = hold_and_spin(
        config.bag_symbols.copy(),
        config,
        _new_hold_and_spin_storage(config),
    )
    feature_positions, feature_counts = collect_bag_positions(
        result.pay_window,
        result.coin_mask,
        config.bag_symbols,
    )

    assert result.locked_row_idx == 1
    assert result.total_spins == config.respin_reset_count
    assert result.collector_meter > 0
    assert feature_counts[1] == 1
    assert feature_positions[1, 0] >= 0
    assert feature_counts.sum() == 1
