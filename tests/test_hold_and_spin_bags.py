import numpy as np

from Numba_Engine import (
    HOLD_AND_SPIN_CONFIG,
    booster_bag,
    collect_bag_positions,
    collect_free_game_jackpot_tokens,
    collector_bag,
    expansion_bag,
    grower_bag,
    hold_and_spin,
    multiplier_bag,
    place_multiplier_cells,
)
from Numba_Engine.core.storage import HoldAndSpinStorage
from Numba_Engine.core.hold_and_spin_kernels import active_coin_win
from Numba_Engine.simulations.free_game import validate_hold_and_spin_config
from Numba_Engine.simulations.full_game import _seed_numba_random


def _new_hold_and_spin_storage(config):
    return HoldAndSpinStorage(
        1,
        1,
        1,
        config.num_rows,
        config.num_reels,
        len(config.jackpot_collection_targets),
    )


def _certain(value, dtype=np.int16):
    return np.array([value], dtype=dtype), np.array([1.0])


def _route_probability_fallbacks():
    return np.full(8, -1.0, dtype=np.float64)


def test_locked_coin_values_pay_only_after_their_row_is_unlocked():
    board = np.zeros((6, 5), dtype=np.int16)
    coin_mask = np.zeros(30, dtype=np.bool_)
    board[1, 0] = 10
    board[3, 0] = 2
    coin_mask[5] = True
    coin_mask[15] = True

    assert active_coin_win(board, coin_mask, locked_row_idx=2) == 2
    assert active_coin_win(board, coin_mask, locked_row_idx=0) == 12


def test_awarded_jackpot_weight_is_not_redistributed_to_rare_types():
    config = HOLD_AND_SPIN_CONFIG._replace(
        jackpot_token_probability=1.0,
        jackpot_type_probabilities=np.array([0.5, 0.5, 0.0, 0.0]),
        jackpot_collection_targets=np.full(4, 100, dtype=np.int16),
        max_jackpot_tokens_per_respin=1,
    )
    board = np.ones((6, 5), dtype=np.int16)
    positions = np.arange(30, dtype=np.int32)
    meters = np.zeros(4, dtype=np.int16)
    awarded = np.array([True, False, False, False])
    # Seed 4 chooses the already-awarded Mini on the first candidate. A
    # retry would eventually turn the same respin into a Minor token.
    _seed_numba_random(4)

    collect_free_game_jackpot_tokens(
        board,
        positions,
        locked_row_idx=-1,
        rules=config,
        jackpot_meters=meters,
        awarded_jackpots=awarded,
    )

    np.testing.assert_array_equal(meters, np.zeros(4, dtype=np.int16))


def test_bag_probability_tables_are_normalized_and_match_mechanic_means():
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
        (
            config.multiplier_cell_counts,
            config.multiplier_cell_count_probabilities,
        ),
        (config.multiplier_values, config.multiplier_probabilities),
    )

    for outcomes, probabilities in tables:
        assert len(outcomes) == len(probabilities)
        assert np.all(np.diff(outcomes) > 0)
        assert np.isclose(probabilities.sum(), 1.0)

    assert np.isclose(
        np.dot(
            config.splitter_source_counts,
            config.splitter_source_count_probabilities,
        ),
        2.25,
    )
    assert np.isclose(
        np.dot(
            config.splitter_copy_counts,
            config.splitter_copy_count_probabilities,
        ),
        2.25,
    )
    assert np.isclose(
        np.dot(
            config.grower_coin_counts,
            config.grower_coin_count_probabilities,
        ),
        3.5,
    )
    assert np.isclose(
        np.dot(
            config.multiplier_values,
            config.multiplier_probabilities,
        ),
        2.7,
    )
    assert np.isclose(
        np.dot(
            config.multiplier_cell_counts,
            config.multiplier_cell_count_probabilities,
        ),
        3.25,
    )

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
        np.array([2, 0, 1, 1, 1, 1]),
    )
    np.testing.assert_allclose(
        config.bag_activation_probabilities,
        np.array([1.0, 0.18, 1.0, 1.0, 1.0, 1.0]),
    )
    assert np.isclose(config.mega_coin_type_probabilities.sum(), 1.0)
    np.testing.assert_array_equal(
        config.jackpot_collection_targets,
        np.array([3, 3, 3, 3]),
    )
    assert config.max_jackpot_tokens_per_respin == 1
    assert np.isclose(config.jackpot_type_probabilities.sum(), 1.0)


def test_jackpot_token_overlay_collects_matching_session_meter():
    config = HOLD_AND_SPIN_CONFIG._replace(
        jackpot_token_probability=1.0,
        jackpot_type_probabilities=np.array([1.0, 0.0, 0.0, 0.0]),
        jackpot_collection_targets=np.array([2, 3, 3, 3], dtype=np.int16),
        max_jackpot_tokens_per_respin=1,
    )
    board = np.zeros((config.num_rows, config.num_reels), dtype=np.int16)
    board[3, 0] = 5
    board[3, 1] = 10
    landed_positions = np.array([15, 16], dtype=np.int32)
    meters = np.array([1, 0, 0, 0], dtype=np.int16)
    awarded = np.zeros(4, dtype=np.bool_)

    overlay, meters_before, newly_awarded = (
        collect_free_game_jackpot_tokens(
            board,
            landed_positions,
            locked_row_idx=2,
            rules=config,
            jackpot_meters=meters,
            awarded_jackpots=awarded,
        )
    )

    assert np.count_nonzero(overlay >= 0) == 1
    assert overlay[overlay >= 0][0] == 0
    np.testing.assert_array_equal(meters_before, np.array([1, 0, 0, 0]))
    np.testing.assert_array_equal(meters, np.array([2, 0, 0, 0]))
    np.testing.assert_array_equal(newly_awarded, np.array([True] + [False] * 3))
    np.testing.assert_array_equal(awarded, np.array([True] + [False] * 3))


def test_jackpot_token_overlay_ignores_new_qhs_in_locked_rows():
    config = HOLD_AND_SPIN_CONFIG._replace(
        jackpot_token_probability=1.0,
    )
    board = np.zeros((config.num_rows, config.num_reels), dtype=np.int16)
    board[1, 0] = 5
    meters = np.zeros(4, dtype=np.int16)
    awarded = np.zeros(4, dtype=np.bool_)

    overlay, _, newly_awarded = collect_free_game_jackpot_tokens(
        board,
        np.array([5], dtype=np.int32),
        locked_row_idx=2,
        rules=config,
        jackpot_meters=meters,
        awarded_jackpots=awarded,
    )

    assert np.all(overlay == -1)
    assert not np.any(meters)
    assert not np.any(newly_awarded)


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


def test_multiplier_feature_places_persistent_cells_for_future_coins():
    board = np.zeros((3, 5), dtype=np.int16)
    board[2, 2] = HOLD_AND_SPIN_CONFIG.multiplier_symbol
    multiplier_cells = np.ones((3, 5), dtype=np.int16)
    counts, count_probabilities = _certain(3, np.int8)
    multipliers, multiplier_probabilities = _certain(3, np.int8)

    placed = place_multiplier_cells(
        board,
        multiplier_cells,
        -1,
        count_probabilities,
        counts,
        multiplier_probabilities,
        multipliers,
    )

    assert placed == 3
    assert np.count_nonzero(multiplier_cells == 3) == 3
    assert multiplier_cells[2, 2] == 1


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
        p_coin_locked_by_feature=_route_probability_fallbacks(),
        p_coin_unlocked_by_feature=_route_probability_fallbacks(),
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


def test_expansion_uses_the_six_by_five_storage_and_unlocks_a_row():
    config = HOLD_AND_SPIN_CONFIG._replace(
        p_coin_locked=0.0,
        p_coin_unlocked=0.0,
        p_coin_locked_by_feature=_route_probability_fallbacks(),
        p_coin_unlocked_by_feature=_route_probability_fallbacks(),
        starting_coin_counts=np.array([1], dtype=np.int8),
        starting_coin_count_probabilities=np.array([1.0]),
    )
    storage = _new_hold_and_spin_storage(config)

    result = hold_and_spin(
        np.array([config.expansion_symbol], dtype=np.int16),
        config,
        storage,
    )

    assert storage.session_count == 1
    assert storage.boards.shape[1:] == (6, 5)
    assert storage.step_locked_row_indices[0] == 2
    assert result.locked_row_idx == 1
    np.testing.assert_array_equal(
        storage.session_starting_symbols[0, :1],
        np.array([config.expansion_symbol]),
    )


def test_single_bag_route_cannot_land_a_different_bag_type():
    probabilities = np.zeros(len(HOLD_AND_SPIN_CONFIG.coin_types))
    probabilities[2] = 1.0
    config = HOLD_AND_SPIN_CONFIG._replace(
        p_coin_locked=0.0,
        p_coin_unlocked=1.0,
        p_coin_locked_by_feature=_route_probability_fallbacks(),
        p_coin_unlocked_by_feature=_route_probability_fallbacks(),
        coin_type_probabilities=probabilities,
        starting_coin_counts=np.array([1], dtype=np.int8),
        starting_coin_count_probabilities=np.array([1.0]),
    )
    storage = _new_hold_and_spin_storage(config)

    result = hold_and_spin(
        np.array([config.splitter_symbol], dtype=np.int16),
        config,
        storage,
    )

    assert storage.session_count == 1
    assert result.total_spins == 1
    assert not np.any(storage.feature_types[: storage.step_count] == 1)
    assert np.count_nonzero(result.pay_window == config.grower_symbol) == 0


def test_single_bag_route_remains_stable_after_bag_resolution():
    probabilities = np.zeros(len(HOLD_AND_SPIN_CONFIG.coin_types))
    probabilities[1] = 0.5
    probabilities[2] = 0.5
    config = HOLD_AND_SPIN_CONFIG._replace(
        p_coin_locked=0.0,
        p_coin_unlocked=0.20,
        p_coin_locked_by_feature=_route_probability_fallbacks(),
        p_coin_unlocked_by_feature=_route_probability_fallbacks(),
        coin_type_probabilities=probabilities,
        starting_coin_counts=np.array([1], dtype=np.int8),
        starting_coin_count_probabilities=np.array([1.0]),
    )
    storage = HoldAndSpinStorage(
        1,
        1,
        1,
        config.num_rows,
        config.num_reels,
        len(config.jackpot_collection_targets),
    )
    _seed_numba_random(20261026)

    for _ in range(25):
        hold_and_spin(
            np.array([config.grower_symbol], dtype=np.int16),
            config,
            storage,
        )

    resolved_types = storage.feature_types[: storage.step_count]
    assert np.any(resolved_types == 1)
    assert not np.any(resolved_types == 0)


def test_fixed_three_by_five_route_never_populates_padding_rows():
    probabilities = np.zeros(len(HOLD_AND_SPIN_CONFIG.coin_types))
    probabilities[0] = 1.0
    config = HOLD_AND_SPIN_CONFIG._replace(
        p_coin_locked=1.0,
        p_coin_unlocked=1.0,
        p_coin_locked_by_feature=_route_probability_fallbacks(),
        p_coin_unlocked_by_feature=_route_probability_fallbacks(),
        coin_type_probabilities=probabilities,
        starting_coin_counts=np.array([1], dtype=np.int8),
        starting_coin_count_probabilities=np.array([1.0]),
    )
    storage = _new_hold_and_spin_storage(config)

    result = hold_and_spin(
        np.array([config.splitter_symbol], dtype=np.int16),
        config,
        storage,
    )

    assert storage.boards.shape[1:] == (6, 5)
    assert not np.any(result.pay_window[:3])
    assert np.all(result.pay_window[3:] > 0)


def test_partial_multi_bag_start_is_rejected_before_board_play():
    config = HOLD_AND_SPIN_CONFIG

    with np.testing.assert_raises_regex(ValueError, "plain, one Bag, or all"):
        hold_and_spin(
            np.array(
                [config.splitter_symbol, config.grower_symbol],
                dtype=np.int16,
            ),
            config,
            _new_hold_and_spin_storage(config),
        )


def test_occupied_count_tables_override_only_the_configured_state():
    unlocked_probabilities = (
        HOLD_AND_SPIN_CONFIG.p_coin_unlocked_by_occupied_count.copy()
    )
    unlocked_probabilities[2] = 1.0
    type_probabilities = (
        HOLD_AND_SPIN_CONFIG.coin_type_probabilities_by_occupied_count.copy()
    )
    type_probabilities[2] = 0.0
    type_probabilities[2, 0] = 1.0
    config = HOLD_AND_SPIN_CONFIG._replace(
        p_coin_locked=0.0,
        p_coin_unlocked=0.0,
        p_coin_locked_by_feature=_route_probability_fallbacks(),
        p_coin_unlocked_by_feature=_route_probability_fallbacks(),
        p_coin_unlocked_by_occupied_count=unlocked_probabilities,
        coin_type_probabilities_by_occupied_count=type_probabilities,
        starting_coin_counts=np.array([1], dtype=np.int8),
        starting_coin_count_probabilities=np.array([1.0]),
    )
    validate_hold_and_spin_config(config)
    storage = _new_hold_and_spin_storage(config)

    result = hold_and_spin(
        np.array([config.grower_symbol], dtype=np.int16),
        config,
        storage,
    )

    assert storage.respin_reset_flags[0]
    assert result.total_spins == config.respin_reset_count + 1
    assert np.count_nonzero(result.coin_mask) == 2


def test_hold_and_spin_validation_rejects_malformed_state_tables():
    invalid = HOLD_AND_SPIN_CONFIG._replace(
        p_coin_unlocked_by_occupied_count=np.full(30, -1.0),
    )

    with np.testing.assert_raises_regex(
        ValueError,
        "one entry for every occupied-cell count",
    ):
        validate_hold_and_spin_config(invalid)
