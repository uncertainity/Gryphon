import numpy as np

from Numba_Engine import (
    BASE_GAME_CONFIG,
    BOOST_FEATURE_CONFIG,
    COLLECT_FEATURE_CONFIG,
    EXPANSION_FEATURE_CONFIG,
    FEATURE_COIN_VALUE_PROBABILITIES,
    FEATURE_COIN_VALUES,
    GROW_FEATURE_CONFIG,
    GROW_VALUE_PROBABILITIES,
    GROW_VALUES,
    MEGA_COMBO_FEATURE_CONFIG,
    MULTIPLIER_FEATURE_CONFIG,
    REEL_DICT,
    SPLITTER_FEATURE_CONFIG,
    run_boost_feature,
    run_collect_feature,
    run_expansion_feature,
    run_grow_feature,
    run_mega_combo_feature,
    run_multiplier_feature,
    run_splitter_feature,
)
from Numba_Engine.simulations.base_game import convert_base_scatters


def test_base_scatter_converts_to_weighted_feature_symbol():
    board = np.full(
        (BASE_GAME_CONFIG.num_rows, BASE_GAME_CONFIG.num_reels),
        REEL_DICT["H1"],
        dtype=np.int16,
    )
    board[0, 0] = BASE_GAME_CONFIG.sc_symbol
    board[1, 2] = BASE_GAME_CONFIG.sc_symbol

    converted, feature_flags, combo_triggered = convert_base_scatters(
        board,
        BASE_GAME_CONFIG,
    )

    assert not np.any(converted == BASE_GAME_CONFIG.sc_symbol)
    assert np.count_nonzero(np.isin(converted, BASE_GAME_CONFIG.scatter_feature_symbols)) == 2
    assert np.any(feature_flags)
    assert not combo_triggered


def test_feature_coin_value_average_is_above_one():
    assert np.isclose(FEATURE_COIN_VALUE_PROBABILITIES.sum(), 1.0)
    assert np.dot(FEATURE_COIN_VALUES, FEATURE_COIN_VALUE_PROBABILITIES) > 1.0
    assert np.isclose(GROW_VALUE_PROBABILITIES.sum(), 1.0)
    assert np.all(np.diff(GROW_VALUES) > 0)


def test_expansion_feature_respects_go_limits_and_grid_shape():
    for seed in range(20):
        result = run_expansion_feature(seed=seed)
        assert result.board.shape == (
            EXPANSION_FEATURE_CONFIG.num_rows,
            EXPANSION_FEATURE_CONFIG.num_reels,
        )
        assert result.go_landed <= EXPANSION_FEATURE_CONFIG.max_go_symbols
        assert result.locked_go_landed <= EXPANSION_FEATURE_CONFIG.max_locked_go_symbols
        assert result.total_spins >= EXPANSION_FEATURE_CONFIG.respin_reset_count
        assert np.all(result.awarded_jackpots >= 0)


def test_multiplier_feature_uses_3x5_window_and_multiplier_cells():
    for seed in range(20):
        result = run_multiplier_feature(seed=seed)
        assert result.board.shape == (
            MULTIPLIER_FEATURE_CONFIG.num_rows,
            MULTIPLIER_FEATURE_CONFIG.num_reels,
        )
        assert result.multiplier_cells.shape == result.board.shape
        assert np.all(np.isin(result.multiplier_cells, [1, 2, 3, 4]))
        assert result.total_spins >= MULTIPLIER_FEATURE_CONFIG.respin_reset_count
        assert np.all(result.awarded_jackpots >= 0)


def test_grow_feature_limits_growers_and_uses_3x5_grid():
    for seed in range(20):
        result = run_grow_feature(seed=seed)
        assert result.board.shape == (
            GROW_FEATURE_CONFIG.num_rows,
            GROW_FEATURE_CONFIG.num_reels,
        )
        assert result.special_landed <= GROW_FEATURE_CONFIG.max_grower_symbols
        assert np.all(result.awarded_jackpots >= 0)


def test_boost_feature_uses_3x5_grid_and_converts_booster():
    for seed in range(20):
        result = run_boost_feature(seed=seed)
        assert result.board.shape == (
            BOOST_FEATURE_CONFIG.num_rows,
            BOOST_FEATURE_CONFIG.num_reels,
        )
        assert not np.any(result.board == BOOST_FEATURE_CONFIG.trigger_symbol)
        assert np.all(result.awarded_jackpots >= 0)


def test_collect_feature_limits_collectors_and_converts_to_coin():
    for seed in range(20):
        result = run_collect_feature(seed=seed)
        assert result.board.shape == (
            COLLECT_FEATURE_CONFIG.num_rows,
            COLLECT_FEATURE_CONFIG.num_reels,
        )
        assert result.special_landed <= COLLECT_FEATURE_CONFIG.max_collector_symbols
        assert not np.any(result.board == COLLECT_FEATURE_CONFIG.collector_symbol)
        assert result.collected_value >= 0.0
        assert np.all(result.awarded_jackpots >= 0)


def test_splitter_feature_uses_2_or_3_coin_cells_and_minimum_rule():
    for seed in range(20):
        result = run_splitter_feature(seed=seed)
        assert result.board.shape == (
            SPLITTER_FEATURE_CONFIG.num_rows,
            SPLITTER_FEATURE_CONFIG.num_reels,
        )
        split_cells = result.splitter_coin_counts[
            result.board == SPLITTER_FEATURE_CONFIG.splitter_symbol
        ]
        assert result.special_landed >= 1
        assert np.all(np.isin(split_cells, [2, 3]))
        if np.count_nonzero(result.board == -1) < 4:
            assert result.special_landed >= SPLITTER_FEATURE_CONFIG.min_splitter_symbols
        assert np.all(result.awarded_jackpots >= 0)


def test_mega_combo_requires_all_six_base_triggers():
    base_window = np.full((3, 5), REEL_DICT["H1"], dtype=np.int16)
    base_window[0, 0] = REEL_DICT["SC1"]
    base_window[0, 1] = REEL_DICT["SC2"]
    base_window[1, 0] = REEL_DICT["SC3"]
    base_window[1, 1] = REEL_DICT["SC4"]
    base_window[2, 0] = REEL_DICT["SC5"]

    try:
        run_mega_combo_feature(base_window=base_window, seed=1)
    except ValueError:
        pass
    else:
        raise AssertionError("combo feature should require SC1-SC6")


def test_mega_combo_uses_exact_base_positions_in_active_area_and_caps():
    base_window = np.full((3, 5), REEL_DICT["H1"], dtype=np.int16)
    placements = [
        (REEL_DICT["SC1"], 0, 0),
        (REEL_DICT["SC2"], 0, 2),
        (REEL_DICT["SC3"], 1, 1),
        (REEL_DICT["SC4"], 1, 3),
        (REEL_DICT["SC5"], 2, 0),
        (REEL_DICT["SC6"], 2, 4),
    ]
    for symbol, row, col in placements:
        base_window[row, col] = symbol

    result = run_mega_combo_feature(base_window=base_window, seed=12)

    assert result.board.shape == (
        MEGA_COMBO_FEATURE_CONFIG.num_rows,
        MEGA_COMBO_FEATURE_CONFIG.num_reels,
    )
    active_offset = (
        MEGA_COMBO_FEATURE_CONFIG.num_rows
        - MEGA_COMBO_FEATURE_CONFIG.starting_unlocked_rows
    )
    for _, row, col in placements:
        assert result.board[active_offset + row, col] != -1
    assert result.multiplier_cells.shape == result.board.shape
    assert result.special_counts["go"] <= MEGA_COMBO_FEATURE_CONFIG.max_go_symbols
    assert (
        result.special_counts["locked_go"]
        <= MEGA_COMBO_FEATURE_CONFIG.max_locked_go_symbols
    )
    assert (
        result.special_counts["grower"]
        <= MEGA_COMBO_FEATURE_CONFIG.max_grower_symbols
    )
    assert (
        result.special_counts["collector"]
        <= MEGA_COMBO_FEATURE_CONFIG.max_collector_symbols
    )
    assert result.special_counts["splitter"] >= MEGA_COMBO_FEATURE_CONFIG.min_splitter_symbols
    assert np.all(result.awarded_jackpots >= 0)
