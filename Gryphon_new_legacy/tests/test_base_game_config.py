import numpy as np

from Numba_Engine import (
    BASE_GAME_CONFIG,
    REEL_DICT,
    has_free_game_trigger,
    make_reel_collection,
)


def test_base_coin_credit_table_is_normalized_and_targets_average_value():
    rules = BASE_GAME_CONFIG

    assert len(rules.coin_credit_values) == len(
        rules.coin_credit_value_probabilities
    )
    assert np.all(np.diff(rules.coin_credit_values) > 0)
    assert np.isclose(rules.coin_credit_value_probabilities.sum(), 1.0)
    assert 1.15 <= np.dot(
        rules.coin_credit_values,
        rules.coin_credit_value_probabilities,
    ) <= 1.25


def test_base_reel_collection_loads_both_reelsets_with_configured_weights():
    reels = make_reel_collection(
        BASE_GAME_CONFIG.reelset_path,
        BASE_GAME_CONFIG,
    )

    assert reels.reelsets.shape[0] == 2
    np.testing.assert_allclose(
        reels.weights,
        np.array([0.50, 0.50]),
    )
    assert np.any(reels.reelsets == BASE_GAME_CONFIG.sc_symbol)
    for scatter_symbol in (
        REEL_DICT["SC1"],
        REEL_DICT["SC2"],
        REEL_DICT["SC3"],
        REEL_DICT["SC4"],
        REEL_DICT["SC5"],
        REEL_DICT["SC6"],
        REEL_DICT["SC7"],
    ):
        assert not np.any(reels.reelsets == scatter_symbol)

    assert not np.any(reels.reelsets == BASE_GAME_CONFIG.coin_symbol)

    reelset_trigger_probabilities = np.zeros(2)
    for reelset_index in range(2):
        no_sc_probability = 1.0
        for reel in range(BASE_GAME_CONFIG.num_reels):
            reel_length = reels.lengths[reelset_index, reel]
            strip = reels.reelsets[reelset_index, :reel_length, reel]
            triggering_stops = 0
            for stop in range(reel_length):
                visible = np.array(
                    [strip[(stop + row) % reel_length] for row in range(3)]
                )
                if np.any(
                    np.isin(visible, BASE_GAME_CONFIG.free_game_symbols)
                ):
                    triggering_stops += 1
            no_sc_probability *= 1.0 - triggering_stops / reel_length
        reelset_trigger_probabilities[reelset_index] = 1.0 - no_sc_probability

    combined_trigger_probability = np.dot(
        reels.weights,
        reelset_trigger_probabilities,
    )
    assert 0.045 <= combined_trigger_probability <= 0.055


def test_any_single_sc_symbol_can_trigger_free_game():
    for scatter_symbol in BASE_GAME_CONFIG.free_game_symbols:
        board = np.zeros(
            (BASE_GAME_CONFIG.num_rows, BASE_GAME_CONFIG.num_reels),
            dtype=np.int16,
        )
        board[1, 2] = scatter_symbol
        assert has_free_game_trigger(board, BASE_GAME_CONFIG)

    board = np.full(
        (BASE_GAME_CONFIG.num_rows, BASE_GAME_CONFIG.num_reels),
        REEL_DICT["H1"],
        dtype=np.int16,
    )
    assert not has_free_game_trigger(board, BASE_GAME_CONFIG)
