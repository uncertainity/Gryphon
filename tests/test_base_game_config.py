import numpy as np

from Numba_Engine import (
    BASE_GAME_CONFIG,
    REEL_DICT,
    has_free_game_trigger,
    make_reel_collection,
)


def test_base_coin_credit_table_is_normalized_and_targets_tuned_average_value():
    rules = BASE_GAME_CONFIG

    assert len(rules.coin_credit_values) == len(
        rules.coin_credit_value_probabilities
    )
    assert np.all(np.diff(rules.coin_credit_values) > 0)
    assert np.isclose(rules.coin_credit_value_probabilities.sum(), 1.0)
    assert np.isclose(
        np.dot(
            rules.coin_credit_values,
            rules.coin_credit_value_probabilities,
        ),
        1.4132084363668658,
    )


def test_base_reel_collection_loads_all_reelsets_with_configured_weights():
    reels = make_reel_collection(
        BASE_GAME_CONFIG.reelset_path,
        BASE_GAME_CONFIG,
    )

    assert reels.reelsets.shape[0] == 3
    np.testing.assert_allclose(
        reels.weights,
        BASE_GAME_CONFIG.reelset_probabilities,
    )
    assert np.any(reels.reelsets == BASE_GAME_CONFIG.sc_symbol)
    for scatter_symbol in (
        REEL_DICT["SC1"],
        REEL_DICT["SC2"],
        REEL_DICT["SC3"],
        REEL_DICT["SC4"],
        REEL_DICT["SC5"],
        REEL_DICT["SC6"],
    ):
        assert not np.any(reels.reelsets == scatter_symbol)

    assert not np.any(reels.reelsets == BASE_GAME_CONFIG.coin_symbol)

    reelset_trigger_probabilities = np.zeros(reels.reelsets.shape[0])
    for reelset_index in range(reels.reelsets.shape[0]):
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
    assert np.isclose(combined_trigger_probability, 0.01)

    # Every ReelSet_3 stop exposes twelve generic SCs.  Mega is therefore
    # reachable through ordinary base-reel generation and independent SC
    # conversion, rather than through a tuning-only launch shortcut.
    for reel in range(4):
        length = reels.lengths[2, reel]
        strip = reels.reelsets[2, :length, reel]
        for stop in range(length):
            visible = np.array(
                [strip[(stop + row) % length] for row in range(3)]
            )
            assert np.all(visible == BASE_GAME_CONFIG.sc_symbol)


def test_only_generic_sc_triggers_conversion_from_the_base_game():
    board = np.zeros(
        (BASE_GAME_CONFIG.num_rows, BASE_GAME_CONFIG.num_reels),
        dtype=np.int16,
    )
    board[1, 2] = BASE_GAME_CONFIG.sc_symbol
    assert has_free_game_trigger(board, BASE_GAME_CONFIG)

    for scatter_symbol in BASE_GAME_CONFIG.scatter_feature_symbols:
        board[1, 2] = scatter_symbol
        assert not has_free_game_trigger(board, BASE_GAME_CONFIG)

    board = np.full(
        (BASE_GAME_CONFIG.num_rows, BASE_GAME_CONFIG.num_reels),
        REEL_DICT["H1"],
        dtype=np.int16,
    )
    assert not has_free_game_trigger(board, BASE_GAME_CONFIG)
