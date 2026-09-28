import numpy as np

from Numba_Engine import (
    BASE_JACKPOT_OVERLAY_CONFIG,
    FULL_GAME_CONFIG,
    JACKPOT_CONFIG,
    REEL_DICT,
    ReelCollections,
)
from Numba_Engine.core import storage as storage_module
from Numba_Engine.output.statistics import store_full_game
from Numba_Engine.serialization.json_output import full_game_storage_to_dict
from Numba_Engine.serialization.pretty_print import (
    format_full_game_statistics,
    format_payload,
)
from Numba_Engine.simulations.full_game import (
    _new_storages,
    extract_starting_bag_symbols,
    run_one_full_round,
)


def _single_stop_reels(symbols):
    return ReelCollections(
        lengths=np.ones((1, 5), dtype=np.int16),
        reelsets=np.array([[symbols]], dtype=np.int16),
        weights=np.array([1.0]),
    )


def _certain_jackpot_config():
    overlay = BASE_JACKPOT_OVERLAY_CONFIG._replace(
        count_values=np.array([1], dtype=np.int8),
        count_probabilities=np.array([1.0]),
        jackpot_type_probabilities=np.array([1.0, 0.0, 0.0, 0.0]),
        eligible_symbols=np.array([REEL_DICT["H1"]], dtype=np.int16),
    )
    feature = FULL_GAME_CONFIG.free_game._replace(
        p_coin_locked=1.0,
        p_coin_unlocked=1.0,
        coin_types=np.array([REEL_DICT["COIN"]], dtype=np.int16),
        coin_type_probabilities=np.array([1.0]),
        jackpot_token_probability=1.0,
        jackpot_type_probabilities=np.array([1.0, 0.0, 0.0, 0.0]),
        jackpot_collection_targets=np.array([1, 3, 3, 3], dtype=np.int16),
        max_jackpot_tokens_per_respin=1,
    )
    return FULL_GAME_CONFIG._replace(
        base_jackpot_overlay=overlay,
        free_game=feature,
    )


def test_starting_bags_preserve_every_duplicate_in_board_order():
    board = np.array(
        [
            [REEL_DICT["SC2"], REEL_DICT["H1"], REEL_DICT["SC2"]],
            [REEL_DICT["SC5"], REEL_DICT["SC2"], REEL_DICT["H2"]],
        ],
        dtype=np.int16,
    )

    result = extract_starting_bag_symbols(
        board,
        FULL_GAME_CONFIG.base_game.free_game_symbols,
    )

    np.testing.assert_array_equal(
        result,
        np.array(
            [
                REEL_DICT["SC2"],
                REEL_DICT["SC2"],
                REEL_DICT["SC5"],
                REEL_DICT["SC2"],
            ],
            dtype=np.int16,
        ),
    )


def test_full_round_links_feature_pays_incremented_jackpot_and_resets():
    config = _certain_jackpot_config()
    reels = _single_stop_reels(
        [
            REEL_DICT["SC6"],
            REEL_DICT["H1"],
            REEL_DICT["H2"],
            REEL_DICT["H3"],
            REEL_DICT["H4"],
        ]
    )
    base, free, full = _new_storages(1, config)
    jackpot_values = JACKPOT_CONFIG.seed_values.copy()

    result = run_one_full_round(
        reels,
        config,
        jackpot_values,
        base,
        free,
        full,
    )

    expected_award = (
        JACKPOT_CONFIG.seed_values[0] + JACKPOT_CONFIG.increment_values[0]
    )
    assert result[5] == 1
    assert full.spin_feature_session_indices[0] == 0
    assert full.spin_jackpot_awards[0, 0]
    assert full.spin_jackpot_award_amounts[0, 0] == expected_award
    assert full.spin_jackpot_wins[0] == expected_award
    assert jackpot_values[0] == JACKPOT_CONFIG.seed_values[0]
    np.testing.assert_array_equal(
        free.session_starting_symbols[0, :3],
        np.full(3, REEL_DICT["SC6"], dtype=np.int16),
    )


def test_full_game_serialization_statistics_and_pretty_print(
    tmp_path,
    monkeypatch,
):
    config = _certain_jackpot_config()
    reels = _single_stop_reels(
        [
            REEL_DICT["SC6"],
            REEL_DICT["H1"],
            REEL_DICT["H2"],
            REEL_DICT["H3"],
            REEL_DICT["H4"],
        ]
    )
    base, free, full = _new_storages(1, config)
    run_one_full_round(
        reels,
        config,
        JACKPOT_CONFIG.seed_values.copy(),
        base,
        free,
        full,
    )

    payload = full_game_storage_to_dict(full, base, free)
    assert payload["storage_type"] == "full_game"
    assert payload["rounds"][0]["spins"][0]["feature_session"] is not None
    pretty_output = format_payload(payload)
    assert "FULL GAME" in pretty_output
    assert "HOLD AND SPIN FEATURE" in pretty_output
    assert "Starting Bags: SC6, SC6, SC6" in pretty_output
    assert "Initial window" in pretty_output
    assert "Hold-and-Spin window:" in pretty_output

    monkeypatch.setattr(storage_module, "NPZ_LIBRARY_DIR", tmp_path)
    output_path = storage_module.write_full_game_npz(full, "full_game.npz")
    statistics = store_full_game(output_path, print_result=False)

    assert statistics.feature_session_count == 1
    assert statistics.jackpot_award_counts[0] == 1
    assert statistics.total_jackpot_win == full.spin_jackpot_wins[0]
    report = format_full_game_statistics(statistics)
    assert "FULL GAME STATISTICS" in report
    assert "JACKPOT AWARDS BY TYPE" in report
