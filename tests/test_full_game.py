import numpy as np

from Numba_Engine import (
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
from Numba_Engine.simulations import full_game as full_game_module
from Numba_Engine.simulations.full_game import (
    _new_storages,
    extract_starting_bag_symbols,
    run_full_rounds_parallel,
    run_one_full_round,
    run_sims,
    select_feature_route,
    starting_bags_for_route,
)


def _single_stop_reels(symbols):
    return ReelCollections(
        lengths=np.ones((1, 5), dtype=np.int16),
        reelsets=np.array([[symbols]], dtype=np.int16),
        weights=np.array([1.0]),
    )


def test_full_game_executes_and_reports_shared_hold_and_spin(
    tmp_path, monkeypatch
):
    base_rules = FULL_GAME_CONFIG.base_game._replace(
        scatter_feature_symbol_probabilities=np.array(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
    )
    hold_rules = FULL_GAME_CONFIG.hold_and_spin._replace(
        plain_route_weight=0.0,
        single_route_weights=np.array(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        ),
    )
    config = FULL_GAME_CONFIG._replace(
        base_game=base_rules,
        hold_and_spin=hold_rules,
    )
    reels = _single_stop_reels(
        [
            REEL_DICT["SC"],
            REEL_DICT["H1"],
            REEL_DICT["H2"],
            REEL_DICT["H3"],
            REEL_DICT["H4"],
        ]
    )
    base, hold_and_spin, full = _new_storages(1, config)
    result = run_one_full_round(
        reels,
        config,
        JACKPOT_CONFIG.seed_values.copy(),
        base,
        hold_and_spin,
        full,
    )

    assert result[5] == 1
    assert hold_and_spin.session_count == 1
    assert hold_and_spin.session_starting_symbol_counts[0] == 1
    np.testing.assert_array_equal(
        hold_and_spin.session_starting_symbols[0, :1],
        np.array([REEL_DICT["SC1"]], dtype=np.int16),
    )
    assert hold_and_spin.boards.shape[1:] == (6, 5)
    assert hold_and_spin.step_locked_row_indices[0] == 2
    assert not np.any(hold_and_spin.boards[: hold_and_spin.step_count, :3])
    assert full.feature_session_count == 1
    assert full.feature_trigger_counts[0] == 1
    # Three respins are the counter value, but a full active area can end the
    # feature earlier than three executed respins.
    assert full.feature_spin_counts[0] >= 1
    assert full.feature_win_amounts[0] >= 0.0
    assert full.spin_feature_masks[0] == 1

    payload = full_game_storage_to_dict(full, base, hold_and_spin)
    assert payload["storage_type"] == "full_game"
    assert payload["rounds"][0]["spins"][0]["feature_session"] is not None
    assert payload["respin_count"] == hold_and_spin.respin_count
    assert payload["rounds"][0]["spins"][0]["feature_routes"] == [
        "Splitter"
    ]
    feature_session = payload["rounds"][0]["spins"][0]["feature_session"]
    assert feature_session["feature_route"] == "Splitter"
    assert feature_session["logical_board_shape"] == [3, 5]
    assert len(feature_session["respins"][0]["steps"][0]["board"]) == 3
    assert "Routed features: Splitter" in format_payload(payload)

    monkeypatch.setattr(storage_module, "NPZ_LIBRARY_DIR", tmp_path)
    output_path = storage_module.write_full_game_npz(
        full, "routed_full_game.npz"
    )
    statistics = store_full_game(output_path, print_result=False)
    assert statistics.feature_session_count == 1
    assert statistics.feature_spin_count == 1
    assert statistics.feature_trigger_counts[0] == 1
    assert statistics.feature_trigger_rates[0] == 1.0
    assert statistics.feature_average_spins[0] >= 1.0
    assert statistics.feature_average_wins[0] >= 0.0
    assert statistics.feature_rtp_by_type[0] == statistics.feature_rtp
    assert np.isclose(
        statistics.total_base_line_win + statistics.total_base_collect_win,
        statistics.total_base_win,
    )
    assert np.isclose(
        statistics.base_line_rtp + statistics.base_collect_rtp,
        statistics.base_rtp,
    )
    assert np.isclose(
        statistics.feature_trigger_rates.sum(),
        statistics.feature_trigger_rate,
    )
    assert np.isclose(
        statistics.jackpot_rtp_by_type.sum(),
        statistics.jackpot_rtp,
    )
    report = format_full_game_statistics(statistics)
    assert "FEATURE RESULTS BY TYPE" in report
    assert "Splitter" in report
    assert "Mega Combo" in report
    assert "Plain" in report
    assert "Payline win / RTP" in report
    assert "Base Collect / RTP" in report
    assert "Trigger Rate" in report
    assert "Respins" in report
    assert "Avg Respins" in report
    assert "Avg Win" in report
    assert "Award Rate" in report


def test_partial_bag_set_selects_plain_or_one_visible_single_route():
    pay_window = np.zeros((3, 5), dtype=np.int16)
    pay_window[0, 0] = REEL_DICT["SC1"]
    pay_window[1, 2] = REEL_DICT["SC2"]
    pay_window[2, 4] = REEL_DICT["SC1"]

    starting_bags = extract_starting_bag_symbols(
        pay_window,
        FULL_GAME_CONFIG.hold_and_spin.bag_symbols,
    )

    np.testing.assert_array_equal(
        starting_bags,
        np.array(
            [REEL_DICT["SC1"], REEL_DICT["SC2"], REEL_DICT["SC1"]],
            dtype=np.int16,
        ),
    )
    feature_flags = np.array(
        [True, True, False, False, False, False],
        dtype=np.bool_,
    )
    sc2_rules = FULL_GAME_CONFIG.hold_and_spin._replace(
        plain_route_weight=0.0,
        single_route_weights=np.array(
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        ),
    )
    feature_index = select_feature_route(feature_flags, False, sc2_rules)
    assert feature_index == 1
    np.testing.assert_array_equal(
        starting_bags_for_route(feature_index, sc2_rules.bag_symbols),
        np.array([REEL_DICT["SC2"]], dtype=np.int16),
    )

    plain_rules = sc2_rules._replace(
        plain_route_weight=1.0,
        single_route_weights=np.zeros(6, dtype=np.float64),
    )
    feature_index = select_feature_route(feature_flags, False, plain_rules)
    assert feature_index == 7
    assert len(
        starting_bags_for_route(feature_index, plain_rules.bag_symbols)
    ) == 0


def test_all_six_visible_bags_route_to_one_mega_combo_session():
    feature_flags = np.ones(6, dtype=np.bool_)
    rules = FULL_GAME_CONFIG.hold_and_spin

    feature_index = select_feature_route(feature_flags, True, rules)

    assert feature_index == 6
    np.testing.assert_array_equal(
        starting_bags_for_route(feature_index, rules.bag_symbols),
        rules.bag_symbols,
    )


def test_parallel_full_game_merges_worker_storage_and_statistics(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(storage_module, "NPZ_LIBRARY_DIR", tmp_path)

    result, base, hold_and_spin, full, output_path, statistics = run_sims(
        num_rounds=8,
        num_workers=2,
        use_parallel=True,
        output_filename="parallel_full_game.npz",
        overwrite=True,
        print_statistics=False,
        seed=1234,
    )

    assert len(result) == 6
    assert base.round_count == 8
    assert full.round_count == 8
    assert hold_and_spin.session_count == full.feature_session_count
    assert base.spin_count == full.spin_count == statistics.spin_count
    assert full.round_spin_offsets[0] == 0
    assert full.round_spin_offsets[full.round_count] == full.spin_count
    assert statistics.round_count == 8
    assert statistics.feature_session_count == int(
        full.feature_trigger_counts.sum()
    )
    assert output_path.is_file()
    assert run_full_rounds_parallel.nopython_signatures


def test_compact_full_game_sizes_feature_history_from_trigger_rate():
    _, hold_and_spin, full = _new_storages(
        10_000,
        FULL_GAME_CONFIG,
        compact_base=True,
    )

    assert hold_and_spin.session_wins.shape[0] == 257
    assert hold_and_spin.session_wins.shape[0] < full.spin_base_wins.shape[0]


def test_full_game_cli_maps_arguments_to_run_sims(monkeypatch):
    observed = {}
    sentinel = object()

    def fake_run_sims(**kwargs):
        observed.update(kwargs)
        return sentinel

    monkeypatch.setattr(full_game_module, "run_sims", fake_run_sims)

    result = full_game_module.main(
        [
            "--num-spins",
            "10m",
            "--num-workers",
            "3",
            "--parallel",
            "--compact-storage",
            "--output-path",
            "cli_full_game.npz",
            "--overwrite",
            "--bet-per-spin",
            "2.5",
            "--seed",
            "1234",
            "--no-statistics",
        ]
    )

    assert result is sentinel
    assert observed == {
        "num_rounds": 10_000_000,
        "num_workers": 3,
        "use_parallel": True,
        "compact_storage": True,
        "output_filename": "cli_full_game.npz",
        "overwrite": True,
        "bet_per_spin": 2.5,
        "print_statistics": False,
        "seed": 1234,
    }
