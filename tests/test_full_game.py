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
from Numba_Engine.simulations.full_game import (
    _new_storages,
    run_full_rounds_parallel,
    run_one_full_round,
    run_sims,
)


def _single_stop_reels(symbols):
    return ReelCollections(
        lengths=np.ones((1, 5), dtype=np.int16),
        reelsets=np.array([[symbols]], dtype=np.int16),
        weights=np.array([1.0]),
    )


def test_full_game_executes_and_reports_branch_feature(tmp_path, monkeypatch):
    base_rules = FULL_GAME_CONFIG.base_game._replace(
        scatter_feature_symbol_probabilities=np.array(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
    )
    config = FULL_GAME_CONFIG._replace(base_game=base_rules)
    reels = _single_stop_reels(
        [
            REEL_DICT["SC"],
            REEL_DICT["H1"],
            REEL_DICT["H2"],
            REEL_DICT["H3"],
            REEL_DICT["H4"],
        ]
    )
    base, full = _new_storages(1, config)
    result = run_one_full_round(
        reels,
        config,
        JACKPOT_CONFIG.seed_values.copy(),
        base,
        full,
    )

    assert result[5] == 1
    assert full.feature_session_count == 1
    assert full.feature_trigger_counts[0] == 1
    assert full.feature_spin_counts[0] >= 3
    assert full.feature_win_amounts[0] >= 0.0
    assert full.spin_feature_masks[0] == 1

    payload = full_game_storage_to_dict(full, base)
    assert payload["storage_type"] == "full_game"
    assert payload["rounds"][0]["spins"][0]["feature_routes"] == [
        "Splitter"
    ]
    assert "Routed features: Splitter" in format_payload(payload)

    monkeypatch.setattr(storage_module, "NPZ_LIBRARY_DIR", tmp_path)
    output_path = storage_module.write_full_game_npz(
        full, "routed_full_game.npz"
    )
    statistics = store_full_game(output_path, print_result=False)
    assert statistics.feature_session_count == 1
    assert statistics.feature_spin_count == 1
    assert statistics.feature_trigger_counts[0] == 1
    report = format_full_game_statistics(statistics)
    assert "FEATURE RESULTS BY TYPE" in report
    assert "Splitter" in report


def test_parallel_full_game_merges_worker_storage_and_statistics(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(storage_module, "NPZ_LIBRARY_DIR", tmp_path)

    result, base, full, output_path, statistics = run_sims(
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
    assert base.spin_count == full.spin_count == statistics.spin_count
    assert full.round_spin_offsets[0] == 0
    assert full.round_spin_offsets[full.round_count] == full.spin_count
    assert statistics.round_count == 8
    assert statistics.feature_session_count == int(
        full.feature_trigger_counts.sum()
    )
    assert output_path.is_file()
    assert run_full_rounds_parallel.nopython_signatures
