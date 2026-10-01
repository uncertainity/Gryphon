import numpy as np

from Numba_Engine import HOLD_AND_SPIN_CONFIG, hold_and_spin
from Numba_Engine.core import storage as storage_module
from Numba_Engine.core.storage import HoldAndSpinStorage
from Numba_Engine.output.statistics import store_hold_and_spin
from Numba_Engine.serialization.json_output import storage_to_dict
from Numba_Engine.serialization.pretty_print import (
    format_hold_and_spin_statistics,
    format_payload,
)


def _jackpot_certain_config():
    return HOLD_AND_SPIN_CONFIG._replace(
        p_coin_locked=1.0,
        p_coin_unlocked=1.0,
        p_coin_locked_by_feature=np.full(8, -1.0),
        p_coin_unlocked_by_feature=np.full(8, -1.0),
        coin_types=np.array([HOLD_AND_SPIN_CONFIG.coin_types[0]], dtype=np.int16),
        coin_type_probabilities=np.array([1.0]),
        jackpot_token_probability=1.0,
        jackpot_type_probabilities=np.array([1.0, 0.0, 0.0, 0.0]),
        jackpot_collection_targets=np.array([1, 3, 3, 3], dtype=np.int16),
        max_jackpot_tokens_per_respin=1,
    )


def _storage(config):
    return HoldAndSpinStorage(
        2,
        1,
        1,
        config.num_rows,
        config.num_reels,
        len(config.jackpot_collection_targets),
    )


def test_free_game_stores_session_meter_and_awarded_type():
    config = _jackpot_certain_config()
    storage = _storage(config)

    result = hold_and_spin(
        np.array([config.splitter_symbol], dtype=np.int16),
        config,
        storage,
    )

    assert result.jackpot_meters[0] == 1
    assert result.awarded_jackpots[0]
    assert storage.session_jackpot_meters[0, 0] == 1
    assert storage.session_jackpot_awards[0, 0]
    assert np.count_nonzero(storage.respin_jackpot_overlay_boards[0] >= 0) == 1

    payload = storage_to_dict(storage)
    report = format_payload(payload)
    assert "Jackpot types awarded: Mini" in report
    assert "Mini 0->1" in report


def test_jackpot_meters_restart_for_each_free_game_session():
    config = _jackpot_certain_config()
    storage = _storage(config)
    starting_symbols = np.array([config.splitter_symbol], dtype=np.int16)

    hold_and_spin(starting_symbols, config, storage)
    hold_and_spin(starting_symbols, config, storage)

    assert storage.session_count == 2
    np.testing.assert_array_equal(
        storage.session_jackpot_meters[:2, 0],
        np.array([1, 1]),
    )
    np.testing.assert_array_equal(
        storage.session_jackpot_awards[:2, 0],
        np.array([True, True]),
    )


def test_hold_and_spin_statistics_report_jackpot_collections(
    tmp_path,
    monkeypatch,
):
    config = _jackpot_certain_config()
    storage = _storage(config)
    hold_and_spin(
        np.array([config.splitter_symbol], dtype=np.int16),
        config,
        storage,
    )
    monkeypatch.setattr(storage_module, "NPZ_LIBRARY_DIR", tmp_path)
    output_path = storage_module.write_hold_and_spin_npz(
        storage,
        "free_game_jackpots.npz",
    )

    statistics = store_hold_and_spin(output_path, print_result=False)

    assert statistics.total_jackpot_tokens == 1
    assert statistics.jackpot_token_respin_count == 1
    assert statistics.jackpot_award_session_count == 1
    np.testing.assert_array_equal(
        statistics.jackpot_token_counts,
        np.array([1, 0, 0, 0]),
    )
    np.testing.assert_array_equal(
        statistics.jackpot_award_counts,
        np.array([1, 0, 0, 0]),
    )

    report = format_hold_and_spin_statistics(statistics)
    assert "JACKPOT COLLECTIONS BY TYPE" in report
    assert "Mini" in report


def test_plain_route_uses_three_by_five_payload_and_zero_starting_bags(
    tmp_path,
    monkeypatch,
):
    config = HOLD_AND_SPIN_CONFIG._replace(
        p_coin_locked=0.0,
        p_coin_unlocked=0.0,
        p_coin_locked_by_feature=np.full(8, -1.0),
        p_coin_unlocked_by_feature=np.full(8, -1.0),
        starting_coin_counts=np.array([1], dtype=np.int8),
        starting_coin_count_probabilities=np.array([1.0]),
    )
    storage = _storage(config)

    hold_and_spin(np.empty(0, dtype=np.int16), config, storage)

    payload = storage_to_dict(storage)
    session = payload["sessions"][0]
    assert session["feature_route"] == "Plain"
    assert session["logical_board_shape"] == [3, 5]
    assert session["starting_symbols"] == []
    assert len(session["respins"][0]["steps"][0]["board"]) == 3
    assert session["respins"][0]["steps"][0]["locked_row_index"] == -1

    monkeypatch.setattr(storage_module, "NPZ_LIBRARY_DIR", tmp_path)
    output_path = storage_module.write_hold_and_spin_npz(
        storage,
        "plain_hold_and_spin.npz",
    )
    statistics = store_hold_and_spin(output_path, print_result=False)
    assert statistics.session_count == 1
    assert statistics.average_starting_symbols == 0.0
