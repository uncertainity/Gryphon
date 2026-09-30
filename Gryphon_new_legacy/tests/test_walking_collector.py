import numpy as np

from Numba_Engine import (
    BASE_GAME_CONFIG,
    JACKPOT_CONFIG,
    REEL_DICT,
    ReelCollections,
)
from Numba_Engine.core.config import PAY_LINES
from Numba_Engine.core.storage import Storage
from Numba_Engine.simulations.base_game import (
    move_collectors_to_next_spin,
    run_one_spin,
)


def _single_stop_reels(symbols):
    return ReelCollections(
        lengths=np.ones((1, 5), dtype=np.int16),
        reelsets=np.array([[symbols]], dtype=np.int16),
        weights=np.array([1.0]),
    )


def _new_storage():
    return Storage(
        1,
        BASE_GAME_CONFIG.num_rows,
        BASE_GAME_CONFIG.num_reels,
        BASE_GAME_CONFIG.num_paying_symbols,
        len(PAY_LINES),
        len(JACKPOT_CONFIG.jackpot_types),
    )


def test_move_collectors_uses_flattened_positions_and_compacts_exits():
    positions = np.empty(15, dtype=np.int32)
    positions[:3] = np.array([3, 4, 7])

    count = move_collectors_to_next_spin(positions, 3, num_reels=5)

    assert count == 2
    np.testing.assert_array_equal(positions[:count], np.array([4, 8]))


def test_round_without_collectors_is_exactly_one_paid_spin():
    reels = _single_stop_reels(
        [
            REEL_DICT["H1"],
            REEL_DICT["H2"],
            REEL_DICT["H3"],
            REEL_DICT["L1"],
            REEL_DICT["L2"],
        ]
    )

    total_win, free_game_triggers, paid_spins = run_one_spin(
        reels,
        BASE_GAME_CONFIG,
        _new_storage(),
        JACKPOT_CONFIG.seed_values.copy(),
    )

    assert total_win >= 0
    assert free_game_triggers == 0
    assert paid_spins == 1


def test_collector_on_final_reel_collects_then_exits_without_extra_spin():
    reels = _single_stop_reels(
        [
            REEL_DICT["COIN"],
            REEL_DICT["H1"],
            REEL_DICT["H2"],
            REEL_DICT["L1"],
            REEL_DICT["COLLECT"],
        ]
    )

    total_win, free_game_triggers, paid_spins = run_one_spin(
        reels,
        BASE_GAME_CONFIG,
        _new_storage(),
        JACKPOT_CONFIG.seed_values.copy(),
    )

    assert total_win > 0
    assert free_game_triggers == 0
    assert paid_spins == 1
