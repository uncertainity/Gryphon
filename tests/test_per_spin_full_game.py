import numpy as np

from Numba_Engine.core.config import FULL_GAME_CONFIG
from Numba_Engine.simulations import per_spin


def test_full_game_session_preserves_then_resets_jackpot_values(monkeypatch):
    observed_values = []
    saved_payloads = []

    def fake_run_full_game(jackpot_values, print_result):
        assert print_result
        observed_values.append(jackpot_values.copy())
        jackpot_values += 0.5
        return (
            (),
            None,
            None,
            jackpot_values,
            {"storage_type": "full_game"},
            None,
        )

    def fake_append(payload, filename, storage_type):
        saved_payloads.append((payload, filename, storage_type))
        return filename

    commands = iter(["", "s", "q"])
    monkeypatch.setattr(per_spin, "run_full_game", fake_run_full_game)
    monkeypatch.setattr(per_spin, "append_json_history", fake_append)

    rounds_played, final_values = per_spin.play_full_game_session(
        input_function=lambda _: next(commands)
    )

    assert rounds_played == 2
    np.testing.assert_allclose(
        observed_values[0],
        FULL_GAME_CONFIG.jackpots.seed_values,
    )
    np.testing.assert_allclose(
        observed_values[1],
        FULL_GAME_CONFIG.jackpots.seed_values + 0.5,
    )
    np.testing.assert_array_equal(
        final_values,
        FULL_GAME_CONFIG.jackpots.seed_values,
    )
    assert len(saved_payloads) == 2
