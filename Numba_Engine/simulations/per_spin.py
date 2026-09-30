import numpy as np

from ..core.config import (
    BASE_GAME_CONFIG,
    FULL_GAME_CONFIG,
    HOLD_AND_SPIN_CONFIG,
    JACKPOT_CONFIG,
    PAY_LINES,
)
from ..core.reels import make_reel_collection
from ..core.storage import HoldAndSpinStorage, Storage
from ..serialization import append_json_history, pretty_print
from ..serialization.json_output import (
    full_game_storage_to_dict,
    storage_to_dict,
    write_json,
    write_json_payload,
)
from .base_game import run_one_spin as _run_one_base_spin
from .free_game import hold_and_free_spin, select_starting_bag_symbols
from .full_game import _new_storages, run_one_full_round


def _finish(result, storage, output_filename, print_result, overwrite):
    payload = storage_to_dict(storage)
    if print_result:
        pretty_print(payload, empty_marker="**")

    output_path = None
    if output_filename is not None:
        output_path = write_json(
            storage,
            output_filename,
            overwrite=overwrite,
        )
    return result, storage, payload, output_path


def run_base_spin(
    output_filename=None,
    print_result=True,
    overwrite=False,
):
    """Run one base round, including any walking-Collector paid spins."""
    rules = BASE_GAME_CONFIG
    reels = make_reel_collection(rules.reelset_path, rules)
    storage = Storage(
        1,
        rules.num_rows,
        rules.num_reels,
        rules.num_paying_symbols,
        len(PAY_LINES),
        len(JACKPOT_CONFIG.jackpot_types),
    )
    jackpot_values = JACKPOT_CONFIG.seed_values.copy()
    result = _run_one_base_spin(reels, rules, storage, jackpot_values)
    return _finish(
        result,
        storage,
        output_filename,
        print_result,
        overwrite,
    )


def run_hold_and_spin(
    starting_bag_symbols=None,
    output_filename=None,
    print_result=True,
    overwrite=False,
):
    """Run one Hold-and-Spin session with supplied or randomized features."""
    rules = HOLD_AND_SPIN_CONFIG
    if starting_bag_symbols is None:
        starting_bag_symbols = select_starting_bag_symbols(rules)
    else:
        starting_bag_symbols = np.asarray(
            starting_bag_symbols,
            dtype=np.int16,
        )
        if starting_bag_symbols.ndim != 1:
            raise ValueError("starting_bag_symbols must be one-dimensional")

    storage = HoldAndSpinStorage(
        1,
        1,
        1,
        rules.num_rows,
        rules.num_reels,
        len(rules.jackpot_collection_targets),
    )
    result = hold_and_free_spin(starting_bag_symbols, rules, storage)
    return _finish(
        result,
        storage,
        output_filename,
        print_result,
        overwrite,
    )


def run_full_game(
    jackpot_values=None,
    output_filename=None,
    print_result=True,
    overwrite=False,
):
    """Run one complete base round and any features it triggers.

    A supplied ``jackpot_values`` array is updated in place, allowing an
    interactive or external caller to preserve progressive state across plays.
    """
    config = FULL_GAME_CONFIG
    reels = make_reel_collection(
        config.base_game.reelset_path,
        config.base_game,
    )
    base_storage, full_storage = _new_storages(1, config)
    if jackpot_values is None:
        jackpot_values = config.jackpots.seed_values.copy()
    else:
        jackpot_values = np.asarray(jackpot_values, dtype=np.float64)
        if jackpot_values.shape != config.jackpots.seed_values.shape:
            raise ValueError("jackpot_values must contain one value per jackpot")

    result = run_one_full_round(
        reels,
        config,
        jackpot_values,
        base_storage,
        full_storage,
    )
    payload = full_game_storage_to_dict(full_storage, base_storage)
    if print_result:
        pretty_print(payload, empty_marker="**")

    output_path = None
    if output_filename is not None:
        output_path = write_json_payload(
            payload,
            output_filename,
            overwrite=overwrite,
        )
    return (
        result,
        base_storage,
        full_storage,
        jackpot_values,
        payload,
        output_path,
    )


def play_full_game_session(input_function=input):
    """Play full-game rounds while preserving one session's jackpots."""
    jackpot_values = FULL_GAME_CONFIG.jackpots.seed_values.copy()
    rounds_played = 0
    jackpot_names = ("Mini", "Minor", "Major", "Grand")
    prompt = (
        "Press Enter (or s) for the next full-game round, "
        "or q to end the session: "
    )
    print(
        "Full-game session started. Jackpot values will persist between "
        "rounds.",
        flush=True,
    )
    while True:
        try:
            command = input_function(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if command == "q":
            break
        if command not in ("", "s"):
            print("Unknown command. Use Enter, s, or q.")
            continue

        print(
            f"Running full-game round {rounds_played + 1}... "
            "The first round may take a while while Numba compiles.",
            flush=True,
        )
        (
            _,
            _,
            _,
            jackpot_values,
            payload,
            _,
        ) = run_full_game(
            jackpot_values=jackpot_values,
            print_result=True,
        )
        rounds_played += 1
        current_values = ", ".join(
            f"{name}={value:g}x"
            for name, value in zip(jackpot_names, jackpot_values)
        )
        print(
            f"Round {rounds_played} complete. Current jackpots: "
            f"{current_values}",
            flush=True,
        )
        history_path = append_json_history(
            payload,
            "full_gameplay_history.json",
            "full_game",
        )
        print(f"Saved to {history_path}")

    # A new full-game session starts from the configured seeds.
    jackpot_values[:] = FULL_GAME_CONFIG.jackpots.seed_values
    print(
        f"Full-game session ended after {rounds_played} round(s). "
        "Jackpots reset to their seeds.",
        flush=True,
    )
    return rounds_played, jackpot_values


def play_spin_by_spin(input_function=input):
    """Interactively play and persist base, feature, or full-game results."""
    prompt = (
        "Press b for a base spin, h for Hold-and-Spin, f for full game, "
        "or q to exit: "
    )
    while True:
        try:
            command = input_function(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if command == "q":
            print("Exiting spin-by-spin mode.", flush=True)
            break
        if command == "b":
            print(
                "Running base-game round... The first round may compile.",
                flush=True,
            )
            _, _, payload, _ = run_base_spin(print_result=True)
            history_path = append_json_history(
                payload,
                "base_gameplay_history.json",
                "base_game",
            )
            print(f"Saved to {history_path}")
        elif command == "h":
            print(
                "Running Hold-and-Spin session... The first session may "
                "compile.",
                flush=True,
            )
            _, _, payload, _ = run_hold_and_spin(print_result=True)
            history_path = append_json_history(
                payload,
                "hold_and_spin_gameplay_history.json",
                "hold_and_spin",
            )
            print(f"Saved to {history_path}")
        elif command == "f":
            play_full_game_session(input_function=input_function)
        else:
            print("Unknown command. Use b, h, f, or q.")


# User-facing feature terminology alias.
run_free_spin = run_hold_and_spin
run_full_spin = run_full_game


if __name__ == "__main__":
    play_spin_by_spin()
