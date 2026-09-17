import numpy as np

from ..core.config import BASE_GAME_CONFIG, HOLD_AND_SPIN_CONFIG, PAY_LINES
from ..core.reels import make_reel_collection
from ..core.storage import HoldAndSpinStorage, Storage
from ..serialization import append_json_history, pretty_print
from ..serialization.json_output import storage_to_dict, write_json
from .base_game import run_one_spin as _run_one_base_spin
from .free_game import hold_and_free_spin, select_starting_bag_symbols


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
    )
    result = _run_one_base_spin(reels, rules, storage)
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
    )
    result = hold_and_free_spin(starting_bag_symbols, rules, storage)
    return _finish(
        result,
        storage,
        output_filename,
        print_result,
        overwrite,
    )


def play_spin_by_spin(input_function=input):
    """Interactively play and persist base or Hold-and-Spin results."""
    prompt = (
        "Press b for a base spin, h for Hold-and-Spin, "
        "or q to exit: "
    )
    while True:
        try:
            command = input_function(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if command == "q":
            break
        if command == "b":
            _, _, payload, _ = run_base_spin(print_result=True)
            history_path = append_json_history(
                payload,
                "base_gameplay_history.json",
                "base_game",
            )
            print(f"Saved to {history_path}")
        elif command == "h":
            _, _, payload, _ = run_hold_and_spin(print_result=True)
            history_path = append_json_history(
                payload,
                "hold_and_spin_gameplay_history.json",
                "hold_and_spin",
            )
            print(f"Saved to {history_path}")
        else:
            print("Unknown command. Use b, h, or q.")


# User-facing feature terminology alias.
run_free_spin = run_hold_and_spin


if __name__ == "__main__":
    play_spin_by_spin()
