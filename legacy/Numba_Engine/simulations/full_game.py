from pathlib import Path

import numpy as np
from numba import njit

from ..core.config import FULL_GAME_CONFIG, PAY_LINES
from ..core.reels import make_reel_collection
from ..core.storage import FullGameStorage, HoldAndSpinStorage, Storage
from .base_game import run_one_paid_spin
from .free_game import hold_and_free_spin


@njit
def extract_starting_bag_symbols(pay_window, trigger_symbols):
    """Return every visible trigger occurrence in board order."""
    starting_bags = np.empty(pay_window.size, dtype=np.int16)
    count = 0
    for symbol in pay_window.ravel():
        for trigger_symbol in trigger_symbols:
            if symbol == trigger_symbol:
                starting_bags[count] = symbol
                count += 1
                break
    return starting_bags[:count]


def validate_full_game_config(config):
    """Validate the contracts shared across the composed game configs."""
    base_triggers = set(int(value) for value in config.base_game.free_game_symbols)
    free_bags = set(int(value) for value in config.free_game.bag_symbols)
    missing = sorted(base_triggers - free_bags)
    if missing:
        raise ValueError(
            "Every base free-game trigger must be a configured free-game Bag; "
            f"missing symbol IDs: {missing}"
        )
    num_jackpots = len(config.jackpots.jackpot_types)
    if len(config.free_game.jackpot_collection_targets) != num_jackpots:
        raise ValueError("Free-game meter count must match jackpot count")
    if len(config.free_game.jackpot_type_probabilities) != num_jackpots:
        raise ValueError("Free-game jackpot weights must match jackpot count")
    return config


@njit
def run_one_full_round(
    initial_reels,
    config,
    jackpot_values,
    base_storage,
    free_game_storage,
    full_game_storage,
):
    """Run one paid round, invoking each triggered feature immediately."""
    base_rules = config.base_game
    jackpot_rules = config.jackpots
    collector_positions = np.empty(
        base_rules.max_active_collectors,
        dtype=np.int32,
    )
    collector_count = 0
    round_base_win = 0.0
    round_feature_win = 0.0
    round_jackpot_win = 0.0
    round_triggers = 0
    paid_spins = 0
    base_storage.begin_round()
    full_game_storage.begin_round()

    while True:
        (
            base_win,
            feature_triggered,
            collector_count,
            pay_window,
        ) = run_one_paid_spin(
            initial_reels,
            base_rules,
            config.base_jackpot_overlay,
            jackpot_rules,
            base_storage,
            jackpot_values,
            collector_positions,
            collector_count,
        )

        values_before_feature = jackpot_values.copy()
        feature_session_index = -1
        feature_win = 0.0
        jackpot_win = 0.0
        jackpot_awards = np.zeros(
            len(jackpot_rules.jackpot_types),
            dtype=np.bool_,
        )
        jackpot_award_amounts = np.zeros(
            len(jackpot_rules.jackpot_types),
            dtype=np.float64,
        )

        if feature_triggered:
            starting_bags = extract_starting_bag_symbols(
                pay_window,
                base_rules.free_game_symbols,
            )
            if len(starting_bags) > 0:
                feature_session_index = free_game_storage.session_count
                feature_win, feature_result = hold_and_free_spin(
                    starting_bags,
                    config.free_game,
                    free_game_storage,
                )
                jackpot_awards = feature_result.awarded_jackpots.copy()
                for jackpot_type in range(len(jackpot_awards)):
                    if jackpot_awards[jackpot_type]:
                        award_amount = jackpot_values[jackpot_type]
                        jackpot_award_amounts[jackpot_type] = award_amount
                        jackpot_win += award_amount
                        jackpot_values[jackpot_type] = (
                            jackpot_rules.seed_values[jackpot_type]
                        )
                round_triggers += 1

        full_game_storage.save_spin(
            feature_session_index,
            base_win,
            feature_win,
            jackpot_win,
            values_before_feature,
            jackpot_awards,
            jackpot_award_amounts,
            jackpot_values,
        )
        round_base_win += base_win
        round_feature_win += feature_win
        round_jackpot_win += jackpot_win
        paid_spins += 1
        if collector_count == 0:
            break

    base_storage.finish_round(round_base_win, round_triggers)
    full_game_storage.finish_round(
        round_base_win,
        round_feature_win,
        round_jackpot_win,
    )
    return (
        round_base_win + round_feature_win + round_jackpot_win,
        round_base_win,
        round_feature_win,
        round_jackpot_win,
        paid_spins,
        round_triggers,
    )


@njit
def run_full_rounds(
    initial_reels,
    config,
    num_rounds,
    base_storage,
    free_game_storage,
    full_game_storage,
):
    """Run full-game rounds with one persistent jackpot state."""
    jackpot_values = config.jackpots.seed_values.copy()
    total_win = 0.0
    total_base_win = 0.0
    total_feature_win = 0.0
    total_jackpot_win = 0.0
    total_paid_spins = 0
    total_triggers = 0
    for _ in range(num_rounds):
        result = run_one_full_round(
            initial_reels,
            config,
            jackpot_values,
            base_storage,
            free_game_storage,
            full_game_storage,
        )
        total_win += result[0]
        total_base_win += result[1]
        total_feature_win += result[2]
        total_jackpot_win += result[3]
        total_paid_spins += result[4]
        total_triggers += result[5]
    return (
        total_win,
        total_base_win,
        total_feature_win,
        total_jackpot_win,
        total_paid_spins,
        total_triggers,
    )


def _new_storages(num_rounds, config):
    estimated_spins = max(1, num_rounds * 2)
    estimated_sessions = estimated_spins
    estimated_respins = max(
        1,
        estimated_sessions * config.free_game.respin_reset_count * 2,
    )
    estimated_steps = max(1, estimated_respins * 3)
    num_jackpots = len(config.jackpots.jackpot_types)
    base_storage = Storage(
        estimated_spins,
        config.base_game.num_rows,
        config.base_game.num_reels,
        config.base_game.num_paying_symbols,
        len(PAY_LINES),
        num_jackpots,
    )
    free_game_storage = HoldAndSpinStorage(
        estimated_steps,
        estimated_respins,
        estimated_sessions,
        config.free_game.num_rows,
        config.free_game.num_reels,
        num_jackpots,
    )
    full_game_storage = FullGameStorage(
        estimated_spins,
        max(1, num_rounds),
        num_jackpots,
    )
    return base_storage, free_game_storage, full_game_storage


def run_sims(
    num_rounds=10_000,
    config=FULL_GAME_CONFIG,
    output_filename="full_game_results.npz",
    overwrite=False,
    bet_per_spin=1.0,
    print_statistics=True,
):
    """Run, store and report combined base, feature and jackpot play."""
    if num_rounds < 1:
        raise ValueError("num_rounds must be positive")
    if bet_per_spin <= 0:
        raise ValueError("bet_per_spin must be positive")
    validate_full_game_config(config)
    output_filename = Path(output_filename)
    if output_filename.is_absolute() or len(output_filename.parts) != 1:
        raise ValueError("output_filename must not contain a directory path")
    if output_filename.suffix == "":
        output_filename = output_filename.with_suffix(".npz")
    elif output_filename.suffix.lower() != ".npz":
        raise ValueError("output_filename must use the .npz extension")

    initial_reels = make_reel_collection(
        config.base_game.reelset_path,
        config.base_game,
    )
    base_storage, free_storage, full_storage = _new_storages(
        num_rounds,
        config,
    )
    result = run_full_rounds(
        initial_reels,
        config,
        num_rounds,
        base_storage,
        free_storage,
        full_storage,
    )

    from ..core.storage import write_full_game_npz
    from ..output.statistics import store_full_game

    output_path = write_full_game_npz(
        full_storage,
        output_filename.name,
        overwrite=overwrite,
    )
    statistics = store_full_game(
        output_path,
        bet_per_spin=bet_per_spin,
        print_result=print_statistics,
    )
    return (
        result,
        base_storage,
        free_storage,
        full_storage,
        output_path,
        statistics,
    )


if __name__ == "__main__":
    run_sims()
