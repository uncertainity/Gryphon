import numpy as np
from pathlib import Path

from numba import njit, prange
from numba.typed import List

from ..core.config import (
    BASE_GAME_CONFIG,
    BASE_JACKPOT_OVERLAY_CONFIG,
    BASE_PAY_TABLE,
    JACKPOT_CONFIG,
    PAY_LINES,
)
from ..core.kernels import (
    apply_jackpot_overlay,
    has_free_game_trigger,
    line_win_eval,
    make_base_jackpot_overlay,
    make_board,
    probChoice,
    select_reelset_index,
)
from ..core.reels import make_reel_collection
from ..core.storage import Storage, merge_npz, write_npz


@njit
def move_collectors_to_next_spin(
    collector_positions,
    collector_count,
    num_reels,
):
    """Move Collectors one reel right and remove those leaving the window."""
    write_index = 0

    for read_index in range(collector_count):
        position = collector_positions[read_index]
        row = position // num_reels
        col = position % num_reels
        next_col = col + 1

        if next_col < num_reels:
            collector_positions[write_index] = row * num_reels + next_col
            write_index += 1

    return write_index


@njit
def assign_coin_credits(pay_window, rules):
    """Create a credit-value window without replacing Coin symbol IDs."""
    coin_value_window = np.zeros(pay_window.shape, dtype=np.float64)

    for row in range(pay_window.shape[0]):
        for col in range(pay_window.shape[1]):
            if pay_window[row, col] == rules.coin_symbol:
                coin_value_window[row, col] = probChoice(
                    rules.coin_credit_value_probabilities,
                    rules.coin_credit_values,
                )

    return coin_value_window


@njit
def drop_coin_credits(pay_window, collector_count, rules):
    """Apply the configured overlay-Coin drop before Collect resolves."""
    coin_value_window = assign_coin_credits(pay_window, rules)
    available_positions = np.empty(pay_window.size, dtype=np.int32)
    num_available = 0

    for row in range(pay_window.shape[0]):
        for col in range(pay_window.shape[1]):
            symbol = pay_window[row, col]
            if (
                symbol == rules.wild_symbol
                or symbol == rules.collect_symbol
                or symbol == rules.coin_symbol
                or symbol == rules.sc_symbol
                or coin_value_window[row, col] > 0.0
            ):
                continue
            available_positions[num_available] = row * pay_window.shape[1] + col
            num_available += 1

    if num_available == 0:
        return coin_value_window

    probability_index = 2
    if collector_count == 0:
        probability_index = 0
    elif collector_count == 1:
        probability_index = 1
    if (
        np.random.uniform(0.0, 1.0)
        >= rules.coin_drop_probabilities[probability_index]
    ):
        return coin_value_window

    drop_count = int(
        probChoice(rules.coin_drop_count_probabilities, rules.coin_drop_counts)
    )
    drop_count = min(drop_count, num_available)
    for _ in range(drop_count):
        available_index = np.random.randint(0, num_available)
        position = available_positions[available_index]
        row = position // pay_window.shape[1]
        col = position % pay_window.shape[1]
        coin_value_window[row, col] = probChoice(
            rules.coin_credit_value_probabilities,
            rules.coin_credit_values,
        )
        num_available -= 1
        available_positions[available_index] = available_positions[
            num_available
        ]
        if num_available == 0:
            break
    return coin_value_window


@njit
def convert_base_scatters(pay_window, rules):
    """Resolve generic SC symbols through the configured feature routing."""
    converted_window = pay_window.copy()
    feature_flags = np.zeros(
        len(rules.scatter_feature_symbols),
        dtype=np.bool_,
    )

    for row in range(converted_window.shape[0]):
        for col in range(converted_window.shape[1]):
            if converted_window[row, col] != rules.sc_symbol:
                continue
            feature_symbol = int(
                probChoice(
                    rules.scatter_feature_symbol_probabilities,
                    rules.scatter_feature_symbols,
                )
            )
            converted_window[row, col] = feature_symbol
            for index in range(len(rules.scatter_feature_symbols)):
                if rules.scatter_feature_symbols[index] == feature_symbol:
                    feature_flags[index] = True
                    break

    combo_triggered = False
    has_all_combo_symbols = True
    for combo_symbol in rules.combo_feature_symbols:
        if not np.any(converted_window == combo_symbol):
            has_all_combo_symbols = False
            break
    if has_all_combo_symbols:
        combo_triggered = (
            np.random.uniform(0.0, 1.0) < rules.combo_feature_probability
        )
        if combo_triggered:
            feature_flags[-1] = True
    return converted_window, feature_flags, combo_triggered


@njit
def _run_one_paid_spin_with_features(
    initial_reels,
    rules,
    overlay_rules,
    jackpot_rules,
    storage,
    jackpot_values,
    collector_positions,
    collector_count,
):
    """Resolve one paid spin and return its continuation state."""
    reelset_index = select_reelset_index(initial_reels.weights)
    pay_window = make_board(
        initial_reels.lengths[reelset_index],
        initial_reels.reelsets[reelset_index],
        rules,
    )
    natural_collector_positions = np.where(
        pay_window.ravel() == rules.collect_symbol
    )[0]

    for idx in range(collector_count):
        position = collector_positions[idx]
        row = position // rules.num_reels
        col = position % rules.num_reels
        pay_window[row, col] = rules.collect_symbol

    for position in natural_collector_positions:
        position_is_occupied = False
        for idx in range(collector_count):
            if collector_positions[idx] == position:
                position_is_occupied = True
                break
        if position_is_occupied:
            continue
        if collector_count == len(collector_positions):
            break
        collector_positions[collector_count] = position
        collector_count += 1

    (
        line_win,
        line_winning_symbols,
        line_match_counts,
        line_win_amounts,
        winning_window,
        hit_counts,
        symbol_win_amounts,
    ) = line_win_eval(
        pay_window,
        BASE_PAY_TABLE,
        PAY_LINES,
        rules.wild_symbol,
    )

    coin_value_window = drop_coin_credits(pay_window, collector_count, rules)
    jackpot_overlay_window = make_base_jackpot_overlay(
        pay_window,
        overlay_rules,
        jackpot_rules.jackpot_types,
    )
    jackpot_values_before, jackpot_increment_counts = apply_jackpot_overlay(
        jackpot_overlay_window,
        jackpot_rules,
        jackpot_values,
    )
    visible_coin_total = np.sum(coin_value_window)
    collect_win = visible_coin_total * collector_count
    free_game_trigger = has_free_game_trigger(pay_window, rules)
    display_window = pay_window
    feature_flags = np.zeros(
        len(rules.scatter_feature_symbols), dtype=np.bool_
    )
    combo_triggered = False
    if free_game_trigger:
        display_window, feature_flags, combo_triggered = convert_base_scatters(
            pay_window, rules
        )

    storage.save_spin(
        display_window,
        coin_value_window,
        jackpot_overlay_window,
        jackpot_values_before,
        jackpot_values,
        jackpot_increment_counts,
        line_win,
        collect_win,
        symbol_win_amounts,
        hit_counts,
        line_winning_symbols,
        line_match_counts,
        line_win_amounts,
        free_game_trigger,
        collector_count,
    )

    collector_count = move_collectors_to_next_spin(
        collector_positions,
        collector_count,
        rules.num_reels,
    )
    return (
        line_win + collect_win,
        free_game_trigger,
        collector_count,
        display_window,
        feature_flags,
        combo_triggered,
    )


@njit
def run_one_paid_spin(
    initial_reels,
    rules,
    overlay_rules,
    jackpot_rules,
    storage,
    jackpot_values,
    collector_positions,
    collector_count,
):
    """Compatibility wrapper returning the original four-value contract."""
    (
        spin_win,
        feature_triggered,
        collector_count,
        pay_window,
        _,
        _,
    ) = _run_one_paid_spin_with_features(
        initial_reels,
        rules,
        overlay_rules,
        jackpot_rules,
        storage,
        jackpot_values,
        collector_positions,
        collector_count,
    )
    return spin_win, feature_triggered, collector_count, pay_window


@njit
def run_one_spin(initial_reels, rules, storage, jackpot_values):
    """Run connected paid spins until all walking Collectors have exited."""
    collector_positions = np.empty(
        rules.max_active_collectors,
        dtype=np.int32,
    )
    collector_count = 0
    num_paid_spins = 0
    total_win = 0.0
    free_game_triggers = 0
    storage.begin_round()

    while True:
        spin_win, free_game_trigger, collector_count, pay_window = (
            run_one_paid_spin(
                initial_reels,
                rules,
                BASE_JACKPOT_OVERLAY_CONFIG,
                JACKPOT_CONFIG,
                storage,
                jackpot_values,
                collector_positions,
                collector_count,
            )
        )
        total_win += spin_win
        free_game_triggers += int(free_game_trigger)
        num_paid_spins += 1
        if collector_count == 0:
            break

    storage.finish_round(total_win, free_game_triggers)
    return total_win, free_game_triggers, num_paid_spins


@njit
def run_spins(initial_reels, rules, num_rounds, storage):
    """Run independent starting rounds and count all paid continuation spins."""
    total_win = 0.0
    free_game_triggers = 0
    total_paid_spins = 0
    jackpot_values = JACKPOT_CONFIG.seed_values.copy()

    for _ in range(num_rounds):
        round_win, round_triggers, round_spins = run_one_spin(
            initial_reels,
            rules,
            storage,
            jackpot_values,
        )
        total_win += round_win
        free_game_triggers += round_triggers
        total_paid_spins += round_spins

    return total_win, free_game_triggers, total_paid_spins

@njit(parallel=True)
def run_spins_parallel(
    initial_reels,
    rules,
    rounds_per_worker,
    worker_storages,
):
    """Run independent worker batches without sharing Collector chains."""
    total_win = 0.0
    free_game_triggers = 0
    total_paid_spins = 0

    for worker_index in prange(len(worker_storages)):
        storage_index = np.int64(worker_index)
        worker_win, worker_triggers, worker_spins = run_spins(
            initial_reels,
            rules,
            rounds_per_worker[storage_index],
            worker_storages[storage_index],
        )
        total_win += worker_win
        free_game_triggers += worker_triggers
        total_paid_spins += worker_spins

    return total_win, free_game_triggers, total_paid_spins


def run_sims(
    num_rounds=10_000,
    num_workers=3,
    use_parallel=False,
    output_filename="base_game_results.npz",
    overwrite=True,
    bet_per_spin=1.0,
    print_statistics=True,
):
    """Run, serialize, aggregate, and optionally report base-game results."""
    if num_rounds < 1:
        raise ValueError("num_rounds must be positive")
    if num_workers < 1:
        raise ValueError("num_workers must be positive")
    if bet_per_spin <= 0:
        raise ValueError("bet_per_spin must be positive")

    output_filename = Path(output_filename)
    if output_filename.is_absolute() or len(output_filename.parts) != 1:
        raise ValueError("output_filename must not contain a directory path")
    if output_filename.suffix == "":
        output_filename = output_filename.with_suffix(".npz")
    elif output_filename.suffix.lower() != ".npz":
        raise ValueError("output_filename must use the .npz extension")

    initial_reels = make_reel_collection(
        BASE_GAME_CONFIG.reelset_path,
        BASE_GAME_CONFIG,
    )

    if use_parallel:
        active_workers = min(num_workers, num_rounds)
        rounds_per_worker = np.full(
            active_workers,
            num_rounds // active_workers,
            dtype=np.int64,
        )
        rounds_per_worker[:num_rounds % active_workers] += 1

        worker_storages = List.empty_list(Storage.class_type.instance_type)
        for worker_index in range(active_workers):
            worker_rounds = int(rounds_per_worker[worker_index])
            worker_storages.append(
                Storage(
                    max(1, worker_rounds * 2),
                    BASE_GAME_CONFIG.num_rows,
                    BASE_GAME_CONFIG.num_reels,
                    BASE_GAME_CONFIG.num_paying_symbols,
                    len(PAY_LINES),
                    len(JACKPOT_CONFIG.jackpot_types),
                )
            )

        total_win, free_game_triggers, total_paid_spins = run_spins_parallel(
            initial_reels,
            BASE_GAME_CONFIG,
            rounds_per_worker,
            worker_storages,
        )

        shard_filenames = []
        for worker_index, worker_storage in enumerate(worker_storages):
            shard_filename = (
                f"{output_filename.stem}_worker_{worker_index}.npz"
            )
            write_npz(
                worker_storage,
                shard_filename,
                overwrite=overwrite,
            )
            shard_filenames.append(shard_filename)

        output_path = merge_npz(
            shard_filenames,
            output_filename.name,
            overwrite=overwrite,
        )
    else:
        storage = Storage(
            max(1, num_rounds * 2),
            BASE_GAME_CONFIG.num_rows,
            BASE_GAME_CONFIG.num_reels,
            BASE_GAME_CONFIG.num_paying_symbols,
            len(PAY_LINES),
            len(JACKPOT_CONFIG.jackpot_types),
        )
        total_win, free_game_triggers, total_paid_spins = run_spins(
            initial_reels,
            BASE_GAME_CONFIG,
            num_rounds,
            storage,
        )
        output_path = write_npz(
            storage,
            output_filename.name,
            overwrite=overwrite,
        )

    from ..output.statistics import store_base_game

    statistics = store_base_game(
        output_path,
        bet_per_spin=bet_per_spin,
        print_result=print_statistics,
    )

    return (
        total_win,
        free_game_triggers,
        total_paid_spins,
        output_path,
        statistics,
    )
    
if __name__ == "__main__":
    run_sims()
    
