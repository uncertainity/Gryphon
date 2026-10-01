import argparse
from pathlib import Path

import numpy as np
from numba import njit, prange
from numba.typed import List

from ..core.config import FULL_GAME_CONFIG, PAY_LINES
from ..core.reels import make_reel_collection
from ..core.storage import (
    CompactBaseStorage,
    FullGameStorage,
    HoldAndSpinStorage,
    Storage,
)
from .base_game import _run_one_paid_spin_with_features
from .free_game import (
    FEATURE_NAMES,
    hold_and_free_spin,
    validate_hold_and_spin_config,
)


@njit
def _seed_numba_random(seed):
    np.random.seed(seed)


@njit
def extract_starting_bag_symbols(pay_window, bag_symbols):
    """Return every visible SC1-SC6 occurrence in board order."""
    starting_bags = np.empty(pay_window.size, dtype=np.int16)
    count = 0
    for symbol in pay_window.ravel():
        for bag_symbol in bag_symbols:
            if symbol == bag_symbol:
                starting_bags[count] = symbol
                count += 1
                break
    return starting_bags[:count]


@njit
def select_feature_route(feature_flags, combo_triggered, rules):
    """Select plain, one visible Bag, or deterministic all-six Mega."""
    num_bags = len(rules.bag_symbols)
    if combo_triggered:
        return num_bags

    total_weight = rules.plain_route_weight
    for bag_index in range(num_bags):
        if feature_flags[bag_index]:
            total_weight += rules.single_route_weights[bag_index]
    if total_weight <= 0.0:
        raise ValueError("No eligible Hold-and-Spin route has positive weight")

    choice = np.random.uniform(0.0, total_weight)
    cumulative_weight = rules.plain_route_weight
    if choice < cumulative_weight:
        return num_bags + 1
    for bag_index in range(num_bags):
        if not feature_flags[bag_index]:
            continue
        cumulative_weight += rules.single_route_weights[bag_index]
        if choice < cumulative_weight:
            return bag_index
    return num_bags + 1


@njit
def starting_bags_for_route(feature_index, bag_symbols):
    """Build the only valid kernel start: none, one Bag, or all six."""
    if feature_index == len(bag_symbols):
        return bag_symbols.copy()
    if feature_index >= 0 and feature_index < len(bag_symbols):
        return np.array([bag_symbols[feature_index]], dtype=np.int16)
    return np.empty(0, dtype=np.int16)


def validate_full_game_config(config):
    """Validate the contracts used by the integrated shared feature."""
    validate_hold_and_spin_config(config.hold_and_spin)
    num_jackpots = len(config.jackpots.jackpot_types)
    if not np.array_equal(
        config.base_game.scatter_feature_symbols,
        config.hold_and_spin.bag_symbols,
    ):
        raise ValueError("Base conversion must define exactly SC1-SC6")
    conversion_probabilities = (
        config.base_game.scatter_feature_symbol_probabilities
    )
    if conversion_probabilities.shape != (6,):
        raise ValueError("Base conversion must have one weight per Bag")
    if np.any(conversion_probabilities < 0.0) or not np.isclose(
        conversion_probabilities.sum(), 1.0
    ):
        raise ValueError("Base conversion weights must sum to one")
    if len(config.jackpot_tokens.symbols) != num_jackpots:
        raise ValueError("Feature jackpot-token count must match jackpot count")
    if config.hold_and_spin.num_rows != 6:
        raise ValueError("Gryphon Hold-and-Spin must use six backing rows")
    if config.hold_and_spin.num_reels != 5:
        raise ValueError("Gryphon Hold-and-Spin must use five reels")
    if config.hold_and_spin.starting_rows != 3:
        raise ValueError("Gryphon Hold-and-Spin must start with three active rows")
    if len(config.hold_and_spin.bag_symbols) != 6:
        raise ValueError("Gryphon Hold-and-Spin must configure six Bags")
    if not np.array_equal(
        config.base_game.combo_feature_symbols,
        config.hold_and_spin.bag_symbols,
    ):
        raise ValueError("Mega Combo must require exactly SC1-SC6")
    payout_multipliers = (
        config.feature_rtp.hold_and_spin_payout_multipliers
    )
    if payout_multipliers.shape != (len(FEATURE_NAMES),):
        raise ValueError("Shared H&S needs one payout multiplier per category")
    if np.any(payout_multipliers < 0.0) or not np.all(
        np.isfinite(payout_multipliers)
    ):
        raise ValueError("Shared H&S payout multipliers must be finite and nonnegative")
    if not np.isfinite(config.max_win) or config.max_win <= 0.0:
        raise ValueError("Full-game max_win must be finite and positive")
    route_probability_total = (
        config.feature_rtp.plain_feature_probability
        + len(config.hold_and_spin.bag_symbols)
        * config.feature_rtp.single_feature_probability
        + config.feature_rtp.mega_feature_probability
    )
    if not np.isclose(
        route_probability_total,
        config.feature_rtp.overall_feature_probability,
    ):
        raise ValueError("Hold-and-Spin route probabilities must sum to the overall rate")
    return config


@njit
def run_one_full_round(
    initial_reels,
    config,
    jackpot_values,
    base_storage,
    hold_and_spin_storage,
    full_game_storage,
):
    """Run one round and launch at most one routed H&S per paid spin."""
    base_rules = config.base_game
    jackpot_rules = config.jackpots
    collector_positions = np.empty(
        base_rules.max_active_collectors, dtype=np.int32
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
            feature_flags,
            combo_triggered,
            line_win,
            collect_win,
        ) = _run_one_paid_spin_with_features(
            initial_reels,
            base_rules,
            config.base_jackpot_overlay,
            jackpot_rules,
            base_storage,
            jackpot_values,
            collector_positions,
            collector_count,
        )

        # The hard cap applies to the complete wager round, including any
        # collector-driven continuation spins and all launched features.
        remaining_cap = max(
            0.0,
            config.max_win
            - round_base_win
            - round_feature_win
            - round_jackpot_win,
        )
        line_win = min(line_win, remaining_cap)
        remaining_cap -= line_win
        collect_win = min(collect_win, remaining_cap)
        remaining_cap -= collect_win
        base_win = line_win + collect_win

        values_before_feature = jackpot_values.copy()
        feature_trigger_counts = np.zeros(len(FEATURE_NAMES), dtype=np.int64)
        feature_wins_by_type = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        feature_spins_by_type = np.zeros(len(FEATURE_NAMES), dtype=np.int64)
        jackpot_awards = np.zeros(
            len(jackpot_rules.jackpot_types), dtype=np.int16
        )
        jackpot_award_amounts = np.zeros(
            len(jackpot_rules.jackpot_types), dtype=np.float64
        )

        if feature_triggered:
            feature_index = select_feature_route(
                feature_flags,
                combo_triggered,
                config.hold_and_spin,
            )
            starting_bags = starting_bags_for_route(
                feature_index,
                config.hold_and_spin.bag_symbols,
            )
            raw_feature_win, feature_result = hold_and_free_spin(
                starting_bags,
                config.hold_and_spin,
                hold_and_spin_storage,
            )
            feature_win = (
                raw_feature_win
                * config.feature_rtp.hold_and_spin_payout_multipliers[
                    feature_index
                ]
            )
            feature_win = min(feature_win, remaining_cap)
            remaining_cap -= feature_win
            feature_trigger_counts[feature_index] += 1
            feature_wins_by_type[feature_index] += feature_win
            feature_spins_by_type[feature_index] += (
                feature_result.total_spins
            )
            for jackpot_type in range(len(jackpot_awards)):
                if feature_result.awarded_jackpots[jackpot_type]:
                    jackpot_awards[jackpot_type] = 1

        jackpot_win = 0.0
        for jackpot_type, award_count in enumerate(jackpot_awards):
            for _ in range(int(award_count)):
                award_amount = min(
                    jackpot_values[jackpot_type],
                    remaining_cap,
                )
                jackpot_award_amounts[jackpot_type] += award_amount
                jackpot_win += award_amount
                remaining_cap -= award_amount
                jackpot_values[jackpot_type] = (
                    jackpot_rules.seed_values[jackpot_type]
                )

        feature_win = float(feature_wins_by_type.sum())
        trigger_count = int(feature_trigger_counts.sum())
        full_game_storage.save_spin(
            feature_trigger_counts,
            feature_wins_by_type,
            feature_spins_by_type,
            base_win,
            line_win,
            collect_win,
            jackpot_win,
            values_before_feature,
            jackpot_awards,
            jackpot_award_amounts,
            jackpot_values,
        )
        round_base_win += base_win
        round_feature_win += feature_win
        round_jackpot_win += jackpot_win
        round_triggers += trigger_count
        paid_spins += 1
        if collector_count == 0:
            break

    base_storage.finish_round(round_base_win, round_triggers)
    full_game_storage.finish_round(
        round_base_win, round_feature_win, round_jackpot_win
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
    hold_and_spin_storage,
    full_game_storage,
    seed=None,
):
    """Run rounds through the routed Hold-and-Spin feature engine."""
    if seed is not None:
        _seed_numba_random(int(seed))
    jackpot_values = config.jackpots.seed_values.copy()
    totals = np.zeros(6, dtype=np.float64)
    for _ in range(num_rounds):
        result = run_one_full_round(
            initial_reels,
            config,
            jackpot_values,
            base_storage,
            hold_and_spin_storage,
            full_game_storage,
        )
        totals[0] += result[0]
        totals[1] += result[1]
        totals[2] += result[2]
        totals[3] += result[3]
        totals[4] += result[4]
        totals[5] += result[5]
    return (
        totals[0],
        totals[1],
        totals[2],
        totals[3],
        int(totals[4]),
        int(totals[5]),
    )


@njit(parallel=True)
def run_full_rounds_parallel(
    initial_reels,
    worker_configs,
    rounds_per_worker,
    base_storages,
    hold_and_spin_storages,
    full_game_storages,
    worker_seeds,
):
    """Run independent full-game shards with race-free worker state."""
    worker_totals = np.zeros((len(rounds_per_worker), 6), np.float64)
    for worker_index in prange(len(rounds_per_worker)):
        storage_index = np.int64(worker_index)
        result = run_full_rounds(
            initial_reels,
            worker_configs[storage_index],
            int(rounds_per_worker[storage_index]),
            base_storages[storage_index],
            hold_and_spin_storages[storage_index],
            full_game_storages[storage_index],
            seed=int(worker_seeds[storage_index]),
        )
        worker_totals[worker_index, 0] = result[0]
        worker_totals[worker_index, 1] = result[1]
        worker_totals[worker_index, 2] = result[2]
        worker_totals[worker_index, 3] = result[3]
        worker_totals[worker_index, 4] = result[4]
        worker_totals[worker_index, 5] = result[5]
    totals = np.zeros(6, np.float64)
    for worker_index in range(len(rounds_per_worker)):
        for result_index in range(6):
            totals[result_index] += worker_totals[worker_index, result_index]
    return (
        totals[0], totals[1], totals[2], totals[3],
        int(totals[4]), int(totals[5]),
    )


def _new_storages(num_rounds, config, compact_base=False):
    """Allocate base, routed Hold-and-Spin, and integration storage."""
    estimated_spins = max(
        1,
        num_rounds + num_rounds // 2 if compact_base else num_rounds * 2,
    )
    num_jackpots = len(config.jackpots.jackpot_types)
    if compact_base:
        base_storage = CompactBaseStorage()
    else:
        base_storage = Storage(
            estimated_spins,
            config.base_game.num_rows,
            config.base_game.num_reels,
            config.base_game.num_paying_symbols,
            len(PAY_LINES),
            num_jackpots,
        )
    # Size detailed feature history from the configured paid-spin launch
    # rate, not as though every paid spin starts Hold-and-Spin.  Every storage
    # level grows dynamically, so the headroom affects allocation only and
    # cannot truncate a high-trigger shard or alter game outcomes.
    expected_sessions = (
        estimated_spins
        * config.feature_rtp.overall_feature_probability
    )
    estimated_sessions = max(
        1,
        int(np.ceil(expected_sessions * 1.5)) + 32,
    )
    estimated_respins = max(
        1,
        estimated_sessions
        * config.hold_and_spin.respin_reset_count
        * 2,
    )
    estimated_steps = max(1, estimated_respins * 3)
    hold_and_spin_storage = HoldAndSpinStorage(
        estimated_steps,
        estimated_respins,
        estimated_sessions,
        config.hold_and_spin.num_rows,
        config.hold_and_spin.num_reels,
        num_jackpots,
    )
    full_game_storage = FullGameStorage(
        estimated_spins,
        max(1, num_rounds),
        num_jackpots,
        len(FEATURE_NAMES),
    )
    return base_storage, hold_and_spin_storage, full_game_storage


def _merge_base_storages(storages, config):
    total_spins = sum(int(storage.spin_count) for storage in storages)
    total_rounds = sum(int(storage.round_count) for storage in storages)
    merged = Storage(
        max(1, total_spins),
        config.base_game.num_rows,
        config.base_game.num_reels,
        config.base_game.num_paying_symbols,
        len(PAY_LINES),
        len(config.jackpots.jackpot_types),
    )
    spin_arrays = (
        "boards", "coin_value_boards", "jackpot_overlay_boards",
        "jackpot_values_before", "jackpot_values_after",
        "jackpot_increment_counts", "wins", "line_wins", "collect_wins",
        "spin_triggers", "collector_counts", "symbol_wins",
        "symbol_hit_counts", "winning_line_numbers",
        "line_winning_symbols", "line_symbol_counts", "line_win_amounts",
    )
    spin_offset = 0
    round_offset = 0
    for storage in storages:
        spin_count = int(storage.spin_count)
        round_count = int(storage.round_count)
        for name in spin_arrays:
            getattr(merged, name)[spin_offset : spin_offset + spin_count] = (
                getattr(storage, name)[:spin_count]
            )
        merged.round_wins[round_offset : round_offset + round_count] = (
            storage.round_wins[:round_count]
        )
        merged.round_triggers[round_offset : round_offset + round_count] = (
            storage.round_triggers[:round_count]
        )
        merged.round_spin_offsets[
            round_offset : round_offset + round_count + 1
        ] = storage.round_spin_offsets[: round_count + 1] + spin_offset
        spin_offset += spin_count
        round_offset += round_count
    merged.spin_count = total_spins
    merged.round_count = total_rounds
    return merged


def _merge_compact_base_storages(storages):
    merged = CompactBaseStorage()
    merged.spin_count = sum(int(storage.spin_count) for storage in storages)
    merged.round_count = sum(int(storage.round_count) for storage in storages)
    return merged


def _merge_hold_and_spin_storages(storages, config):
    """Merge detailed worker H&S histories and rebase both offset levels."""
    total_steps = sum(int(storage.step_count) for storage in storages)
    total_respins = sum(int(storage.respin_count) for storage in storages)
    total_sessions = sum(int(storage.session_count) for storage in storages)
    merged = HoldAndSpinStorage(
        max(1, total_steps),
        max(1, total_respins),
        max(1, total_sessions),
        config.hold_and_spin.num_rows,
        config.hold_and_spin.num_reels,
        len(config.jackpots.jackpot_types),
    )
    step_arrays = (
        "boards",
        "coin_masks",
        "feature_types",
        "feature_positions",
        "step_remaining_spins",
        "step_locked_row_indices",
        "step_collector_meters",
    )
    respin_arrays = (
        "respin_remaining_before",
        "respin_remaining_after",
        "respin_reset_flags",
        "respin_jackpot_overlay_boards",
        "respin_jackpot_meters_before",
        "respin_jackpot_meters_after",
        "respin_jackpot_awards",
    )
    session_arrays = (
        "session_wins",
        "session_coin_wins",
        "session_collector_wins",
        "session_total_respins",
        "session_starting_symbols",
        "session_starting_symbol_counts",
        "session_jackpot_meters",
        "session_jackpot_awards",
    )
    step_offset = 0
    respin_offset = 0
    session_offset = 0
    for storage in storages:
        step_count = int(storage.step_count)
        respin_count = int(storage.respin_count)
        session_count = int(storage.session_count)
        for name in step_arrays:
            getattr(merged, name)[step_offset : step_offset + step_count] = (
                getattr(storage, name)[:step_count]
            )
        for name in respin_arrays:
            getattr(merged, name)[
                respin_offset : respin_offset + respin_count
            ] = getattr(storage, name)[:respin_count]
        for name in session_arrays:
            getattr(merged, name)[
                session_offset : session_offset + session_count
            ] = getattr(storage, name)[:session_count]
        merged.respin_step_offsets[
            respin_offset : respin_offset + respin_count + 1
        ] = storage.respin_step_offsets[: respin_count + 1] + step_offset
        merged.session_respin_offsets[
            session_offset : session_offset + session_count + 1
        ] = storage.session_respin_offsets[: session_count + 1] + respin_offset
        step_offset += step_count
        respin_offset += respin_count
        session_offset += session_count
    merged.step_count = total_steps
    merged.respin_count = total_respins
    merged.session_count = total_sessions
    return merged


def _merge_full_game_storages(storages, config):
    total_spins = sum(int(storage.spin_count) for storage in storages)
    total_rounds = sum(int(storage.round_count) for storage in storages)
    merged = FullGameStorage(
        max(1, total_spins),
        max(1, total_rounds),
        len(config.jackpots.jackpot_types),
        len(FEATURE_NAMES),
    )
    spin_arrays = (
        "spin_feature_masks", "spin_base_wins", "spin_feature_wins",
        "spin_jackpot_wins", "spin_total_wins",
        "spin_jackpot_values_before_feature", "spin_jackpot_awards",
        "spin_jackpot_award_amounts", "spin_jackpot_values_after_feature",
    )
    round_arrays = (
        "round_base_wins", "round_feature_wins", "round_jackpot_wins",
        "round_total_wins",
    )
    spin_offset = 0
    round_offset = 0
    session_offset = 0
    for storage in storages:
        spin_count = int(storage.spin_count)
        round_count = int(storage.round_count)
        for name in spin_arrays:
            getattr(merged, name)[spin_offset : spin_offset + spin_count] = (
                getattr(storage, name)[:spin_count]
            )
        indices = storage.spin_feature_session_indices[:spin_count].copy()
        for index in range(spin_count):
            if indices[index] >= 0:
                indices[index] += session_offset
        merged.spin_feature_session_indices[
            spin_offset : spin_offset + spin_count
        ] = indices
        for name in round_arrays:
            getattr(merged, name)[round_offset : round_offset + round_count] = (
                getattr(storage, name)[:round_count]
            )
        merged.round_spin_offsets[
            round_offset : round_offset + round_count + 1
        ] = storage.round_spin_offsets[: round_count + 1] + spin_offset
        merged.feature_trigger_counts += storage.feature_trigger_counts
        merged.feature_win_amounts += storage.feature_win_amounts
        merged.feature_spin_counts += storage.feature_spin_counts
        merged.base_line_win_amount += storage.base_line_win_amount
        merged.base_collect_win_amount += storage.base_collect_win_amount
        spin_offset += spin_count
        round_offset += round_count
        session_offset += int(storage.feature_session_count)
    merged.spin_count = total_spins
    merged.round_count = total_rounds
    merged.feature_session_count = session_offset
    return merged


def run_sims(
    num_rounds=10_000,
    num_workers=3,
    use_parallel=False,
    compact_storage=False,
    config=FULL_GAME_CONFIG,
    output_filename="full_game_results.npz",
    overwrite=False,
    bet_per_spin=1.0,
    print_statistics=True,
    seed=None,
):
    """Run, store and report base, routed H&S, and jackpot play."""
    if num_rounds < 1:
        raise ValueError("num_rounds must be positive")
    if num_workers < 1:
        raise ValueError("num_workers must be positive")
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
    if use_parallel:
        active_workers = min(num_workers, num_rounds)
        rounds_per_worker = np.full(
            active_workers, num_rounds // active_workers, dtype=np.int64
        )
        rounds_per_worker[: num_rounds % active_workers] += 1
        if compact_storage:
            base_storages = List.empty_list(
                CompactBaseStorage.class_type.instance_type
            )
        else:
            base_storages = List.empty_list(Storage.class_type.instance_type)
        full_storages = List.empty_list(
            FullGameStorage.class_type.instance_type
        )
        hold_and_spin_storages = List.empty_list(
            HoldAndSpinStorage.class_type.instance_type
        )
        worker_configs = List()
        for worker_rounds in rounds_per_worker:
            base_shard, hold_and_spin_shard, full_shard = _new_storages(
                int(worker_rounds), config, compact_base=compact_storage
            )
            base_storages.append(base_shard)
            hold_and_spin_storages.append(hold_and_spin_shard)
            full_storages.append(full_shard)
            worker_configs.append(config)
        if seed is None:
            worker_seeds = np.random.SeedSequence().generate_state(
                active_workers, dtype=np.uint32
            ).astype(np.int64)
        else:
            worker_seeds = (
                int(seed) + np.arange(active_workers, dtype=np.int64) * 104729
            ) % np.iinfo(np.int32).max
        result = run_full_rounds_parallel(
            initial_reels,
            worker_configs,
            rounds_per_worker,
            base_storages,
            hold_and_spin_storages,
            full_storages,
            worker_seeds,
        )
        if compact_storage:
            base_storage = _merge_compact_base_storages(base_storages)
        else:
            base_storage = _merge_base_storages(base_storages, config)
        hold_and_spin_storage = _merge_hold_and_spin_storages(
            hold_and_spin_storages,
            config,
        )
        full_storage = _merge_full_game_storages(full_storages, config)
    else:
        base_storage, hold_and_spin_storage, full_storage = _new_storages(
            num_rounds, config, compact_base=compact_storage
        )
        result = run_full_rounds(
            initial_reels,
            config,
            num_rounds,
            base_storage,
            hold_and_spin_storage,
            full_storage,
            seed=seed,
        )

    if int(hold_and_spin_storage.session_count) != int(
        full_storage.feature_session_count
    ):
        raise RuntimeError(
            "Hold-and-Spin and full-game feature session counts diverged"
        )
    if int(result[4]) != int(full_storage.spin_count):
        raise RuntimeError("Full-game paid-spin count diverged from storage")
    if int(result[5]) != int(full_storage.feature_session_count):
        raise RuntimeError("Full-game trigger count diverged from storage")
    if not np.isclose(result[0], full_storage.spin_total_wins[: full_storage.spin_count].sum()):
        raise RuntimeError("Full-game total win diverged from storage")

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
        hold_and_spin_storage,
        full_storage,
        output_path,
        statistics,
    )


def _positive_count(value):
    """Parse positive CLI counts, including values such as 10m."""
    text = value.strip().lower().replace("_", "").replace(",", "")
    multiplier = 1
    if text.endswith("k"):
        text = text[:-1]
        multiplier = 1_000
    elif text.endswith("m"):
        text = text[:-1]
        multiplier = 1_000_000
    try:
        number = int(text) * multiplier
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer count, got {value!r}"
        ) from error
    if number < 1:
        raise argparse.ArgumentTypeError("count must be positive")
    return number


def _positive_float(value):
    """Parse a positive floating-point command-line value."""
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected a positive number, got {value!r}"
        ) from error
    if not np.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return number


def build_argument_parser():
    """Build the command-line interface for integrated simulations."""
    parser = argparse.ArgumentParser(
        description=(
            "Run integrated Gryphon base-game, Hold-and-Spin, and jackpot "
            "simulations."
        )
    )
    parser.add_argument(
        "-n",
        "--num-rounds",
        "--num-spins",
        dest="num_rounds",
        type=_positive_count,
        default=10_000,
        help=(
            "number of full-game rounds (accepts k/m suffixes; one round "
            "can contain multiple paid spins), default: 10k"
        ),
    )
    parser.add_argument(
        "-w",
        "--num-workers",
        type=_positive_count,
        default=3,
        help="number of parallel worker shards, default: 3",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        "--use-parallel",
        dest="use_parallel",
        action="store_true",
        help="use the parallel full-game execution path",
    )
    parser.add_argument(
        "-c",
        "--compact-storage",
        action="store_true",
        help="use compact base-game storage to reduce memory consumption",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        "--output-filename",
        dest="output_filename",
        default="full_game_results.npz",
        help=(
            "NPZ filename inside Numba_Engine/output/npz_library, "
            "default: full_game_results.npz"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing NPZ output file",
    )
    parser.add_argument(
        "--bet-per-spin",
        type=_positive_float,
        default=1.0,
        help="bet used to calculate reported RTP, default: 1.0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="optional random seed",
    )
    parser.add_argument(
        "--no-statistics",
        dest="print_statistics",
        action="store_false",
        help="do not print the final statistics report",
    )
    return parser


def main(argv=None):
    """Parse command-line arguments and run the requested simulation."""
    args = build_argument_parser().parse_args(argv)
    return run_sims(
        num_rounds=args.num_rounds,
        num_workers=args.num_workers,
        use_parallel=args.use_parallel,
        compact_storage=args.compact_storage,
        output_filename=args.output_filename,
        overwrite=args.overwrite,
        bet_per_spin=args.bet_per_spin,
        print_statistics=args.print_statistics,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
