from pathlib import Path

import numpy as np
from numba import njit, prange
from numba.typed import List

from ..core.config import FULL_GAME_CONFIG, PAY_LINES
from ..core.feature_kernels import (
    run_boost_kernel,
    run_collect_kernel,
    run_expansion_kernel,
    run_grow_kernel,
    run_mega_combo_kernel,
    run_multiplier_kernel,
    run_splitter_kernel,
)
from ..core.reels import make_reel_collection
from ..core.storage import CompactBaseStorage, FullGameStorage, Storage
from .base_game import _run_one_paid_spin_with_features
from .free_game import FEATURE_NAMES


@njit
def _seed_numba_random(seed):
    np.random.seed(seed)


def validate_full_game_config(config):
    """Validate the contracts shared across the composed game configs."""
    num_jackpots = len(config.jackpots.jackpot_types)
    if len(config.base_game.scatter_feature_symbols) != len(FEATURE_NAMES):
        raise ValueError("Base feature routing must define seven feature symbols")
    if len(config.jackpot_tokens.symbols) != num_jackpots:
        raise ValueError("Feature jackpot-token count must match jackpot count")
    return config


@njit
def run_one_full_round(
    initial_reels,
    config,
    jackpot_values,
    base_storage,
    full_game_storage,
):
    """Run one round and dispatch all configured branch feature mechanics."""
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
            payout_multiplier = config.feature_rtp.feature_payout_multiplier
            if combo_triggered:
                feature_result = run_mega_combo_kernel(
                    pay_window, config.features.mega_combo, payout_multiplier
                )
                feature_index = 6
                feature_trigger_counts[feature_index] += 1
                fixed_jackpot_component = (
                    feature_result[3] * payout_multiplier
                )
                feature_wins_by_type[feature_index] += max(
                    0.0,
                    feature_result[0] - fixed_jackpot_component,
                )
                feature_spins_by_type[feature_index] += feature_result[1]
                jackpot_awards += feature_result[2]
            else:
                for feature_index in range(6):
                    if not feature_flags[feature_index]:
                        continue
                    if feature_index == 0:
                        feature_result = run_splitter_kernel(
                            config.features.splitter, payout_multiplier
                        )
                    elif feature_index == 1:
                        feature_result = run_grow_kernel(
                            config.features.grow, payout_multiplier
                        )
                    elif feature_index == 2:
                        feature_result = run_boost_kernel(
                            config.features.boost, payout_multiplier
                        )
                    elif feature_index == 3:
                        feature_result = run_multiplier_kernel(
                            config.features.multiplier, payout_multiplier
                        )
                    elif feature_index == 4:
                        feature_result = run_collect_kernel(
                            config.features.collect, payout_multiplier
                        )
                    else:
                        feature_result = run_expansion_kernel(
                            config.features.expansion, payout_multiplier
                        )
                    feature_trigger_counts[feature_index] += 1
                    fixed_jackpot_component = (
                        feature_result[3] * payout_multiplier
                    )
                    feature_wins_by_type[feature_index] += max(
                        0.0, feature_result[0] - fixed_jackpot_component
                    )
                    feature_spins_by_type[feature_index] += feature_result[1]
                    jackpot_awards += feature_result[2]

        jackpot_win = 0.0
        for jackpot_type, award_count in enumerate(jackpot_awards):
            for _ in range(int(award_count)):
                award_amount = jackpot_values[jackpot_type]
                jackpot_award_amounts[jackpot_type] += award_amount
                jackpot_win += award_amount
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
    full_game_storage,
    seed=None,
):
    """Run full-game rounds through the seven configured feature routes."""
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
    """Allocate storage for the configured seven-feature full game."""
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
    full_game_storage = FullGameStorage(
        estimated_spins,
        max(1, num_rounds),
        num_jackpots,
        len(FEATURE_NAMES),
    )
    return base_storage, full_game_storage


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
    """Run, store and report combined base, routed-feature and jackpot play."""
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
        worker_configs = List()
        for worker_rounds in rounds_per_worker:
            base_shard, full_shard = _new_storages(
                int(worker_rounds), config, compact_base=compact_storage
            )
            base_storages.append(base_shard)
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
            full_storages,
            worker_seeds,
        )
        if compact_storage:
            base_storage = _merge_compact_base_storages(base_storages)
        else:
            base_storage = _merge_base_storages(base_storages, config)
        full_storage = _merge_full_game_storages(full_storages, config)
    else:
        base_storage, full_storage = _new_storages(
            num_rounds, config, compact_base=compact_storage
        )
        result = run_full_rounds(
            initial_reels,
            config,
            num_rounds,
            base_storage,
            full_storage,
            seed=seed,
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
        full_storage,
        output_path,
        statistics,
    )


if __name__ == "__main__":
    run_sims()
