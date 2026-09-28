from pathlib import Path

import numpy as np
from numba import njit, prange
from numba.typed import List

from ..core.config import HOLD_AND_SPIN_CONFIG
from ..core.hold_and_spin_kernels import hold_and_spin
from ..core.kernels import probChoice
from ..core.storage import (
    HoldAndSpinStorage,
    merge_hold_and_spin_npz,
    write_hold_and_spin_npz,
)


@njit
def select_starting_bag_symbols(rules):
    """Choose a weighted count and sample distinct starting Bag symbols."""
    symbol_count = int(
        probChoice(
            rules.starting_scatter_count_probabilities,
            rules.starting_scatter_counts,
        )
    )
    symbol_count = min(symbol_count, len(rules.bag_symbols))
    starting_bag_symbols = np.empty(symbol_count, dtype=np.int16)
    candidate_indices = np.arange(len(rules.bag_symbols), dtype=np.int32)
    for symbol_index in range(symbol_count):
        swap_index = np.random.randint(symbol_index, len(rules.bag_symbols))
        candidate_index = candidate_indices[swap_index]
        candidate_indices[swap_index] = candidate_indices[symbol_index]
        candidate_indices[symbol_index] = candidate_index
        starting_bag_symbols[symbol_index] = rules.bag_symbols[candidate_index]
    return starting_bag_symbols


@njit
def hold_and_free_spin(starting_bag_symbols, rules, storage):
    """Run and store one complete Hold-and-Spin feature session."""
    result = hold_and_spin(starting_bag_symbols, rules, storage)

    coin_win = 0
    flattened_window = result.pay_window.ravel()
    for position in range(len(result.coin_mask)):
        if result.coin_mask[position]:
            coin_win += flattened_window[position]

    feature_win = float(coin_win + result.collector_meter)
    return feature_win, result


@njit
def run_sessions(
    starting_bag_symbols,
    rules,
    num_sessions,
    storage,
    randomize_starting_bag_symbols=False,
):
    """Run independent Hold-and-Spin sessions into one Storage shard."""
    total_win = 0.0
    total_respins = 0

    for _ in range(num_sessions):
        session_starting_bag_symbols = starting_bag_symbols
        if randomize_starting_bag_symbols:
            session_starting_bag_symbols = select_starting_bag_symbols(rules)

        feature_win, result = hold_and_free_spin(
            session_starting_bag_symbols,
            rules,
            storage,
        )
        total_win += feature_win
        total_respins += result.total_spins

    return total_win, total_respins


@njit(parallel=True)
def run_parallel_num_sessions(
    starting_bag_symbols,
    rules,
    sessions_per_worker,
    worker_storages,
    randomize_starting_bag_symbols=False,
):
    """Run independent Hold-and-Spin session shards in parallel."""
    total_win = 0.0
    total_respins = 0

    for worker_index in prange(len(worker_storages)):
        storage_index = np.int64(worker_index)
        worker_win, worker_respins = run_sessions(
            starting_bag_symbols,
            rules,
            sessions_per_worker[storage_index],
            worker_storages[storage_index],
            randomize_starting_bag_symbols,
        )
        total_win += worker_win
        total_respins += worker_respins

    return total_win, total_respins


def _new_storage(num_sessions, rules):
    estimated_respins = max(1, num_sessions * rules.respin_reset_count * 2)
    estimated_steps = max(1, estimated_respins * 3)
    return HoldAndSpinStorage(
        estimated_steps,
        estimated_respins,
        max(1, num_sessions),
        rules.num_rows,
        rules.num_reels,
        len(rules.jackpot_collection_targets),
    )


def run_sims(
    num_sessions=1_000,
    starting_bag_symbols=None,
    rules=HOLD_AND_SPIN_CONFIG,
    num_workers=3,
    use_parallel=False,
    output_filename="hold_and_spin_results.npz",
    overwrite=False,
    bet_per_session=1.0,
    print_statistics=True,
):
    """Run, serialize, aggregate, and report Hold-and-Spin sessions."""
    if num_sessions < 1:
        raise ValueError("num_sessions must be positive")
    if num_workers < 1:
        raise ValueError("num_workers must be positive")
    if bet_per_session <= 0:
        raise ValueError("bet_per_session must be positive")

    output_filename = Path(output_filename)
    if output_filename.is_absolute() or len(output_filename.parts) != 1:
        raise ValueError("output_filename must not contain a directory path")
    if output_filename.suffix == "":
        output_filename = output_filename.with_suffix(".npz")
    elif output_filename.suffix.lower() != ".npz":
        raise ValueError("output_filename must use the .npz extension")

    randomize_starting_bag_symbols = starting_bag_symbols is None
    if randomize_starting_bag_symbols:
        starting_bag_symbols = np.empty(0, dtype=np.int16)
    else:
        starting_bag_symbols = np.asarray(
            starting_bag_symbols,
            dtype=np.int16,
        )
    if starting_bag_symbols.ndim != 1:
        raise ValueError("starting_bag_symbols must be one-dimensional")

    if use_parallel:
        active_workers = min(num_workers, num_sessions)
        sessions_per_worker = np.full(
            active_workers,
            num_sessions // active_workers,
            dtype=np.int64,
        )
        sessions_per_worker[:num_sessions % active_workers] += 1

        worker_storages = List.empty_list(
            HoldAndSpinStorage.class_type.instance_type
        )
        for worker_index in range(active_workers):
            worker_storages.append(
                _new_storage(
                    int(sessions_per_worker[worker_index]),
                    rules,
                )
            )

        total_win, total_respins = run_parallel_num_sessions(
            starting_bag_symbols,
            rules,
            sessions_per_worker,
            worker_storages,
            randomize_starting_bag_symbols,
        )

        shard_filenames = []
        for worker_index, worker_storage in enumerate(worker_storages):
            shard_filename = (
                f"{output_filename.stem}_worker_{worker_index}.npz"
            )
            write_hold_and_spin_npz(
                worker_storage,
                shard_filename,
                overwrite=overwrite,
            )
            shard_filenames.append(shard_filename)

        output_path = merge_hold_and_spin_npz(
            shard_filenames,
            output_filename.name,
            overwrite=overwrite,
        )
    else:
        storage = _new_storage(num_sessions, rules)
        total_win, total_respins = run_sessions(
            starting_bag_symbols,
            rules,
            num_sessions,
            storage,
            randomize_starting_bag_symbols,
        )
        output_path = write_hold_and_spin_npz(
            storage,
            output_filename.name,
            overwrite=overwrite,
        )

    from ..output.statistics import store_hold_and_spin

    statistics = store_hold_and_spin(
        output_path,
        bet_per_session=bet_per_session,
        print_result=print_statistics,
    )
    return total_win, total_respins, output_path, statistics


if __name__ == "__main__":
    run_sims()
