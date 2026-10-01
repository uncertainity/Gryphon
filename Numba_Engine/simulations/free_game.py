from pathlib import Path

import numpy as np
from numba import njit, prange
from numba.typed import List

from ..core.config import HOLD_AND_SPIN_CONFIG
from ..core.hold_and_spin_kernels import active_coin_win, hold_and_spin
from ..core.kernels import probChoice
from ..core.storage import (
    HoldAndSpinStorage,
    merge_hold_and_spin_npz,
    write_hold_and_spin_npz,
)
from .feature_flows import (
    run_boost_feature,
    run_collect_feature,
    run_expansion_feature,
    run_grow_feature,
    run_mega_combo_feature,
    run_multiplier_feature,
    run_splitter_feature,
)


FEATURE_NAMES = (
    "Splitter",
    "Grow",
    "Boost",
    "Multiplier",
    "Collect",
    "Expansion",
    "Mega Combo",
    "Plain",
)


def validate_hold_and_spin_config(rules):
    """Validate the Numba-friendly Hold-and-Spin state-table contract."""
    max_positions = rules.num_rows * rules.num_reels
    if rules.num_rows < 1 or rules.num_reels < 1:
        raise ValueError("Hold-and-Spin board dimensions must be positive")
    if rules.starting_rows < 1 or rules.starting_rows > rules.num_rows:
        raise ValueError("starting_rows must fit inside the backing board")
    if not 0.0 <= rules.p_coin_locked <= 1.0:
        raise ValueError("p_coin_locked must be between zero and one")
    if not 0.0 <= rules.p_coin_unlocked <= 1.0:
        raise ValueError("p_coin_unlocked must be between zero and one")

    expected_state_shape = (max_positions + 1,)
    state_tables = (
        (
            "p_coin_locked_by_occupied_count",
            rules.p_coin_locked_by_occupied_count,
        ),
        (
            "p_coin_unlocked_by_occupied_count",
            rules.p_coin_unlocked_by_occupied_count,
        ),
    )
    for name, probabilities in state_tables:
        if probabilities.shape != expected_state_shape:
            raise ValueError(
                f"{name} must have one entry for every occupied-cell count"
            )
        if not np.all(np.isfinite(probabilities)):
            raise ValueError(f"{name} must contain only finite values")
        configured = probabilities[probabilities >= 0.0]
        if np.any(configured > 1.0):
            raise ValueError(f"Configured {name} values cannot exceed one")

    num_coin_types = len(rules.coin_types)
    expected_type_shape = (max_positions + 1, num_coin_types)
    state_type_probabilities = (
        rules.coin_type_probabilities_by_occupied_count
    )
    if state_type_probabilities.shape != expected_type_shape:
        raise ValueError(
            "coin_type_probabilities_by_occupied_count has an invalid shape"
        )
    if not np.all(np.isfinite(state_type_probabilities)):
        raise ValueError(
            "coin_type_probabilities_by_occupied_count must be finite"
        )
    for occupied_count in range(max_positions + 1):
        probabilities = state_type_probabilities[occupied_count]
        if np.all(probabilities < 0.0):
            continue
        if np.any(probabilities < 0.0) or not np.isclose(
            probabilities.sum(), 1.0
        ):
            raise ValueError(
                "Each configured occupied-count coin-type row must be a "
                "complete probability distribution"
            )

    if rules.coin_type_probabilities.shape != (num_coin_types,):
        raise ValueError("coin_type_probabilities must match coin_types")
    if not np.all(np.isfinite(rules.coin_type_probabilities)) or np.any(
        rules.coin_type_probabilities < 0.0
    ) or not np.isclose(
        rules.coin_type_probabilities.sum(), 1.0
    ):
        raise ValueError("coin_type_probabilities must sum to one")
    if rules.coin_type_respin_reset_flags.shape != (num_coin_types,):
        raise ValueError("coin_type_respin_reset_flags must match coin_types")
    if num_coin_types != len(rules.bag_symbols) + 1 or not np.array_equal(
        rules.coin_types[1:],
        rules.bag_symbols,
    ):
        raise ValueError(
            "coin_types must contain the natural Coin followed by SC1-SC6"
        )
    if not np.isfinite(rules.plain_route_weight) or (
        rules.plain_route_weight < 0.0
    ):
        raise ValueError("plain_route_weight must be finite and nonnegative")
    if rules.single_route_weights.shape != (len(rules.bag_symbols),):
        raise ValueError("single_route_weights must contain one weight per Bag")
    if not np.all(np.isfinite(rules.single_route_weights)) or np.any(
        rules.single_route_weights < 0.0
    ):
        raise ValueError("single_route_weights must be finite and nonnegative")
    if rules.plain_route_weight + rules.single_route_weights.sum() <= 0.0:
        raise ValueError("At least one Hold-and-Spin route weight is required")
    if len(rules.bag_symbols) != len(rules.bag_resolution_order):
        raise ValueError("bag_resolution_order must cover every Bag")
    if not np.array_equal(
        np.sort(rules.bag_resolution_order),
        np.arange(len(rules.bag_symbols)),
    ):
        raise ValueError("bag_resolution_order must be a Bag-index permutation")
    if len(rules.bag_symbols) != len(rules.bag_symbol_actions):
        raise ValueError("bag_symbol_actions must cover every Bag")
    if np.any(rules.bag_symbol_actions < 0) or np.any(
        rules.bag_symbol_actions > 2
    ):
        raise ValueError("bag_symbol_actions contains an unknown action")
    if not 0.0 <= rules.jackpot_token_probability <= 1.0:
        raise ValueError("jackpot_token_probability must be between zero and one")
    if np.any(rules.jackpot_type_probabilities < 0.0) or not np.isclose(
        rules.jackpot_type_probabilities.sum(), 1.0
    ):
        raise ValueError("jackpot_type_probabilities must sum to one")
    if len(rules.jackpot_type_probabilities) != len(
        rules.jackpot_collection_targets
    ):
        raise ValueError("Jackpot type weights and meter targets must align")
    return rules


def run_configured_features(
    pay_window,
    feature_flags,
    combo_triggered,
    config,
    random_generator,
):
    """Route a converted base window through the configured feature engine."""
    flags = np.asarray(feature_flags, dtype=np.bool_).copy()
    for index, symbol in enumerate(config.base_game.scatter_feature_symbols):
        if np.any(pay_window == symbol):
            flags[index] = True

    def next_seed():
        return int(random_generator.integers(0, np.iinfo(np.int32).max))

    if combo_triggered:
        result = run_mega_combo_feature(
            base_window=pay_window,
            rules=config.features.mega_combo,
            seed=next_seed(),
            payout_multiplier=config.feature_rtp.feature_payout_multiplier,
        )
        return [(6, result)]

    routed = []
    runners = (
        (run_splitter_feature, config.features.splitter),
        (run_grow_feature, config.features.grow),
        (run_boost_feature, config.features.boost),
        (run_multiplier_feature, config.features.multiplier),
        (run_collect_feature, config.features.collect),
        (run_expansion_feature, config.features.expansion),
    )
    for feature_index, (runner, rules) in enumerate(runners):
        if flags[feature_index]:
            routed.append(
                (
                    feature_index,
                    runner(
                        rules=rules,
                        seed=next_seed(),
                        payout_multiplier=(
                            config.feature_rtp.feature_payout_multiplier
                        ),
                    ),
                )
            )
    return routed


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

    coin_win = active_coin_win(
        result.pay_window,
        result.coin_mask,
        result.locked_row_idx,
    )

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
    validate_hold_and_spin_config(rules)

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
