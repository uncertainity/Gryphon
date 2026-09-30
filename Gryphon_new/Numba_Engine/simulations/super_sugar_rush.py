import numpy as np
from pathlib import Path
from numba import njit
from numba.typed import List

from .free_game import (
    run_one_session as _run_one_session,
    run_one_spin as _run_one_spin,
    run_sessions as _run_sessions,
    run_sessions_parallel as _run_sessions_parallel,
)
from ..core.storage import Storage, merge_npz, write_npz


@njit
def run_one_spin(
    initial_reels,
    cascade_reels,
    rules,
    global_multiplier,
    storage,
):
    """Run one Super Sugar Rush spin using its supplied rules."""
    return _run_one_spin(
        initial_reels,
        cascade_reels,
        rules,
        global_multiplier,
        storage,
    )


@njit
def run_one_session(
    initial_reels,
    cascade_reels,
    rules,
    session_rules,
    num_spins,
    storage,
):
    """Run one Super Sugar Rush session, including retriggers."""
    return _run_one_session(
        initial_reels,
        cascade_reels,
        rules,
        session_rules,
        num_spins,
        storage,
    )


@njit
def run_sessions(
    initial_reels,
    cascade_reels,
    rules,
    session_rules,
    num_spins,
    num_sessions,
    storage,
):
    """Run Super Sugar Rush sessions into shared hierarchical storage."""
    return _run_sessions(
        initial_reels,
        cascade_reels,
        rules,
        session_rules,
        num_spins,
        num_sessions,
        storage,
    )


def run_sessions_parallel(
    initial_reels,
    cascade_reels,
    rules,
    session_rules,
    num_spins,
    sessions_per_worker,
    worker_storages,
):
    """Run independent Super Sugar Rush session shards in parallel."""
    return _run_sessions_parallel(
        initial_reels,
        cascade_reels,
        rules,
        session_rules,
        num_spins,
        sessions_per_worker,
        worker_storages,
    )


def run_sims(
    num_sessions=1_000,
    num_spins=8,
    num_workers=3,
    use_parallel=False,
    output_filename="super_sugar_rush_results.npz",
    overwrite=False,
):
    from ..core.config import (
        SUPER_SUGAR_RUSH_CASCADE_REELS,
        SUPER_SUGAR_RUSH_INITIAL_REELS,
        SUPER_SUGAR_RUSH_RULES,
        SUPER_SUGAR_RUSH_SESSION_RULES,
    )

    if num_sessions < 1:
        raise ValueError("num_sessions must be positive")
    if num_spins < 1:
        raise ValueError("num_spins must be positive")
    if num_workers < 1:
        raise ValueError("num_workers must be positive")

    output_filename = Path(output_filename)
    if output_filename.is_absolute() or len(output_filename.parts) != 1:
        raise ValueError("output_filename must not contain a directory path")
    if output_filename.suffix == "":
        output_filename = output_filename.with_suffix(".npz")

    if use_parallel:
        active_workers = min(num_workers, num_sessions)
        sessions_per_worker = np.full(
            active_workers,
            num_sessions // active_workers,
            dtype=np.int64,
        )
        sessions_per_worker[:num_sessions % active_workers] += 1

        worker_storages = List.empty_list(Storage.class_type.instance_type)
        for worker_index in range(active_workers):
            worker_sessions = int(sessions_per_worker[worker_index])
            estimated_spins = worker_sessions * num_spins
            worker_storages.append(
                Storage(
                    max(1, estimated_spins * 5),
                    max(1, estimated_spins),
                    max(1, worker_sessions),
                    SUPER_SUGAR_RUSH_RULES.num_rows,
                    SUPER_SUGAR_RUSH_RULES.num_reels,
                    SUPER_SUGAR_RUSH_RULES.num_paying_symbols,
                )
            )

        total_win = run_sessions_parallel(
            SUPER_SUGAR_RUSH_INITIAL_REELS,
            SUPER_SUGAR_RUSH_CASCADE_REELS,
            SUPER_SUGAR_RUSH_RULES,
            SUPER_SUGAR_RUSH_SESSION_RULES,
            num_spins,
            sessions_per_worker,
            worker_storages,
        )

        shard_filenames = []
        for worker_index, storage in enumerate(worker_storages):
            shard_filename = (
                f"{output_filename.stem}_worker_{worker_index}.npz"
            )
            write_npz(storage, shard_filename, overwrite=overwrite)
            shard_filenames.append(shard_filename)

        output_path = merge_npz(
            shard_filenames,
            output_filename.name,
            overwrite=overwrite,
        )
    else:
        estimated_spins = num_sessions * num_spins
        storage = Storage(
            max(1, estimated_spins * 5),
            max(1, estimated_spins),
            num_sessions,
            SUPER_SUGAR_RUSH_RULES.num_rows,
            SUPER_SUGAR_RUSH_RULES.num_reels,
            SUPER_SUGAR_RUSH_RULES.num_paying_symbols,
        )
        total_win = run_sessions(
            SUPER_SUGAR_RUSH_INITIAL_REELS,
            SUPER_SUGAR_RUSH_CASCADE_REELS,
            SUPER_SUGAR_RUSH_RULES,
            SUPER_SUGAR_RUSH_SESSION_RULES,
            num_spins,
            num_sessions,
            storage,
        )
        output_path = write_npz(
            storage,
            output_filename.name,
            overwrite=overwrite,
        )

    return total_win, output_path


def _run_self_test():
    from ..core.config import (
        SUPER_SUGAR_RUSH_CASCADE_REELS,
        SUPER_SUGAR_RUSH_INITIAL_REELS,
        SUPER_SUGAR_RUSH_RULES,
        SUPER_SUGAR_RUSH_SESSION_RULES,
    )

    storage = Storage(
        1,
        1,
        1,
        SUPER_SUGAR_RUSH_RULES.num_rows,
        SUPER_SUGAR_RUSH_RULES.num_reels,
        SUPER_SUGAR_RUSH_RULES.num_paying_symbols,
    )
    num_sessions = 2
    total_win = run_sessions(
        SUPER_SUGAR_RUSH_INITIAL_REELS,
        SUPER_SUGAR_RUSH_CASCADE_REELS,
        SUPER_SUGAR_RUSH_RULES,
        SUPER_SUGAR_RUSH_SESSION_RULES,
        2,
        num_sessions,
        storage,
    )

    assert storage.session_count == num_sessions
    assert storage.session_spin_offsets[0] == 0
    assert (
        storage.session_spin_offsets[storage.session_count]
        == storage.spin_count
    )
    assert storage.spin_step_offsets[0] == 0
    assert storage.spin_step_offsets[storage.spin_count] == storage.step_count
    assert np.all(
        np.diff(
            storage.session_spin_offsets[:storage.session_count + 1]
        )
        >= 2
    )
    assert np.all(
        np.diff(storage.spin_step_offsets[:storage.spin_count + 1]) >= 1
    )
    assert np.isclose(
        storage.session_wins[:storage.session_count].sum(),
        total_win,
    )
    assert np.isclose(storage.spin_wins[:storage.spin_count].sum(), total_win)
    assert np.isclose(storage.wins[:storage.step_count].sum(), total_win)
    assert np.allclose(
        storage.symbol_wins[:storage.step_count].sum(axis=1),
        storage.wins[:storage.step_count],
    )
    assert storage.boards.shape[0] > 1
    assert storage.spin_wins.shape[0] > 1
    assert storage.session_wins.shape[0] > 1

    print(
        "super_sugar_rush self-test passed:",
        f"sessions={storage.session_count}",
        f"spins={storage.spin_count}",
        f"steps={storage.step_count}",
        f"total_win={total_win}",
    )


if __name__ == "__main__":
    _run_self_test()
