"""Simulation entry points, loaded lazily to support ``python -m`` usage."""


__all__ = [
    "run_base_sims",
    "run_base_spins",
    "run_base_spins_parallel",
    "run_one_base_spin",
    "hold_and_free_spin",
    "run_hold_and_spin_sims",
    "run_parallel_num_sessions",
    "run_sessions",
    "select_starting_bag_symbols",
]


def __getattr__(name):
    base_names = {
        "run_base_sims",
        "run_base_spins",
        "run_base_spins_parallel",
        "run_one_base_spin",
    }
    if name in base_names:
        from .base_game import (
            run_one_spin,
            run_sims,
            run_spins,
            run_spins_parallel,
        )

        exports = {
            "run_base_sims": run_sims,
            "run_base_spins": run_spins,
            "run_base_spins_parallel": run_spins_parallel,
            "run_one_base_spin": run_one_spin,
        }
        globals().update(exports)
        return exports[name]

    hold_names = {
        "hold_and_free_spin",
        "run_hold_and_spin_sims",
        "run_parallel_num_sessions",
        "run_sessions",
        "select_starting_bag_symbols",
    }
    if name in hold_names:
        from .free_game import (
            hold_and_free_spin,
            run_parallel_num_sessions,
            run_sessions,
            run_sims,
            select_starting_bag_symbols,
        )

        exports = {
            "hold_and_free_spin": hold_and_free_spin,
            "run_hold_and_spin_sims": run_sims,
            "run_parallel_num_sessions": run_parallel_num_sessions,
            "run_sessions": run_sessions,
            "select_starting_bag_symbols": select_starting_bag_symbols,
        }
        globals().update(exports)
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
