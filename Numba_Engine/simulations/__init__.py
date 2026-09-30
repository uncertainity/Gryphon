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
    "run_boost_feature",
    "run_collect_feature",
    "run_expansion_feature",
    "run_full_game_sims",
    "run_full_rounds",
    "run_grow_feature",
    "run_mega_combo_feature",
    "run_multiplier_feature",
    "run_splitter_feature",
    "run_one_full_round",
    "validate_full_game_config",
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

    feature_flow_names = {
        "run_boost_feature",
        "run_collect_feature",
        "run_expansion_feature",
        "run_grow_feature",
        "run_mega_combo_feature",
        "run_multiplier_feature",
        "run_splitter_feature",
    }
    if name in feature_flow_names:
        from .feature_flows import (
            run_boost_feature,
            run_collect_feature,
            run_expansion_feature,
            run_grow_feature,
            run_mega_combo_feature,
            run_multiplier_feature,
            run_splitter_feature,
        )

        exports = {
            "run_boost_feature": run_boost_feature,
            "run_collect_feature": run_collect_feature,
            "run_expansion_feature": run_expansion_feature,
            "run_grow_feature": run_grow_feature,
            "run_mega_combo_feature": run_mega_combo_feature,
            "run_multiplier_feature": run_multiplier_feature,
            "run_splitter_feature": run_splitter_feature,
        }
        globals().update(exports)
        return exports[name]

    full_game_names = {
        "run_full_game_sims",
        "run_full_rounds",
        "run_one_full_round",
        "validate_full_game_config",
    }
    if name in full_game_names:
        from .full_game import (
            run_full_rounds,
            run_one_full_round,
            run_sims,
            validate_full_game_config,
        )

        exports = {
            "run_full_game_sims": run_sims,
            "run_full_rounds": run_full_rounds,
            "run_one_full_round": run_one_full_round,
            "validate_full_game_config": validate_full_game_config,
        }
        globals().update(exports)
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
