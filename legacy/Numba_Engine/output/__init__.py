"""Generated simulation output and reporting helpers."""

from ..serialization.pretty_print import (
    format_base_game_statistics,
    format_free_game_statistics,
    format_full_game_statistics,
    format_hold_and_spin_statistics,
    pretty_print_base_game,
    pretty_print_free_game,
    pretty_print_full_game,
    pretty_print_hold_and_spin,
)


__all__ = [
    "BaseGameStatistics",
    "FeatureStatistics",
    "FullGameStatistics",
    "HoldAndSpinStatistics",
    "format_base_game_statistics",
    "format_free_game_statistics",
    "format_full_game_statistics",
    "format_hold_and_spin_statistics",
    "pretty_print_base_game",
    "pretty_print_free_game",
    "pretty_print_full_game",
    "pretty_print_hold_and_spin",
    "store_base_game",
    "store_free_game",
    "store_full_game",
    "store_hold_and_spin",
]


def __getattr__(name):
    statistics_names = {
        "BaseGameStatistics",
        "FeatureStatistics",
        "FullGameStatistics",
        "HoldAndSpinStatistics",
        "store_base_game",
        "store_free_game",
        "store_full_game",
        "store_hold_and_spin",
    }
    if name in statistics_names:
        from .statistics import (
            BaseGameStatistics,
            FeatureStatistics,
            FullGameStatistics,
            HoldAndSpinStatistics,
            store_base_game,
            store_free_game,
            store_full_game,
            store_hold_and_spin,
        )

        exports = {
            "BaseGameStatistics": BaseGameStatistics,
            "FeatureStatistics": FeatureStatistics,
            "FullGameStatistics": FullGameStatistics,
            "HoldAndSpinStatistics": HoldAndSpinStatistics,
            "store_base_game": store_base_game,
            "store_free_game": store_free_game,
            "store_full_game": store_full_game,
            "store_hold_and_spin": store_hold_and_spin,
        }
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
