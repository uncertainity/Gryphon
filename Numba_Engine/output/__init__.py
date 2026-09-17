"""Generated simulation output and reporting helpers."""

from ..serialization.pretty_print import (
    format_base_game_statistics,
    format_free_game_statistics,
    format_hold_and_spin_statistics,
    pretty_print_base_game,
    pretty_print_free_game,
    pretty_print_hold_and_spin,
)


__all__ = [
    "BaseGameStatistics",
    "FeatureStatistics",
    "HoldAndSpinStatistics",
    "format_base_game_statistics",
    "format_free_game_statistics",
    "format_hold_and_spin_statistics",
    "pretty_print_base_game",
    "pretty_print_free_game",
    "pretty_print_hold_and_spin",
    "store_base_game",
    "store_free_game",
    "store_hold_and_spin",
]


def __getattr__(name):
    statistics_names = {
        "BaseGameStatistics",
        "FeatureStatistics",
        "HoldAndSpinStatistics",
        "store_base_game",
        "store_free_game",
        "store_hold_and_spin",
    }
    if name in statistics_names:
        from .statistics import (
            BaseGameStatistics,
            FeatureStatistics,
            HoldAndSpinStatistics,
            store_base_game,
            store_free_game,
            store_hold_and_spin,
        )

        exports = {
            "BaseGameStatistics": BaseGameStatistics,
            "FeatureStatistics": FeatureStatistics,
            "HoldAndSpinStatistics": HoldAndSpinStatistics,
            "store_base_game": store_base_game,
            "store_free_game": store_free_game,
            "store_hold_and_spin": store_hold_and_spin,
        }
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
