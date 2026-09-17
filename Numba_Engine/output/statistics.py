"""Statistics calculated from stored simulation NPZ files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_WIN_HISTOGRAM_EDGES = np.array(
    [
        0.0,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        50.0,
        100.0,
        250.0,
        500.0,
        1_000.0,
        np.inf,
    ],
    dtype=np.float64,
)

_BASE_REQUIRED_KEYS = (
    "spin_count",
    "round_count",
    "round_spin_offsets",
    "wins",
    "line_wins",
    "collect_wins",
    "collector_counts",
    "spin_triggers",
)

_HOLD_AND_SPIN_REQUIRED_KEYS = (
    "storage_kind",
    "step_count",
    "respin_count",
    "session_count",
    "feature_types",
    "respin_step_offsets",
    "respin_reset_flags",
    "session_respin_offsets",
    "session_wins",
    "session_coin_wins",
    "session_collector_wins",
    "session_total_respins",
    "session_starting_symbol_counts",
)

_FEATURE_REQUIRED_KEYS = (
    "step_count",
    "spin_count",
    "session_count",
    "spin_step_offsets",
    "spin_wins",
    "spin_triggers",
    "spin_global_after",
    "session_spin_offsets",
    "session_wins",
    "session_initial_spins",
    "session_total_spins",
)


@dataclass(frozen=True)
class BaseGameStatistics:
    """Aggregated statistics for a paid base-game simulation."""

    source_path: Path
    spin_count: int
    step_count: int
    bet_per_spin: float
    total_bet: float
    total_win: float
    total_line_win: float
    total_collect_win: float
    average_collect_win_per_spin: float
    maximum_collect_win: float
    average_collectors_per_spin: float
    collector_active_spin_count: int
    collector_active_spin_rate: float
    maximum_win: float
    rtp: float
    hit_count: int
    hit_rate: float
    trigger_count: int
    trigger_rate: float
    average_win_on_hit: float
    return_variance: float
    return_standard_deviation: float
    average_cascades: float
    maximum_cascades: int
    zero_win_count: int
    win_histogram_edges: np.ndarray
    win_histogram_counts: np.ndarray
    cascade_histogram_counts: np.ndarray

    def pretty_print(self, file=None):
        """Print this summary using the shared pretty-print formatter."""
        from ..serialization.pretty_print import pretty_print_base_game

        pretty_print_base_game(self, file=file)


@dataclass(frozen=True)
class FeatureStatistics:
    """Aggregated statistics for simulations whose top level is a session."""

    source_path: Path
    session_count: int
    spin_count: int
    step_count: int
    bet_per_session: float
    total_bet: float
    total_win: float
    maximum_win: float
    rtp: float
    hit_count: int
    hit_rate: float
    trigger_count: int
    trigger_rate: float
    average_retriggers_per_session: float
    average_spins_per_session: float
    average_win_on_hit: float
    return_variance: float
    return_standard_deviation: float
    average_global_multiplier: float
    average_session_trigger_rate: float
    average_session_hit_rate: float
    average_session_cascade_rate: float
    maximum_session_cascades: int
    zero_win_count: int
    win_histogram_edges: np.ndarray
    win_histogram_counts: np.ndarray
    cascade_histogram_counts: np.ndarray

    def pretty_print(self, file=None):
        """Print this summary using the shared pretty-print formatter."""
        from ..serialization.pretty_print import pretty_print_free_game

        pretty_print_free_game(self, file=file)


@dataclass(frozen=True)
class HoldAndSpinStatistics:
    """Aggregated statistics for Hold-and-Spin feature sessions."""

    source_path: Path
    session_count: int
    respin_count: int
    step_count: int
    bet_per_session: float
    total_bet: float
    total_win: float
    total_coin_win: float
    total_collector_win: float
    maximum_win: float
    rtp: float
    hit_count: int
    hit_rate: float
    average_win_on_hit: float
    return_variance: float
    return_standard_deviation: float
    average_respins_per_session: float
    maximum_respins: int
    average_steps_per_respin: float
    maximum_steps_per_respin: int
    reset_count: int
    reset_rate: float
    average_starting_symbols: float
    zero_win_count: int
    win_histogram_edges: np.ndarray
    win_histogram_counts: np.ndarray
    respin_histogram_counts: np.ndarray
    feature_resolution_counts: np.ndarray

    def pretty_print(self, file=None):
        """Print this summary using the shared Hold-and-Spin formatter."""
        from ..serialization.pretty_print import pretty_print_hold_and_spin

        pretty_print_hold_and_spin(self, file=file)


def _validated_win_edges(win_histogram_edges):
    if win_histogram_edges is None:
        return DEFAULT_WIN_HISTOGRAM_EDGES.copy()

    edges = np.asarray(win_histogram_edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("win_histogram_edges must be a one-dimensional array")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("win_histogram_edges must be strictly increasing")
    if edges[0] > 0 or not np.isposinf(edges[-1]):
        raise ValueError(
            "win_histogram_edges must start at or below zero and end at +inf"
        )
    return edges.copy()


def store_base_game(
    npz_path,
    bet_per_spin=1.0,
    win_histogram_edges=None,
    print_result=True,
):
    """Load, aggregate, and optionally print base-game statistics."""
    source_path = Path(npz_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Base-game NPZ does not exist: {source_path}")
    if bet_per_spin <= 0:
        raise ValueError("bet_per_spin must be positive")

    edges = _validated_win_edges(win_histogram_edges)

    with np.load(source_path, allow_pickle=False) as data:
        missing = [key for key in _BASE_REQUIRED_KEYS if key not in data]
        if missing:
            raise ValueError(
                f"Base-game NPZ is missing required arrays: {missing}"
            )

        spin_count = int(data["spin_count"])
        round_count = int(data["round_count"])
        step_count = spin_count
        spin_wins = np.asarray(data["wins"], dtype=np.float64)
        line_wins = np.asarray(data["line_wins"], dtype=np.float64)
        collect_wins = np.asarray(data["collect_wins"], dtype=np.float64)
        collector_counts = np.asarray(data["collector_counts"], dtype=np.int64)
        spin_triggers = np.asarray(data["spin_triggers"], dtype=np.int8)
        round_spin_offsets = np.asarray(
            data["round_spin_offsets"],
            dtype=np.int64,
        )

    if spin_count < 1:
        raise ValueError("Base-game NPZ must contain at least one spin")
    if round_count < 1:
        raise ValueError("Base-game NPZ must contain at least one round")
    if spin_wins.shape != (spin_count,):
        raise ValueError("spin_wins does not match spin_count")
    if line_wins.shape != (spin_count,):
        raise ValueError("line_wins does not match spin_count")
    if collect_wins.shape != (spin_count,):
        raise ValueError("collect_wins does not match spin_count")
    if collector_counts.shape != (spin_count,):
        raise ValueError("collector_counts does not match spin_count")
    if spin_triggers.shape != (spin_count,):
        raise ValueError("spin_triggers does not match spin_count")
    if round_spin_offsets.shape != (round_count + 1,):
        raise ValueError("round_spin_offsets does not match round_count")
    if round_spin_offsets[0] != 0 or round_spin_offsets[-1] != spin_count:
        raise ValueError("round_spin_offsets does not cover all paid spins")
    if np.any(np.diff(round_spin_offsets) < 1):
        raise ValueError("Every round must contain at least one paid spin")
    if np.any(spin_wins < 0):
        raise ValueError("spin_wins cannot contain negative values")
    if np.any(line_wins < 0) or np.any(collect_wins < 0):
        raise ValueError("Line and Collector wins cannot be negative")
    if np.any(collector_counts < 0):
        raise ValueError("collector_counts cannot contain negative values")
    if not np.allclose(line_wins + collect_wins, spin_wins):
        raise ValueError("Line and Collector wins do not reconcile with wins")
    if np.any((spin_triggers != 0) & (spin_triggers != 1)):
        raise ValueError("spin_triggers must contain only 0 or 1")

    normalized_wins = spin_wins / bet_per_spin
    positive_wins = normalized_wins[normalized_wins > 0]
    hit_count = int(positive_wins.size)
    trigger_count = int(spin_triggers.sum(dtype=np.int64))
    cascade_counts = np.zeros(spin_count, dtype=np.int64)

    win_histogram_counts, _ = np.histogram(
        positive_wins,
        bins=edges,
    )
    cascade_histogram_counts = np.bincount(cascade_counts).astype(
        np.int64,
        copy=False,
    )

    total_bet = float(spin_count * bet_per_spin)
    total_win = float(spin_wins.sum(dtype=np.float64))
    total_line_win = float(line_wins.sum(dtype=np.float64))
    total_collect_win = float(collect_wins.sum(dtype=np.float64))
    collector_active_spin_count = int(np.count_nonzero(collector_counts))

    statistics = BaseGameStatistics(
        source_path=source_path.resolve(),
        spin_count=spin_count,
        step_count=step_count,
        bet_per_spin=float(bet_per_spin),
        total_bet=total_bet,
        total_win=total_win,
        total_line_win=total_line_win,
        total_collect_win=total_collect_win,
        average_collect_win_per_spin=total_collect_win / spin_count,
        maximum_collect_win=float(collect_wins.max()),
        average_collectors_per_spin=float(collector_counts.mean()),
        collector_active_spin_count=collector_active_spin_count,
        collector_active_spin_rate=collector_active_spin_count / spin_count,
        maximum_win=float(spin_wins.max()),
        rtp=total_win / total_bet,
        hit_count=hit_count,
        hit_rate=hit_count / spin_count,
        trigger_count=trigger_count,
        trigger_rate=trigger_count / spin_count,
        average_win_on_hit=(
            float(spin_wins[spin_wins > 0].mean()) if hit_count else 0.0
        ),
        return_variance=float(np.var(normalized_wins)),
        return_standard_deviation=float(np.std(normalized_wins)),
        average_cascades=float(cascade_counts.mean()),
        maximum_cascades=int(cascade_counts.max()),
        zero_win_count=spin_count - hit_count,
        win_histogram_edges=edges,
        win_histogram_counts=win_histogram_counts,
        cascade_histogram_counts=cascade_histogram_counts,
    )
    if print_result:
        statistics.pretty_print()
    return statistics


def store_free_game(
    npz_path,
    bet_per_session=1.0,
    win_histogram_edges=None,
    print_result=True,
):
    """Load, aggregate, and optionally print free-game session statistics."""
    source_path = Path(npz_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Free-game NPZ does not exist: {source_path}")
    if bet_per_session <= 0:
        raise ValueError("bet_per_session must be positive")

    edges = _validated_win_edges(win_histogram_edges)

    with np.load(source_path, allow_pickle=False) as data:
        missing = [key for key in _FEATURE_REQUIRED_KEYS if key not in data]
        if missing:
            raise ValueError(
                f"Free-game NPZ is missing required arrays: {missing}"
            )

        session_count = int(data["session_count"])
        spin_count = int(data["spin_count"])
        step_count = int(data["step_count"])
        spin_wins = np.asarray(data["spin_wins"], dtype=np.float64)
        spin_triggers = np.asarray(data["spin_triggers"], dtype=np.int8)
        spin_global_after = np.asarray(
            data["spin_global_after"],
            dtype=np.int64,
        )
        spin_step_offsets = np.asarray(
            data["spin_step_offsets"],
            dtype=np.int64,
        )
        session_spin_offsets = np.asarray(
            data["session_spin_offsets"],
            dtype=np.int64,
        )
        session_wins = np.asarray(data["session_wins"], dtype=np.float64)
        session_initial_spins = np.asarray(
            data["session_initial_spins"],
            dtype=np.int64,
        )
        session_total_spins = np.asarray(
            data["session_total_spins"],
            dtype=np.int64,
        )

    if session_count < 1:
        raise ValueError("store_free_game requires an NPZ containing sessions")
    if spin_count < 1:
        raise ValueError("Free-game NPZ must contain at least one spin")
    if spin_wins.shape != (spin_count,):
        raise ValueError("spin_wins does not match spin_count")
    if spin_triggers.shape != (spin_count,):
        raise ValueError("spin_triggers does not match spin_count")
    if spin_global_after.shape != (spin_count,):
        raise ValueError("spin_global_after does not match spin_count")
    if spin_step_offsets.shape != (spin_count + 1,):
        raise ValueError("spin_step_offsets does not match spin_count")
    if session_spin_offsets.shape != (session_count + 1,):
        raise ValueError("session_spin_offsets does not match session_count")
    for name, values in (
        ("session_wins", session_wins),
        ("session_initial_spins", session_initial_spins),
        ("session_total_spins", session_total_spins),
    ):
        if values.shape != (session_count,):
            raise ValueError(f"{name} does not match session_count")

    if spin_step_offsets[0] != 0 or spin_step_offsets[-1] != step_count:
        raise ValueError("spin_step_offsets does not cover all stored steps")
    if np.any(np.diff(spin_step_offsets) < 1):
        raise ValueError("Every spin must contain at least one evaluation step")
    if session_spin_offsets[0] != 0 or session_spin_offsets[-1] != spin_count:
        raise ValueError("session_spin_offsets does not cover all stored spins")

    session_spin_counts = np.diff(session_spin_offsets)
    if np.any(session_spin_counts < 1):
        raise ValueError("Every session must contain at least one spin")
    if not np.array_equal(session_total_spins, session_spin_counts):
        raise ValueError("session_total_spins does not match stored spin offsets")
    if np.any(session_initial_spins < 1):
        raise ValueError("session_initial_spins must be positive")
    if np.any(session_initial_spins > session_total_spins):
        raise ValueError("A session cannot contain fewer than its initial spins")
    if np.any(spin_wins < 0) or np.any(session_wins < 0):
        raise ValueError("Win arrays cannot contain negative values")
    if np.any((spin_triggers != 0) & (spin_triggers != 1)):
        raise ValueError("spin_triggers must contain only 0 or 1")

    session_starts = session_spin_offsets[:-1]
    session_ends = session_spin_offsets[1:]
    spin_cascades = np.diff(spin_step_offsets) - 1
    spin_hits = (spin_wins > 0).astype(np.int64)

    calculated_session_wins = np.add.reduceat(spin_wins, session_starts)
    if not np.allclose(calculated_session_wins, session_wins):
        raise ValueError("session_wins does not match the contained spin wins")

    session_hits = np.add.reduceat(spin_hits, session_starts)
    session_triggers = np.add.reduceat(spin_triggers, session_starts)
    session_cascades = np.add.reduceat(spin_cascades, session_starts)
    final_global_multipliers = spin_global_after[session_ends - 1]

    normalized_session_wins = session_wins / bet_per_session
    positive_session_wins = normalized_session_wins[
        normalized_session_wins > 0
    ]
    hit_count = int(positive_session_wins.size)
    trigger_count = int(spin_triggers.sum(dtype=np.int64))

    win_histogram_counts, _ = np.histogram(
        positive_session_wins,
        bins=edges,
    )
    cascade_histogram_counts = np.bincount(session_cascades).astype(
        np.int64,
        copy=False,
    )

    total_bet = float(session_count * bet_per_session)
    total_win = float(session_wins.sum(dtype=np.float64))

    statistics = FeatureStatistics(
        source_path=source_path.resolve(),
        session_count=session_count,
        spin_count=spin_count,
        step_count=step_count,
        bet_per_session=float(bet_per_session),
        total_bet=total_bet,
        total_win=total_win,
        maximum_win=float(session_wins.max()),
        rtp=total_win / total_bet,
        hit_count=hit_count,
        hit_rate=hit_count / session_count,
        trigger_count=trigger_count,
        trigger_rate=trigger_count / spin_count,
        average_retriggers_per_session=float(session_triggers.mean()),
        average_spins_per_session=float(session_spin_counts.mean()),
        average_win_on_hit=(
            float(session_wins[session_wins > 0].mean())
            if hit_count
            else 0.0
        ),
        return_variance=float(np.var(normalized_session_wins)),
        return_standard_deviation=float(np.std(normalized_session_wins)),
        average_global_multiplier=float(final_global_multipliers.mean()),
        average_session_trigger_rate=float(
            np.mean(session_triggers / session_spin_counts)
        ),
        average_session_hit_rate=float(
            np.mean(session_hits / session_spin_counts)
        ),
        average_session_cascade_rate=float(
            np.mean(session_cascades / session_spin_counts)
        ),
        maximum_session_cascades=int(session_cascades.max()),
        zero_win_count=session_count - hit_count,
        win_histogram_edges=edges,
        win_histogram_counts=win_histogram_counts,
        cascade_histogram_counts=cascade_histogram_counts,
    )
    if print_result:
        statistics.pretty_print()
    return statistics


def store_hold_and_spin(
    npz_path,
    bet_per_session=1.0,
    win_histogram_edges=None,
    print_result=True,
):
    """Load and aggregate Hold-and-Spin session statistics."""
    source_path = Path(npz_path)
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Hold-and-Spin NPZ does not exist: {source_path}"
        )
    if bet_per_session <= 0:
        raise ValueError("bet_per_session must be positive")

    edges = _validated_win_edges(win_histogram_edges)

    with np.load(source_path, allow_pickle=False) as data:
        missing = [
            key for key in _HOLD_AND_SPIN_REQUIRED_KEYS if key not in data
        ]
        if missing:
            raise ValueError(
                f"Hold-and-Spin NPZ is missing required arrays: {missing}"
            )
        if str(data["storage_kind"]) != "hold_and_spin":
            raise ValueError("NPZ is not a Hold-and-Spin storage archive")

        step_count = int(data["step_count"])
        respin_count = int(data["respin_count"])
        session_count = int(data["session_count"])
        feature_types = np.asarray(data["feature_types"], dtype=np.int8)
        respin_step_offsets = np.asarray(
            data["respin_step_offsets"],
            dtype=np.int64,
        )
        respin_reset_flags = np.asarray(
            data["respin_reset_flags"],
            dtype=np.bool_,
        )
        session_respin_offsets = np.asarray(
            data["session_respin_offsets"],
            dtype=np.int64,
        )
        session_wins = np.asarray(data["session_wins"], dtype=np.float64)
        session_coin_wins = np.asarray(
            data["session_coin_wins"],
            dtype=np.float64,
        )
        session_collector_wins = np.asarray(
            data["session_collector_wins"],
            dtype=np.float64,
        )
        session_total_respins = np.asarray(
            data["session_total_respins"],
            dtype=np.int64,
        )
        session_starting_symbol_counts = np.asarray(
            data["session_starting_symbol_counts"],
            dtype=np.int64,
        )

    if session_count < 1:
        raise ValueError("Hold-and-Spin NPZ must contain at least one session")
    if respin_count < 1 or step_count < 1:
        raise ValueError("Every archive must contain respins and steps")

    for name, values in (
        ("session_wins", session_wins),
        ("session_coin_wins", session_coin_wins),
        ("session_collector_wins", session_collector_wins),
        ("session_total_respins", session_total_respins),
        ("session_starting_symbol_counts", session_starting_symbol_counts),
    ):
        if values.shape != (session_count,):
            raise ValueError(f"{name} does not match session_count")
    if feature_types.shape != (step_count,):
        raise ValueError("feature_types does not match step_count")
    if respin_reset_flags.shape != (respin_count,):
        raise ValueError("respin_reset_flags does not match respin_count")
    if respin_step_offsets.shape != (respin_count + 1,):
        raise ValueError("respin_step_offsets does not match respin_count")
    if session_respin_offsets.shape != (session_count + 1,):
        raise ValueError("session_respin_offsets does not match session_count")
    if (
        respin_step_offsets[0] != 0
        or respin_step_offsets[-1] != step_count
        or np.any(np.diff(respin_step_offsets) < 1)
    ):
        raise ValueError("respin_step_offsets does not cover all steps")
    if (
        session_respin_offsets[0] != 0
        or session_respin_offsets[-1] != respin_count
        or np.any(np.diff(session_respin_offsets) < 1)
    ):
        raise ValueError("session_respin_offsets does not cover all respins")

    respins_per_session = np.diff(session_respin_offsets)
    steps_per_respin = np.diff(respin_step_offsets)
    if not np.array_equal(session_total_respins, respins_per_session):
        raise ValueError(
            "session_total_respins does not match session-respin offsets"
        )
    if np.any(session_wins < 0):
        raise ValueError("session_wins cannot contain negative values")
    if not np.allclose(
        session_coin_wins + session_collector_wins,
        session_wins,
    ):
        raise ValueError(
            "Coin and Collector wins do not reconcile with session wins"
        )
    if np.any(session_starting_symbol_counts < 1):
        raise ValueError("Every session must contain a starting Bag symbol")
    if np.any((feature_types < -2) | (feature_types > 5)):
        raise ValueError("feature_types contains an unknown event code")

    normalized_wins = session_wins / bet_per_session
    positive_wins = normalized_wins[normalized_wins > 0]
    hit_count = int(positive_wins.size)
    win_histogram_counts, _ = np.histogram(positive_wins, bins=edges)
    respin_histogram_counts = np.bincount(respins_per_session).astype(
        np.int64,
        copy=False,
    )
    feature_resolution_counts = np.bincount(
        feature_types[feature_types >= 0],
        minlength=6,
    ).astype(np.int64, copy=False)

    total_bet = float(session_count * bet_per_session)
    total_win = float(session_wins.sum(dtype=np.float64))
    reset_count = int(respin_reset_flags.sum(dtype=np.int64))

    statistics = HoldAndSpinStatistics(
        source_path=source_path.resolve(),
        session_count=session_count,
        respin_count=respin_count,
        step_count=step_count,
        bet_per_session=float(bet_per_session),
        total_bet=total_bet,
        total_win=total_win,
        total_coin_win=float(session_coin_wins.sum(dtype=np.float64)),
        total_collector_win=float(
            session_collector_wins.sum(dtype=np.float64)
        ),
        maximum_win=float(session_wins.max()),
        rtp=total_win / total_bet,
        hit_count=hit_count,
        hit_rate=hit_count / session_count,
        average_win_on_hit=(
            float(session_wins[session_wins > 0].mean())
            if hit_count
            else 0.0
        ),
        return_variance=float(np.var(normalized_wins)),
        return_standard_deviation=float(np.std(normalized_wins)),
        average_respins_per_session=float(respins_per_session.mean()),
        maximum_respins=int(respins_per_session.max()),
        average_steps_per_respin=float(steps_per_respin.mean()),
        maximum_steps_per_respin=int(steps_per_respin.max()),
        reset_count=reset_count,
        reset_rate=reset_count / respin_count,
        average_starting_symbols=float(session_starting_symbol_counts.mean()),
        zero_win_count=session_count - hit_count,
        win_histogram_edges=edges,
        win_histogram_counts=win_histogram_counts,
        respin_histogram_counts=respin_histogram_counts,
        feature_resolution_counts=feature_resolution_counts,
    )
    if print_result:
        statistics.pretty_print()
    return statistics


__all__ = [
    "BaseGameStatistics",
    "DEFAULT_WIN_HISTOGRAM_EDGES",
    "FeatureStatistics",
    "HoldAndSpinStatistics",
    "store_base_game",
    "store_free_game",
    "store_hold_and_spin",
]
