"""Evaluate Gryphon tuning candidates through the production Numba kernels.

Examples
--------
Run a quick route-frequency and conditional-award calibration::

    python -m Numba_Engine.Tuning.evaluate \
        --routing-spins 1m --feature-sessions 20k

The routing pass uses the real reel loader, board generator, generic-SC
converter, and route selector.  The feature pass calls the real shared
Hold-and-Spin kernel.  This module contains measurement code only; it does not
reimplement or bypass game mechanics.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
from numba import float64, int64, njit
from numba.experimental import jitclass

from ..core.config import FULL_GAME_CONFIG
from ..core.kernels import (
    has_free_game_trigger,
    make_board,
    select_reelset_index,
)
from ..core.reels import make_reel_collection
from ..core.storage import CompactBaseStorage
from ..simulations.base_game import (
    _run_one_paid_spin_with_features,
    convert_base_scatters,
)
from ..simulations.free_game import (
    FEATURE_NAMES,
    _new_storage,
    hold_and_free_spin,
    validate_hold_and_spin_config,
)
from ..simulations.full_game import (
    run_full_rounds,
    select_feature_route,
    starting_bags_for_route,
    validate_full_game_config,
)
from .targets import reconciled_feature_probabilities, reconciled_rtp_budget


TARGET_PATH = Path(__file__).with_name("rtp_targets.csv")


compact_hold_and_spin_storage_spec = [
    ("step_count", int64),
    ("respin_count", int64),
    ("session_count", int64),
]


base_tuning_storage_spec = [
    ("spin_count", int64),
    ("round_count", int64),
    ("collector_active_spin_count", int64),
    ("collector_symbol_count", int64),
    ("collected_coin_cell_count", int64),
    ("coin_drop_spin_count", int64),
    ("coin_drop_cell_count", int64),
]


@jitclass(base_tuning_storage_spec)
class BaseTuningStorage:
    """Aggregate base sink for the new Collect-frequency constraints."""

    def __init__(self):
        self.spin_count = 0
        self.round_count = 0
        self.collector_active_spin_count = 0
        self.collector_symbol_count = 0
        self.collected_coin_cell_count = 0
        self.coin_drop_spin_count = 0
        self.coin_drop_cell_count = 0

    def begin_round(self):
        return

    def finish_round(self, round_win, round_triggers):
        self.round_count += 1

    def save_spin(
        self,
        board,
        coin_value_board,
        jackpot_overlay_board,
        jackpot_values_before,
        jackpot_values_after,
        jackpot_increment_counts,
        line_win,
        collect_win,
        symbol_wins,
        symbol_hit_counts,
        line_winning_symbols,
        line_match_counts,
        line_win_amounts,
        free_game_trigger,
        collector_count,
    ):
        coin_cells = 0
        for value in coin_value_board.ravel():
            if value > 0.0:
                coin_cells += 1
        if coin_cells > 0:
            self.coin_drop_spin_count += 1
            self.coin_drop_cell_count += coin_cells
        if collector_count > 0:
            self.collector_active_spin_count += 1
            self.collector_symbol_count += collector_count
            self.collected_coin_cell_count += coin_cells * collector_count
        self.spin_count += 1


@jitclass(compact_hold_and_spin_storage_spec)
class CompactHoldAndSpinTuningStorage:
    """Counter-only storage accepted by the production H&S kernel."""

    def __init__(self):
        self.step_count = 0
        self.respin_count = 0
        self.session_count = 0

    def begin_session(self, starting_bag_symbols):
        return

    def finish_session(
        self,
        session_win,
        coin_win,
        collector_win,
        jackpot_meters,
        jackpot_awards,
    ):
        self.session_count += 1

    def begin_respin(self, remaining_spins):
        return

    def finish_respin(self, remaining_spins, reset_flag):
        self.respin_count += 1

    def save_jackpot_respin(
        self,
        overlay_board,
        meters_before,
        meters_after,
        jackpot_awards,
    ):
        return

    def save_step(
        self,
        board,
        coin_mask,
        feature_type,
        feature_position,
        remaining_spins,
        locked_row_idx,
        collector_meter,
    ):
        self.step_count += 1


compact_full_game_storage_spec = [
    ("feature_trigger_counts", int64[:]),
    ("feature_win_amounts", float64[:]),
    ("feature_spin_counts", int64[:]),
    ("jackpot_award_counts", int64[:]),
    ("jackpot_award_amounts", float64[:]),
    ("base_line_win_amount", float64),
    ("base_collect_win_amount", float64),
    ("base_positive_spin_count", int64),
    ("spin_count", int64),
    ("round_count", int64),
    ("feature_session_count", int64),
]


@jitclass(compact_full_game_storage_spec)
class CompactFullGameTuningStorage:
    """Aggregate-only sink for production integrated-game evaluation."""

    def __init__(self, num_feature_types, num_jackpots):
        self.feature_trigger_counts = np.zeros(
            num_feature_types, dtype=np.int64
        )
        self.feature_win_amounts = np.zeros(
            num_feature_types, dtype=np.float64
        )
        self.feature_spin_counts = np.zeros(
            num_feature_types, dtype=np.int64
        )
        self.jackpot_award_counts = np.zeros(
            num_jackpots, dtype=np.int64
        )
        self.jackpot_award_amounts = np.zeros(
            num_jackpots, dtype=np.float64
        )
        self.base_line_win_amount = 0.0
        self.base_collect_win_amount = 0.0
        self.base_positive_spin_count = 0
        self.spin_count = 0
        self.round_count = 0
        self.feature_session_count = 0

    def begin_round(self):
        return

    def save_spin(
        self,
        feature_trigger_counts,
        feature_wins_by_type,
        feature_spins_by_type,
        base_win,
        line_win,
        collect_win,
        jackpot_win,
        jackpot_values_before_feature,
        jackpot_awards,
        jackpot_award_amounts,
        jackpot_values_after_feature,
    ):
        self.feature_trigger_counts += feature_trigger_counts
        self.feature_win_amounts += feature_wins_by_type
        self.feature_spin_counts += feature_spins_by_type
        self.jackpot_award_counts += jackpot_awards
        self.jackpot_award_amounts += jackpot_award_amounts
        self.base_line_win_amount += line_win
        self.base_collect_win_amount += collect_win
        self.base_positive_spin_count += int(base_win > 0.0)
        feature_count = int(feature_trigger_counts.sum())
        self.feature_session_count += feature_count
        self.spin_count += 1

    def finish_round(self, base_win, feature_win, jackpot_win):
        self.round_count += 1


def _positive_count(value: str) -> int:
    """Parse positive integer counts with optional ``k``/``m`` suffixes."""
    text = value.strip().lower().replace("_", "").replace(",", "")
    multiplier = 1
    if text.endswith("k"):
        text = text[:-1]
        multiplier = 1_000
    elif text.endswith("m"):
        text = text[:-1]
        multiplier = 1_000_000
    try:
        result = int(text) * multiplier
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer count, got {value!r}"
        ) from error
    if result < 1:
        raise argparse.ArgumentTypeError("count must be positive")
    return result


def load_feature_targets(path: Path = TARGET_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Load paid-spin probabilities and conditional means in engine order."""
    aliases = {
        "Plain": "Plain Hold-and-Spin",
        "Collect": "Feature Collect",
    }
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as target_file:
        for row in csv.DictReader(target_file):
            if row["category"] == "feature":
                rows[row["component"]] = row

    probabilities, _ = reconciled_feature_probabilities()
    conditional_means = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    for feature_index, feature_name in enumerate(FEATURE_NAMES):
        target_name = aliases.get(feature_name, feature_name)
        row = rows[target_name]
        conditional_means[feature_index] = float(
            row["target_conditional_mean_win"]
        )
    return probabilities, conditional_means


def load_game_targets(path: Path = TARGET_PATH) -> dict[str, float]:
    """Load the top-level and base component targets used by reports."""
    result = {}
    with path.open(newline="", encoding="utf-8") as target_file:
        for row in csv.DictReader(target_file):
            component = row["component"]
            if row["target_rtp"]:
                key = component if component.endswith("RTP") else f"{component} RTP"
                result[key] = float(row["target_rtp"])
            if row["target_base_game_hit_rate"]:
                result["Base hit rate"] = float(
                    row["target_base_game_hit_rate"]
                )
            if row["target_hold_and_spin_trigger_rate"]:
                result["H&S trigger rate"] = float(
                    row["target_hold_and_spin_trigger_rate"]
                )
    _, trigger_diagnostics = reconciled_feature_probabilities()
    # The new sheet's fallback fixes base RTP and treats the overall/Mega
    # trigger targets as hard constraints. Component RTPs remain useful as
    # reporting references until the final calibrated mix is written back.
    result.update(reconciled_rtp_budget())
    result["Base hit rate"] = 0.25
    result["H&S trigger rate"] = trigger_diagnostics[
        "overall_probability"
    ]
    return result


@njit
def _evaluate_base_kernel(initial_reels, config, num_spins, seed):
    """Measure base math by calling the production paid-spin resolver."""
    np.random.seed(seed)
    storage = BaseTuningStorage()
    jackpot_values = config.jackpots.seed_values.copy()
    collector_positions = np.empty(
        config.base_game.max_active_collectors, dtype=np.int32
    )
    collector_count = 0
    total_win = 0.0
    line_win = 0.0
    collect_win = 0.0
    positive_base_spins = 0
    positive_line_spins = 0
    positive_collect_spins = 0
    collect_only_spins = 0
    trigger_count = 0

    for _ in range(num_spins):
        (
            spin_win,
            triggered,
            collector_count,
            _,
            _,
            _,
            spin_line_win,
            spin_collect_win,
        ) = _run_one_paid_spin_with_features(
            initial_reels,
            config.base_game,
            config.base_jackpot_overlay,
            config.jackpots,
            storage,
            jackpot_values,
            collector_positions,
            collector_count,
        )
        total_win += spin_win
        line_win += spin_line_win
        collect_win += spin_collect_win
        positive_base_spins += int(spin_win > 0.0)
        positive_line_spins += int(spin_line_win > 0.0)
        positive_collect_spins += int(spin_collect_win > 0.0)
        collect_only_spins += int(
            spin_line_win == 0.0 and spin_collect_win > 0.0
        )
        trigger_count += int(triggered)

    return (
        total_win,
        line_win,
        collect_win,
        positive_base_spins,
        positive_line_spins,
        positive_collect_spins,
        collect_only_spins,
        trigger_count,
        storage.collector_active_spin_count,
        storage.collector_symbol_count,
        storage.collected_coin_cell_count,
        storage.coin_drop_spin_count,
        storage.coin_drop_cell_count,
    )


@njit
def _evaluate_routing_kernel(initial_reels, config, num_spins, seed):
    """Measure route selection using the complete production base path."""
    np.random.seed(seed)
    num_routes = len(config.hold_and_spin.bag_symbols) + 2
    route_counts = np.zeros(num_routes, dtype=np.int64)
    reelset_counts = np.zeros(len(initial_reels.weights), dtype=np.int64)
    trigger_counts_by_reelset = np.zeros(
        len(initial_reels.weights), dtype=np.int64
    )
    sc_count_histogram = np.zeros(
        config.base_game.num_rows * config.base_game.num_reels + 1,
        dtype=np.int64,
    )
    distinct_bag_histogram = np.zeros(
        len(config.hold_and_spin.bag_symbols) + 1,
        dtype=np.int64,
    )

    for _ in range(num_spins):
        reelset_index = select_reelset_index(initial_reels.weights)
        reelset_counts[reelset_index] += 1
        pay_window = make_board(
            initial_reels.lengths[reelset_index],
            initial_reels.reelsets[reelset_index],
            config.base_game,
        )
        if not has_free_game_trigger(pay_window, config.base_game):
            continue

        trigger_counts_by_reelset[reelset_index] += 1
        sc_count = 0
        for symbol in pay_window.ravel():
            if symbol == config.base_game.sc_symbol:
                sc_count += 1
        sc_count_histogram[sc_count] += 1

        _, feature_flags, combo_triggered = convert_base_scatters(
            pay_window,
            config.base_game,
        )
        distinct_count = 0
        for flag in feature_flags:
            if flag:
                distinct_count += 1
        distinct_bag_histogram[distinct_count] += 1

        route_index = select_feature_route(
            feature_flags,
            combo_triggered,
            config.hold_and_spin,
        )
        route_counts[route_index] += 1

    return (
        route_counts,
        reelset_counts,
        trigger_counts_by_reelset,
        sc_count_histogram,
        distinct_bag_histogram,
    )


@njit
def _evaluate_feature_kernel(
    starting_bag_symbols,
    rules,
    num_sessions,
    storage,
    seed,
):
    """Measure one route by repeatedly calling the shared production H&S."""
    np.random.seed(seed)
    total_raw_win = 0.0
    total_respins = 0
    jackpot_awards = np.zeros(
        len(rules.jackpot_collection_targets), dtype=np.int64
    )
    maximum_raw_win = 0.0

    for _ in range(num_sessions):
        raw_win, result = hold_and_free_spin(
            starting_bag_symbols,
            rules,
            storage,
        )
        total_raw_win += raw_win
        total_respins += result.total_spins
        maximum_raw_win = max(maximum_raw_win, raw_win)
        for jackpot_index in range(len(jackpot_awards)):
            if result.awarded_jackpots[jackpot_index]:
                jackpot_awards[jackpot_index] += 1

    return total_raw_win, total_respins, jackpot_awards, maximum_raw_win


def evaluate_routing(num_spins: int, seed: int, config=FULL_GAME_CONFIG) -> dict:
    """Return aggregate paid-spin route metrics for a candidate config."""
    validate_full_game_config(config)
    initial_reels = make_reel_collection(
        config.base_game.reelset_path,
        config.base_game,
    )
    (
        route_counts,
        reelset_counts,
        trigger_counts_by_reelset,
        sc_count_histogram,
        distinct_bag_histogram,
    ) = _evaluate_routing_kernel(initial_reels, config, num_spins, seed)
    target_probabilities, _ = load_feature_targets()

    route_rates = route_counts.astype(np.float64) / num_spins
    route_rows = []
    for route_index, route_name in enumerate(FEATURE_NAMES):
        target = target_probabilities[route_index]
        rate = route_rates[route_index]
        route_rows.append(
            {
                "route": route_name,
                "count": int(route_counts[route_index]),
                "rate": float(rate),
                "target_rate": float(target),
                "delta": float(rate - target),
            }
        )

    return {
        "spins": int(num_spins),
        "seed": int(seed),
        "trigger_count": int(route_counts.sum()),
        "trigger_rate": float(route_counts.sum() / num_spins),
        "target_trigger_rate": float(target_probabilities.sum()),
        "routes": route_rows,
        "reelset_counts": reelset_counts.tolist(),
        "trigger_counts_by_reelset": trigger_counts_by_reelset.tolist(),
        "sc_count_histogram": sc_count_histogram.tolist(),
        "distinct_bag_histogram": distinct_bag_histogram.tolist(),
    }


def evaluate_base(num_spins: int, seed: int, config=FULL_GAME_CONFIG) -> dict:
    """Return base RTP/hit metrics from the production paid-spin kernel."""
    validate_full_game_config(config)
    initial_reels = make_reel_collection(
        config.base_game.reelset_path,
        config.base_game,
    )
    (
        total_win,
        line_win,
        collect_win,
        positive_base_spins,
        positive_line_spins,
        positive_collect_spins,
        collect_only_spins,
        trigger_count,
        collector_active_spins,
        collector_symbols,
        collected_coin_cells,
        coin_drop_spins,
        coin_drop_cells,
    ) = _evaluate_base_kernel(initial_reels, config, num_spins, seed)
    targets = load_game_targets()
    return {
        "spins": int(num_spins),
        "seed": int(seed),
        "base_rtp": float(total_win / num_spins),
        "target_base_rtp": targets["Base subtotal RTP"],
        "line_rtp": float(line_win / num_spins),
        "target_line_rtp": targets["Base paylines RTP"],
        "collect_rtp": float(collect_win / num_spins),
        "target_collect_rtp": targets["Base Collect RTP"],
        "base_hit_rate": float(positive_base_spins / num_spins),
        "line_hit_rate": float(positive_line_spins / num_spins),
        "collect_hit_rate": float(positive_collect_spins / num_spins),
        "collect_only_hit_rate": float(collect_only_spins / num_spins),
        "collector_active_spin_rate": float(
            collector_active_spins / num_spins
        ),
        "average_collectors_on_active_spin": float(
            collector_symbols / max(1, collector_active_spins)
        ),
        "average_coins_collected_per_collector": float(
            collected_coin_cells / max(1, collector_symbols)
        ),
        "coin_drop_spin_rate": float(coin_drop_spins / num_spins),
        "average_coin_cells_per_drop": float(
            coin_drop_cells / max(1, coin_drop_spins)
        ),
        "target_base_hit_rate": targets["Base hit rate"],
        "target_base_hit_rate_max": 0.30,
        "trigger_rate": float(trigger_count / num_spins),
        "target_trigger_rate": targets["H&S trigger rate"],
    }


def evaluate_full_game(
    num_rounds: int,
    seed: int,
    config=FULL_GAME_CONFIG,
) -> dict:
    """Run the integrated production game into aggregate-only tuning sinks."""
    validate_full_game_config(config)
    initial_reels = make_reel_collection(
        config.base_game.reelset_path,
        config.base_game,
    )
    base_storage = CompactBaseStorage()
    hold_and_spin_storage = CompactHoldAndSpinTuningStorage()
    full_storage = CompactFullGameTuningStorage(
        len(FEATURE_NAMES),
        len(config.jackpots.jackpot_types),
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
    paid_spins = int(result[4])
    if paid_spins != int(full_storage.spin_count):
        raise RuntimeError("Integrated tuning spin counts diverged")
    if int(result[5]) != int(full_storage.feature_session_count):
        raise RuntimeError("Integrated tuning feature counts diverged")
    if int(hold_and_spin_storage.session_count) != int(result[5]):
        raise RuntimeError("Integrated tuning H&S counts diverged")

    targets = load_game_targets()
    target_probabilities, _ = load_feature_targets()
    feature_rows = []
    for feature_index, feature_name in enumerate(FEATURE_NAMES):
        feature_rows.append(
            {
                "route": feature_name,
                "count": int(
                    full_storage.feature_trigger_counts[feature_index]
                ),
                "trigger_rate": float(
                    full_storage.feature_trigger_counts[feature_index]
                    / paid_spins
                ),
                "target_trigger_rate": float(
                    target_probabilities[feature_index]
                ),
                "rtp": float(
                    full_storage.feature_win_amounts[feature_index]
                    / paid_spins
                ),
                "average_win": float(
                    full_storage.feature_win_amounts[feature_index]
                    / max(
                        1,
                        full_storage.feature_trigger_counts[feature_index],
                    )
                ),
            }
        )

    jackpot_names = ("Mini", "Minor", "Major", "Grand")
    jackpot_rows = []
    for jackpot_index, jackpot_name in enumerate(jackpot_names):
        jackpot_rows.append(
            {
                "jackpot": jackpot_name,
                "count": int(
                    full_storage.jackpot_award_counts[jackpot_index]
                ),
                "award_rate": float(
                    full_storage.jackpot_award_counts[jackpot_index]
                    / paid_spins
                ),
                "rtp": float(
                    full_storage.jackpot_award_amounts[jackpot_index]
                    / paid_spins
                ),
                "target_rtp": targets[f"{jackpot_name} RTP"],
                "average_award": float(
                    full_storage.jackpot_award_amounts[jackpot_index]
                    / max(
                        1,
                        full_storage.jackpot_award_counts[jackpot_index],
                    )
                ),
            }
        )

    return {
        "rounds": int(num_rounds),
        "paid_spins": paid_spins,
        "seed": int(seed),
        "total_rtp": float(result[0] / paid_spins),
        "target_total_rtp": targets["Total RTP"],
        "base_rtp": float(result[1] / paid_spins),
        "target_base_rtp": targets["Base subtotal RTP"],
        "base_line_rtp": float(
            full_storage.base_line_win_amount / paid_spins
        ),
        "base_collect_rtp": float(
            full_storage.base_collect_win_amount / paid_spins
        ),
        "base_hit_rate": float(
            full_storage.base_positive_spin_count / paid_spins
        ),
        "target_base_hit_rate": targets["Base hit rate"],
        "target_base_hit_rate_max": 0.30,
        "feature_rtp": float(result[2] / paid_spins),
        "target_feature_rtp": targets["Non-jackpot feature subtotal RTP"],
        "jackpot_rtp": float(result[3] / paid_spins),
        "target_jackpot_rtp": targets["Jackpot subtotal RTP"],
        "trigger_rate": float(result[5] / paid_spins),
        "target_trigger_rate": targets["H&S trigger rate"],
        "features": feature_rows,
        "jackpots": jackpot_rows,
        "feature_respins": int(full_storage.feature_spin_counts.sum()),
        "hold_and_spin_steps": int(hold_and_spin_storage.step_count),
    }


def _summarize_feature_history(storage, rules) -> dict:
    """Derive mechanic diagnostics from production H&S history arrays."""
    session_count = int(storage.session_count)
    bag_resolution_counts = np.zeros(
        len(rules.bag_symbols), dtype=np.int64
    )
    final_coin_total = 0
    final_visible_total = 0
    final_backing_visible_total = 0
    final_coin_value_total = 0.0
    unlocked_row_total = 0
    all_rows_unlocked = 0
    splitter_generated_coins = 0
    splitter_resolution_count = 0
    multiplier_affected_coins = 0
    multiplier_resolution_count = 0
    booster_affected_coins = 0
    booster_resolution_count = 0
    collector_event_count = 0

    for session_index in range(session_count):
        respin_start = int(
            storage.session_respin_offsets[session_index]
        )
        respin_end = int(
            storage.session_respin_offsets[session_index + 1]
        )
        step_start = int(storage.respin_step_offsets[respin_start])
        step_end = int(storage.respin_step_offsets[respin_end])
        final_step = step_end - 1
        final_locked_row = int(
            storage.step_locked_row_indices[final_step]
        )
        final_mask = storage.coin_masks[final_step]
        final_board = storage.boards[final_step]
        active_slice = slice(final_locked_row + 1, rules.num_rows)
        active_mask = final_mask[active_slice]
        active_board = final_board[active_slice]
        final_coin_total += int(np.count_nonzero(active_mask))
        final_visible_total += int(np.count_nonzero(active_board))
        final_backing_visible_total += int(np.count_nonzero(final_board))
        final_coin_value_total += float(active_board[active_mask].sum())
        unlocked_rows = rules.num_rows - (final_locked_row + 1)
        unlocked_row_total += unlocked_rows - rules.starting_rows
        all_rows_unlocked += int(final_locked_row < 0)

        for step_index in range(step_start, step_end):
            bag_index = int(storage.feature_types[step_index])
            if bag_index < 0:
                continue
            bag_resolution_counts[bag_index] += 1
            previous_step = max(step_start, step_index - 1)
            before_mask = storage.coin_masks[previous_step]
            after_mask = storage.coin_masks[step_index]
            locked_row = int(storage.step_locked_row_indices[step_index])
            active = slice(locked_row + 1, rules.num_rows)
            if bag_index == 0:
                splitter_resolution_count += 1
                added = (
                    int(np.count_nonzero(after_mask[active]))
                    - int(np.count_nonzero(before_mask[active]))
                )
                # The Splitter itself converts into one Coin. Everything
                # beyond that cell was generated by the split action.
                splitter_generated_coins += max(0, added - 1)
            elif bag_index == 2:
                booster_resolution_count += 1
                booster_affected_coins += int(
                    np.count_nonzero(before_mask[active])
                )
            elif bag_index == 3:
                multiplier_resolution_count += 1
                multiplier_affected_coins += int(
                    np.count_nonzero(before_mask[active])
                )
            elif bag_index == 4:
                before_meter = (
                    int(storage.step_collector_meters[previous_step])
                )
                after_meter = int(
                    storage.step_collector_meters[step_index]
                )
                collector_event_count += int(after_meter > before_meter)

    final_average_coin_value = (
        final_coin_value_total / max(1, final_coin_total)
    )
    return {
        "average_final_coins": final_coin_total / max(1, session_count),
        "average_final_visible_symbols": (
            final_visible_total / max(1, session_count)
        ),
        "average_final_backing_visible_symbols": (
            final_backing_visible_total / max(1, session_count)
        ),
        "average_final_coin_value": final_average_coin_value,
        "average_rows_unlocked": (
            unlocked_row_total / max(1, session_count)
        ),
        "all_rows_unlocked_rate": (
            all_rows_unlocked / max(1, session_count)
        ),
        "average_bag_resolutions": (
            bag_resolution_counts / max(1, session_count)
        ).tolist(),
        "average_splitter_generated_coins": (
            splitter_generated_coins / max(1, session_count)
        ),
        "average_coins_per_splitter_resolution": (
            splitter_generated_coins / max(1, splitter_resolution_count)
        ),
        "average_multiplier_resolutions": (
            multiplier_resolution_count / max(1, session_count)
        ),
        "average_coins_per_multiplier_resolution": (
            multiplier_affected_coins
            / max(1, multiplier_resolution_count)
        ),
        "average_booster_resolutions": (
            booster_resolution_count / max(1, session_count)
        ),
        "average_coins_per_booster_resolution": (
            booster_affected_coins / max(1, booster_resolution_count)
        ),
        "average_collector_events": (
            collector_event_count / max(1, session_count)
        ),
    }


def evaluate_features(
    num_sessions: int,
    seed: int,
    config=FULL_GAME_CONFIG,
) -> dict:
    """Return conditional means and recommended multipliers for all routes."""
    validate_full_game_config(config)
    validate_hold_and_spin_config(config.hold_and_spin)
    _, target_means = load_feature_targets()
    current_multipliers = (
        config.feature_rtp.hold_and_spin_payout_multipliers
    )
    route_rows = []

    for route_index, route_name in enumerate(FEATURE_NAMES):
        starting_bags = starting_bags_for_route(
            route_index,
            config.hold_and_spin.bag_symbols,
        )
        storage = _new_storage(num_sessions, config.hold_and_spin)
        (
            total_raw_win,
            total_respins,
            jackpot_awards,
            maximum_raw_win,
        ) = _evaluate_feature_kernel(
            starting_bags,
            config.hold_and_spin,
            num_sessions,
            storage,
            seed + route_index * 104_729,
        )
        raw_mean = total_raw_win / num_sessions
        session_wins = np.asarray(
            storage.session_wins[: storage.session_count]
        )
        raw_standard_deviation = float(
            np.std(session_wins, ddof=1)
        ) if num_sessions > 1 else 0.0
        mechanic_metrics = _summarize_feature_history(
            storage,
            config.hold_and_spin,
        )
        target_mean = target_means[route_index]
        recommended_multiplier = (
            target_mean / raw_mean if raw_mean > 0.0 else 0.0
        )
        route_rows.append(
            {
                "route": route_name,
                "sessions": int(num_sessions),
                "raw_mean": float(raw_mean),
                "raw_standard_deviation": raw_standard_deviation,
                "raw_standard_error": float(
                    raw_standard_deviation / np.sqrt(num_sessions)
                ),
                "current_multiplier": float(current_multipliers[route_index]),
                "current_scaled_mean": float(
                    raw_mean * current_multipliers[route_index]
                ),
                "target_mean": float(target_mean),
                "recommended_multiplier": float(recommended_multiplier),
                "average_respins": float(total_respins / num_sessions),
                "maximum_raw_win": float(maximum_raw_win),
                "jackpot_award_counts": jackpot_awards.tolist(),
                **mechanic_metrics,
            }
        )
        del storage
        gc.collect()

    return {
        "sessions_per_route": int(num_sessions),
        "seed": int(seed),
        "routes": route_rows,
    }


def _format_base(result: dict) -> str:
    rows = (
        ("Base RTP", "base_rtp", "target_base_rtp"),
        ("Payline RTP", "line_rtp", "target_line_rtp"),
        ("Collect RTP", "collect_rtp", "target_collect_rtp"),
        ("Base hit rate", "base_hit_rate", "target_base_hit_rate"),
        ("H&S trigger", "trigger_rate", "target_trigger_rate"),
    )
    lines = [
        "BASE GAME",
        f"Paid spins: {result['spins']:,}",
        "",
        f"{'Metric':<18} {'Observed':>12} {'Target':>12} {'Delta':>12}",
    ]
    for label, observed_key, target_key in rows:
        observed = result[observed_key]
        target = result[target_key]
        if label == "Base hit rate":
            target_max = result["target_base_hit_rate_max"]
            target_text = f"{target:.0%}-{target_max:.0%}"
            status = (
                "in range"
                if target <= observed <= target_max
                else "out of range"
            )
            lines.append(
                f"{label:<18} {observed:>11.6%} "
                f"{target_text:>12} {status:>12}"
            )
            continue
        lines.append(
            f"{label:<18} {observed:>11.6%} {target:>11.6%} "
            f"{observed - target:>+11.6%}"
        )
    return "\n".join(lines)


def _format_routing(result: dict) -> str:
    lines = [
        "ROUTING",
        f"Spins: {result['spins']:,}",
        (
            f"H&S trigger: {result['trigger_rate']:.6%} "
            f"(target {result['target_trigger_rate']:.6%})"
        ),
        "",
        f"{'Route':<14} {'Observed':>12} {'Target':>12} {'Delta':>12}",
    ]
    for row in result["routes"]:
        lines.append(
            f"{row['route']:<14} {row['rate']:>11.6%} "
            f"{row['target_rate']:>11.6%} {row['delta']:>+11.6%}"
        )
    return "\n".join(lines)


def _format_features(result: dict) -> str:
    lines = [
        "CONDITIONAL HOLD-AND-SPIN",
        f"Sessions per route: {result['sessions_per_route']:,}",
        "",
        (
            f"{'Route':<14} {'Raw mean':>10} {'Current':>10} "
            f"{'Target':>10} {'New mult.':>10}"
        ),
    ]
    for row in result["routes"]:
        lines.append(
            f"{row['route']:<14} {row['raw_mean']:>10.4f} "
            f"{row['current_scaled_mean']:>10.4f} "
            f"{row['target_mean']:>10.4f} "
            f"{row['recommended_multiplier']:>10.6f}"
        )
    return "\n".join(lines)


def _format_full_game(result: dict) -> str:
    lines = [
        "INTEGRATED FULL GAME",
        (
            f"Rounds: {result['rounds']:,} | "
            f"Paid spins: {result['paid_spins']:,}"
        ),
        "",
        f"{'Component':<18} {'Observed':>12} {'Target':>12} {'Delta':>12}",
    ]
    components = (
        ("Total RTP", "total_rtp", "target_total_rtp"),
        ("Base RTP", "base_rtp", "target_base_rtp"),
        ("Feature RTP", "feature_rtp", "target_feature_rtp"),
        ("Jackpot RTP", "jackpot_rtp", "target_jackpot_rtp"),
        ("Base hit rate", "base_hit_rate", "target_base_hit_rate"),
        ("H&S trigger", "trigger_rate", "target_trigger_rate"),
    )
    for label, observed_key, target_key in components:
        observed = result[observed_key]
        target = result[target_key]
        if label == "Base hit rate":
            target_max = result["target_base_hit_rate_max"]
            target_text = f"{target:.0%}-{target_max:.0%}"
            status = (
                "in range"
                if target <= observed <= target_max
                else "out of range"
            )
            lines.append(
                f"{label:<18} {observed:>11.6%} "
                f"{target_text:>12} {status:>12}"
            )
            continue
        lines.append(
            f"{label:<18} {observed:>11.6%} {target:>11.6%} "
            f"{observed - target:>+11.6%}"
        )
    lines.extend(
        [
            "",
            f"{'Jackpot':<18} {'Awards':>12} {'RTP':>12} {'Target':>12}",
        ]
    )
    for row in result["jackpots"]:
        lines.append(
            f"{row['jackpot']:<18} {row['count']:>12,} "
            f"{row['rtp']:>11.6%} {row['target_rtp']:>11.6%}"
        )
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Gryphon candidates through production Numba kernels."
    )
    parser.add_argument(
        "--full-rounds",
        type=_positive_count,
        default=1_000_000,
        help="rounds for aggregate integrated evaluation, default: 1m",
    )
    parser.add_argument(
        "--base-spins",
        type=_positive_count,
        default=1_000_000,
        help="paid spins for base RTP/hit evaluation, default: 1m",
    )
    parser.add_argument(
        "--routing-spins",
        type=_positive_count,
        default=1_000_000,
        help="paid spins for route-frequency evaluation, default: 1m",
    )
    parser.add_argument(
        "--feature-sessions",
        type=_positive_count,
        default=20_000,
        help="sessions per H&S route, default: 20k",
    )
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument(
        "--full-only",
        action="store_true",
        help="run only the aggregate integrated full-game evaluation",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="run only the production base-game evaluation",
    )
    parser.add_argument(
        "--routing-only",
        action="store_true",
        help="skip conditional H&S evaluation",
    )
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="skip base routing evaluation",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="optional JSON output path (must be inside Tuning)",
    )
    return parser


def main(argv=None) -> dict:
    args = build_argument_parser().parse_args(argv)
    selected_only_modes = sum(
        (
            args.full_only,
            args.base_only,
            args.routing_only,
            args.features_only,
        )
    )
    if selected_only_modes > 1:
        raise ValueError(
            "the --*-only evaluation modes are mutually exclusive"
        )

    result = {}
    if args.full_only:
        result["full_game"] = evaluate_full_game(
            args.full_rounds,
            args.seed,
        )
        print(_format_full_game(result["full_game"]))
    if args.base_only or selected_only_modes == 0:
        result["base"] = evaluate_base(
            args.base_spins,
            args.seed,
        )
        print(_format_base(result["base"]))
    if args.routing_only or selected_only_modes == 0:
        if result:
            print()
        result["routing"] = evaluate_routing(
            args.routing_spins,
            args.seed,
        )
        print(_format_routing(result["routing"]))
    if args.features_only or selected_only_modes == 0:
        if result:
            print()
        result["features"] = evaluate_features(
            args.feature_sessions,
            args.seed,
        )
        print(_format_features(result["features"]))

    if args.json_output is not None:
        output_path = args.json_output.resolve()
        tuning_root = Path(__file__).resolve().parent
        if tuning_root not in output_path.parents:
            raise ValueError("--json-output must be inside Numba_Engine/Tuning")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


if __name__ == "__main__":
    main()
