from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Numba_Engine.core.config import BASE_GAME_CONFIG, BASE_PAY_TABLE, PAY_LINES
from Numba_Engine.core.kernels import (
    line_win_eval,
    make_board,
    probChoice,
    select_reelset_index,
)
from Numba_Engine.core.reels import REEL_DICT, make_reel_collection


REEL_DIR = ROOT / "Reels" / "Base_Game"
SUMMARY_PATH = ROOT / "Numba_Engine" / "Tuning" / "latest_base_tuning.json"

SYMBOLS = [
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "L6",
    "WD",
    "COLLECT",
    "SC",
]


def _can_place_sc(strip: list[str], idx: int) -> bool:
    n = len(strip)
    for offset in (-2, -1, 1, 2):
        if strip[(idx + offset) % n] == "COLLECT":
            return False
    for offset in (-1, 1):
        if strip[(idx + offset) % n] == "SC":
            return False
    return True


def _can_place_collect(strip: list[str], idx: int) -> bool:
    n = len(strip)
    for offset in (-2, -1, 0, 1, 2):
        if strip[(idx + offset) % n] == "SC":
            return False
    return True


def _place_stack(strip: list[str], symbol: str, length: int, start: int) -> None:
    n = len(strip)
    for offset in range(length):
        idx = (start + offset) % n
        if strip[idx] != "":
            raise ValueError(f"Cannot place {symbol} stack over occupied cell {idx}")
        strip[idx] = symbol


def _place_scatter_pair(strip: list[str], start: int) -> None:
    first = start % len(strip)
    second = (start + 2) % len(strip)
    if strip[first] or strip[second]:
        raise ValueError("Scatter pair target occupied")
    if not _can_place_sc(strip, first) or not _can_place_sc(strip, second):
        raise ValueError("Scatter pair violates spacing")
    strip[first] = "SC"
    strip[second] = "SC"


def _fill_required_symbols(strip: list[str], reel_index: int) -> None:
    required = [
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "L1",
        "L2",
        "L3",
        "L4",
        "L5",
        "L6",
    ]
    if reel_index != 0:
        required.append("WD")
    for symbol in required:
        if symbol in strip:
            continue
        idx = strip.index("")
        strip[idx] = symbol


def build_reel(
    length: int,
    reel_index: int,
    collect_starts: list[int],
    collect_lengths: list[int],
    scatter_starts: list[int],
    rng: np.random.Generator,
    premium_bias: float,
) -> list[str]:
    strip = [""] * length

    for start, stack_len in zip(collect_starts, collect_lengths):
        for offset in range(stack_len):
            if not _can_place_collect(strip, start + offset):
                raise ValueError("Collect stack violates spacing")
        _place_stack(strip, "COLLECT", stack_len, start)

    for start in scatter_starts:
        _place_scatter_pair(strip, start)

    _fill_required_symbols(strip, reel_index)

    filler_symbols = np.array(
        ["H1", "H2", "H3", "H4", "H5", "L1", "L2", "L3", "L4", "L5", "L6", "WD"],
        dtype=object,
    )
    weights = np.array(
        [
            0.018,
            0.024,
            0.030,
            0.045,
            0.045,
            0.120,
            0.120,
            0.120,
            0.120,
            0.120,
            0.120,
            0.038,
        ],
        dtype=np.float64,
    )
    weights[:5] *= premium_bias
    if reel_index == 0:
        weights[-1] = 0.0
    weights = weights / weights.sum()

    for idx in range(length):
        if strip[idx]:
            continue
        strip[idx] = str(rng.choice(filler_symbols, p=weights))

    return strip


def write_reelsets(length: int = 1000, premium_bias: float = 16.8) -> None:
    rng = np.random.default_rng(20260928)
    reel_specs = [
        ([110, 410, 710], [2, 1, 1], [40, 540]),
        ([130, 430, 730], [2, 1, 1], [70, 570]),
        ([150, 450, 750], [3, 2, 1], [100, 600]),
        ([170, 470, 770], [3, 2, 1], [130, 630]),
        ([190, 490, 790], [2, 1, 1], [160, 660]),
    ]

    columns = {}
    for reel_index, (collect_starts, collect_lengths, scatter_starts) in enumerate(
        reel_specs
    ):
        columns[f"Reel{reel_index + 1}"] = build_reel(
            length,
            reel_index,
            collect_starts,
            collect_lengths,
            scatter_starts,
            rng,
            premium_bias,
        )

    REEL_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(columns)
    frame.to_csv(REEL_DIR / "ReelSet_1.csv", index=False)
    frame.to_csv(REEL_DIR / "ReelSet_2.csv", index=False)


@njit
def _run_metric_spins(initial_reels, rules, num_spins):
    line_total = 0.0
    collect_total = 0.0
    trigger_total = 0
    coin_spin_total = 0
    coin_count_total = 0
    collector_spin_total = 0
    collector_total = 0

    for _ in range(num_spins):
        reelset_index = select_reelset_index(initial_reels.weights)
        pay_window = make_board(
            initial_reels.lengths[reelset_index],
            initial_reels.reelsets[reelset_index],
            rules,
        )
        collector_count = 0
        for row in range(pay_window.shape[0]):
            for col in range(pay_window.shape[1]):
                if pay_window[row, col] == rules.collect_symbol:
                    collector_count += 1

        (
            line_win,
            _line_winning_symbols,
            _line_match_counts,
            _line_win_amounts,
            _winning_window,
            _hit_counts,
            _symbol_win_amounts,
        ) = line_win_eval(
            pay_window,
            BASE_PAY_TABLE,
            PAY_LINES,
            rules.wild_symbol,
        )

        coin_value_window = np.zeros(pay_window.shape, dtype=np.float64)
        available_positions = np.empty(pay_window.size, dtype=np.int32)
        num_available = 0
        for row in range(pay_window.shape[0]):
            for col in range(pay_window.shape[1]):
                symbol = pay_window[row, col]
                if (
                    symbol == rules.wild_symbol
                    or symbol == rules.collect_symbol
                    or symbol == rules.coin_symbol
                    or symbol == rules.sc_symbol
                ):
                    continue
                available_positions[num_available] = row * pay_window.shape[1] + col
                num_available += 1

        probability_index = 2
        if collector_count == 0:
            probability_index = 0
        elif collector_count == 1:
            probability_index = 1

        dropped = 0
        if (
            num_available > 0
            and np.random.uniform(0.0, 1.0)
            < rules.coin_drop_probabilities[probability_index]
        ):
            drop_count = int(
                probChoice(
                    rules.coin_drop_count_probabilities,
                    rules.coin_drop_counts,
                )
            )
            drop_count = min(drop_count, num_available)
            for _drop_index in range(drop_count):
                available_index = np.random.randint(0, num_available)
                position = available_positions[available_index]
                row = position // pay_window.shape[1]
                col = position % pay_window.shape[1]
                coin_value_window[row, col] = probChoice(
                    rules.coin_credit_value_probabilities,
                    rules.coin_credit_values,
                )
                dropped += 1
                num_available -= 1
                available_positions[available_index] = available_positions[
                    num_available
                ]

        visible_coin_total = np.sum(coin_value_window)
        collect_win = visible_coin_total * collector_count

        line_total += line_win
        collect_total += collect_win
        if collector_count > 0:
            collector_spin_total += 1
            collector_total += collector_count
        if dropped > 0:
            coin_spin_total += 1
            coin_count_total += dropped
        has_sc = False
        for row in range(pay_window.shape[0]):
            for col in range(pay_window.shape[1]):
                if pay_window[row, col] == rules.sc_symbol:
                    has_sc = True
                    break
            if has_sc:
                break
        if has_sc:
            trigger_total += 1

    return (
        line_total,
        collect_total,
        trigger_total,
        coin_spin_total,
        coin_count_total,
        collector_spin_total,
        collector_total,
    )


def evaluate(num_spins: int) -> dict[str, float]:
    reels = make_reel_collection(BASE_GAME_CONFIG.reelset_path, BASE_GAME_CONFIG)
    (
        line_total,
        collect_total,
        trigger_total,
        coin_spin_total,
        coin_count_total,
        collector_spin_total,
        collector_total,
    ) = _run_metric_spins(reels, BASE_GAME_CONFIG, num_spins)
    return {
        "spins": num_spins,
        "line_rtp": line_total / num_spins,
        "collect_rtp": collect_total / num_spins,
        "collector_frequency": collector_spin_total / num_spins,
        "average_collectors_on_collector_spin": (
            collector_total / collector_spin_total if collector_spin_total else 0.0
        ),
        "coin_window_frequency": coin_spin_total / num_spins,
        "average_coin_count_when_dropped": (
            coin_count_total / coin_spin_total if coin_spin_total else 0.0
        ),
        "average_coin_count_per_spin": coin_count_total / num_spins,
        "sc_window_frequency": trigger_total / num_spins,
    }


def main() -> None:
    write_reelsets()
    metrics = evaluate(500_000)
    SUMMARY_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
