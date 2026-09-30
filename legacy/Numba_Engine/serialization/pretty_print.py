"""Human-readable formatting for per-spin JSON payloads."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING, TextIO

from ..core.reels import REEL_DICT

if TYPE_CHECKING:
    from ..output.statistics import (
        BaseGameStatistics,
        FeatureStatistics,
        HoldAndSpinStatistics,
    )


_SYMBOL_NAMES = {value: key for key, value in REEL_DICT.items()}


def _number(value):
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _format_matrix(matrix, empty_marker, map_symbols=False):
    display_rows = [
        [
            empty_marker
            if value == -1
            else _SYMBOL_NAMES.get(value, _number(value))
            if map_symbols
            else _number(value)
            for value in row
        ]
        for row in matrix
    ]
    if not display_rows:
        return "[]"

    width = max(len(value) for row in display_rows for value in row)
    rows = []
    for row_index, row in enumerate(display_rows):
        opening = "[[" if row_index == 0 else " ["
        closing = "]]" if row_index == len(display_rows) - 1 else "]"
        cells = " ".join(value.rjust(width) for value in row)
        rows.append(f"{opening}{cells}{closing}")
    return "\n".join(rows)


def _removed_window(board, winning_mask):
    if len(board) != len(winning_mask):
        raise ValueError("board and winning_mask row counts do not match")

    removed = []
    for board_row, mask_row in zip(board, winning_mask):
        if len(board_row) != len(mask_row):
            raise ValueError("board and winning_mask column counts do not match")
        removed.append(
            [
                -1 if is_winner else symbol
                for symbol, is_winner in zip(board_row, mask_row)
            ]
        )
    return removed


def _append_matrix(
    lines,
    label,
    matrix,
    empty_marker,
    map_symbols=False,
):
    lines.append(f"    {label}:")
    lines.extend(
        f"      {line}"
        for line in _format_matrix(
            matrix,
            empty_marker,
            map_symbols=map_symbols,
        ).splitlines()
    )


def _append_spin(lines, spin, display_index, total_spins, empty_marker):
    multiplier_before = spin["global_multiplier_before"]
    multiplier_after = spin["global_multiplier_after"]
    triggered = "yes" if spin["free_game_trigger"] else "no"
    lines.append(
        f"  Spin {display_index}/{total_spins} | "
        f"win: {_number(spin['win'])} | "
        f"global multiplier: {multiplier_before} -> {multiplier_after} | "
        f"retrigger: {triggered}"
    )

    for cascade_index, step in enumerate(spin["steps"], start=1):
        lines.append(
            f"    Cascade {cascade_index} | "
            f"win: {_number(step['win'])} | "
            f"payout multiplier: {step['payout_multiplier']}"
        )
        _append_matrix(
            lines,
            "Pay window",
            step["board"],
            empty_marker,
            map_symbols=True,
        )
        _append_matrix(
            lines,
            "Multiplier window",
            step["multiplier_board"],
            empty_marker,
        )
        cascaded_window = _removed_window(step["board"], step["winning_mask"])
        _append_matrix(
            lines,
            "Cascaded window",
            cascaded_window,
            empty_marker,
            map_symbols=True,
        )

        symbol_wins = [
            f"{_SYMBOL_NAMES.get(symbol, symbol)}={_number(win)}"
            for symbol, win in enumerate(step["symbol_wins"])
            if win
        ]
        if symbol_wins:
            lines.append(f"      Symbol wins: {', '.join(symbol_wins)}")


def _format_hold_board(board, coin_mask, locked_row_index, empty_marker):
    display_rows = []
    for row_index, (board_row, mask_row) in enumerate(zip(board, coin_mask)):
        display_row = []
        for value, is_coin in zip(board_row, mask_row):
            if is_coin:
                display_row.append(f"C:{value}")
            elif value == 0:
                display_row.append(
                    "##" if row_index <= locked_row_index else empty_marker
                )
            else:
                display_row.append(_SYMBOL_NAMES.get(value, _number(value)))
        display_rows.append(display_row)

    width = max(len(value) for row in display_rows for value in row)
    rows = []
    for row_index, row in enumerate(display_rows):
        opening = "[[" if row_index == 0 else " ["
        closing = "]]" if row_index == len(display_rows) - 1 else "]"
        rows.append(
            f"{opening}{' '.join(value.rjust(width) for value in row)}{closing}"
        )
    return "\n".join(rows)


def _format_base_game_payload(payload, empty_marker):
    lines = [
        "BASE GAME ROUND",
        f"Rounds: {payload['round_count']} | Paid spins: {payload['spin_count']}",
    ]
    for round_position, round_result in enumerate(payload["rounds"], start=1):
        lines.append(
            f"Round {round_position} | win: {_number(round_result['win'])} | "
            f"free-game triggers: {round_result['free_game_triggers']}"
        )
        for spin_position, spin in enumerate(round_result["spins"], start=1):
            lines.append(
                f"  Paid spin {spin_position}/{len(round_result['spins'])} | "
                f"win: {_number(spin['win'])} | "
                f"line: {_number(spin['line_win'])} | "
                f"collect: {_number(spin['collect_win'])} | "
                f"collectors: {spin['collector_count']} | "
                f"feature trigger: {'yes' if spin['free_game_trigger'] else 'no'}"
            )
            _append_matrix(
                lines,
                "Pay window",
                spin["board"],
                empty_marker,
                map_symbols=True,
            )
            _append_matrix(
                lines,
                "Coin values",
                spin["coin_value_board"],
                empty_marker,
            )
            for line_win in spin["winning_lines"]:
                symbol = _SYMBOL_NAMES.get(
                    line_win["symbol"],
                    line_win["symbol"],
                )
                lines.append(
                    f"      Line {line_win['line_number']} | {symbol} x"
                    f"{line_win['symbol_count']} | "
                    f"win: {_number(line_win['win'])}"
                )
    total_win = sum(round_result["win"] for round_result in payload["rounds"])
    lines.extend(("", f"TOTAL WIN: {_number(total_win)}"))
    return "\n".join(lines)


def _format_hold_and_spin_payload(payload, empty_marker):
    feature_names = {
        -2: "Initial window",
        -1: "Symbols landed",
        0: "Splitter",
        1: "Grower",
        2: "Booster",
        3: "Multiplier",
        4: "Collector",
        5: "Expansion",
    }
    lines = [
        "HOLD AND SPIN",
        f"Sessions: {payload['session_count']} | "
        f"Respins: {payload['respin_count']} | Steps: {payload['step_count']}",
    ]
    for session_position, session in enumerate(payload["sessions"], start=1):
        starting_symbols = ", ".join(
            _SYMBOL_NAMES.get(symbol, str(symbol))
            for symbol in session["starting_symbols"]
        )
        lines.append(
            f"Session {session_position} | win: {_number(session['win'])} | "
            f"coins: {_number(session['coin_win'])} | "
            f"collector: {_number(session['collector_win'])} | "
            f"starting features: {starting_symbols}"
        )
        for respin_position, respin in enumerate(session["respins"], start=1):
            lines.append(
                f"  Respin {respin_position}/{session['total_respins']} | "
                f"remaining: {respin['remaining_before']} -> "
                f"{respin['remaining_after']} | "
                f"reset: {'yes' if respin['reset'] else 'no'}"
            )
            for step_position, step in enumerate(respin["steps"], start=1):
                feature_type = step["feature_type"]
                event_name = feature_names.get(
                    feature_type,
                    f"Feature {feature_type}",
                )
                position = step["feature_position"]
                position_text = "" if position < 0 else f" at {position}"
                lines.append(
                    f"    Step {step_position} | {event_name}{position_text} | "
                    f"locked row: {step['locked_row_index']} | "
                    f"collector meter: {step['collector_meter']}"
                )
                formatted = _format_hold_board(
                    step["board"],
                    step["coin_mask"],
                    step["locked_row_index"],
                    empty_marker,
                )
                lines.append("      Window:")
                lines.extend(f"        {line}" for line in formatted.splitlines())
    total_win = sum(session["win"] for session in payload["sessions"])
    lines.extend(("", f"TOTAL WIN: {_number(total_win)}"))
    return "\n".join(lines)


def format_payload(payload: Mapping, empty_marker="##"):
    """Return a readable spin/cascade report without mutating ``payload``."""
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    if not isinstance(empty_marker, str) or not empty_marker:
        raise ValueError("empty_marker must be a non-empty string")

    storage_type = payload.get("storage_type")
    if storage_type == "base_game":
        return _format_base_game_payload(payload, empty_marker)
    if storage_type == "hold_and_spin":
        return _format_hold_and_spin_payload(payload, empty_marker)

    lines = [
        f"Steps: {payload.get('step_count', 0)} | "
        f"Spins: {payload.get('spin_count', 0)} | "
        f"Sessions: {payload.get('session_count', 0)}"
    ]

    sessions = payload.get("sessions")
    if sessions is not None:
        for session_position, session in enumerate(sessions, start=1):
            if session_position > 1:
                lines.append("")
            spins = session["spins"]
            lines.append(
                f"Session {session_position} | "
                f"win: {_number(session['win'])} | "
                f"initial spins: {session['initial_spins']} | "
                f"total spins: {session['total_spins']}"
            )
            for spin_position, spin in enumerate(spins, start=1):
                _append_spin(
                    lines,
                    spin,
                    spin_position,
                    len(spins),
                    empty_marker,
                )
    else:
        spins = payload.get("spins")
        if spins is None:
            raise ValueError("payload must contain 'spins' or 'sessions'")
        for spin_position, spin in enumerate(spins, start=1):
            _append_spin(
                lines,
                spin,
                spin_position,
                len(spins),
                empty_marker,
            )

    return "\n".join(lines)


def pretty_print(
    payload: Mapping,
    empty_marker="##",
    file: TextIO | None = None,
):
    """Print a readable spin/cascade report made from a JSON payload."""
    print(
        format_payload(payload, empty_marker=empty_marker),
        file=sys.stdout if file is None else file,
    )


def _stat_number(value):
    if value == float("inf"):
        return "+inf"
    return f"{value:g}"


def _win_range_label(lower, upper):
    if lower <= 0:
        return f">0 to <{_stat_number(upper)}"
    if upper == float("inf"):
        return f"{_stat_number(lower)}+"
    return f"{_stat_number(lower)} to <{_stat_number(upper)}"


def _table(lines, headers, rows):
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    lines.append(
        "  ".join(
            header.ljust(widths[index])
            for index, header in enumerate(headers)
        )
    )
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append(
            "  ".join(
                value.rjust(widths[index])
                for index, value in enumerate(row)
            )
        )


def format_base_game_statistics(statistics: BaseGameStatistics):
    """Return a readable base-game statistics report."""
    stats = statistics
    lines = [
        "BASE GAME STATISTICS",
        f"Source: {stats.source_path}",
        "",
        f"Spins:                     {stats.spin_count:,}",
        f"Evaluation steps:          {stats.step_count:,}",
        f"Bet per spin:              {stats.bet_per_spin:,.4f}",
        f"Total bet:                 {stats.total_bet:,.4f}",
        f"Total win:                 {stats.total_win:,.4f}",
        f"Total line win:            {stats.total_line_win:,.4f}",
        f"Total Collector win:       {stats.total_collect_win:,.4f}",
        f"Average Collector win:     "
        f"{stats.average_collect_win_per_spin:,.4f}",
        f"Maximum Collector win:     {stats.maximum_collect_win:,.4f}",
        f"Average active Collectors: {stats.average_collectors_per_spin:,.4f}",
        f"Collector-active spin rate:{stats.collector_active_spin_rate:>10.4%} "
        f"({stats.collector_active_spin_count:,}/{stats.spin_count:,})",
        f"Maximum spin win:          {stats.maximum_win:,.4f}",
        f"RTP:                       {stats.rtp:.4%}",
        f"Hit rate:                  {stats.hit_rate:.4%} "
        f"({stats.hit_count:,}/{stats.spin_count:,})",
        f"Free-game trigger rate:    {stats.trigger_rate:.4%} "
        f"({stats.trigger_count:,}/{stats.spin_count:,})",
        f"Average win on hit:        {stats.average_win_on_hit:,.4f}",
        f"Return variance:           {stats.return_variance:,.6f}",
        f"Return standard deviation: {stats.return_standard_deviation:,.6f}",
        f"Average cascades:          {stats.average_cascades:,.4f}",
        f"Maximum cascades:          {stats.maximum_cascades:,}",
        "",
        "WIN HISTOGRAM (multiples of bet)",
    ]

    win_rows = [
        (
            "0 (no win)",
            f"{stats.zero_win_count:,}",
            f"{stats.zero_win_count / stats.spin_count:.4%}",
        )
    ]
    for index, count in enumerate(stats.win_histogram_counts):
        win_rows.append(
            (
                _win_range_label(
                    stats.win_histogram_edges[index],
                    stats.win_histogram_edges[index + 1],
                ),
                f"{int(count):,}",
                f"{int(count) / stats.spin_count:.4%}",
            )
        )
    _table(lines, ("Win", "Count", "% Spins"), win_rows)

    lines.extend(("", "CASCADE HISTOGRAM"))
    cascade_rows = [
        (
            f"{cascade_count}",
            f"{int(count):,}",
            f"{int(count) / stats.spin_count:.4%}",
        )
        for cascade_count, count in enumerate(stats.cascade_histogram_counts)
    ]
    _table(lines, ("Cascades", "Count", "% Spins"), cascade_rows)

    return "\n".join(lines)


def pretty_print_base_game(
    statistics: BaseGameStatistics,
    file: TextIO | None = None,
):
    """Print a readable base-game statistics report."""
    print(
        format_base_game_statistics(statistics),
        file=sys.stdout if file is None else file,
    )


def format_free_game_statistics(statistics: FeatureStatistics):
    """Return a readable free-game session statistics report."""
    stats = statistics
    lines = [
        "FREE GAME STATISTICS",
        f"Source: {stats.source_path}",
        "",
        f"Sessions:                         {stats.session_count:,}",
        f"Spins including retriggers:       {stats.spin_count:,}",
        f"Evaluation steps:                 {stats.step_count:,}",
        f"Bet per session:                  {stats.bet_per_session:,.4f}",
        f"Total bet:                        {stats.total_bet:,.4f}",
        f"Total win:                        {stats.total_win:,.4f}",
        f"Maximum session win:              {stats.maximum_win:,.4f}",
        f"RTP:                              {stats.rtp:.4%}",
        f"Winning-session rate:             {stats.hit_rate:.4%} "
        f"({stats.hit_count:,}/{stats.session_count:,})",
        f"Retrigger rate per spin:          {stats.trigger_rate:.4%} "
        f"({stats.trigger_count:,}/{stats.spin_count:,})",
        f"Average retriggers per session:   "
        f"{stats.average_retriggers_per_session:,.4f}",
        f"Average spins per session:        "
        f"{stats.average_spins_per_session:,.4f}",
        f"Average win on winning session:   "
        f"{stats.average_win_on_hit:,.4f}",
        f"Return variance:                  {stats.return_variance:,.6f}",
        f"Return standard deviation:        "
        f"{stats.return_standard_deviation:,.6f}",
        f"Average final global multiplier:  "
        f"{stats.average_global_multiplier:,.4f}",
        f"Average session trigger rate:     "
        f"{stats.average_session_trigger_rate:.4%}",
        f"Average session spin hit rate:    "
        f"{stats.average_session_hit_rate:.4%}",
        f"Average session cascade rate:     "
        f"{stats.average_session_cascade_rate:,.4f}",
        f"Maximum cascades in one session:  "
        f"{stats.maximum_session_cascades:,}",
        "",
        "SESSION WIN HISTOGRAM (multiples of session bet)",
    ]

    win_rows = [
        (
            "0 (no win)",
            f"{stats.zero_win_count:,}",
            f"{stats.zero_win_count / stats.session_count:.4%}",
        )
    ]
    for index, count in enumerate(stats.win_histogram_counts):
        win_rows.append(
            (
                _win_range_label(
                    stats.win_histogram_edges[index],
                    stats.win_histogram_edges[index + 1],
                ),
                f"{int(count):,}",
                f"{int(count) / stats.session_count:.4%}",
            )
        )
    _table(lines, ("Session win", "Count", "% Sessions"), win_rows)

    lines.extend(("", "SESSION TOTAL CASCADE HISTOGRAM"))
    cascade_rows = [
        (
            f"{cascade_count}",
            f"{int(count):,}",
            f"{int(count) / stats.session_count:.4%}",
        )
        for cascade_count, count in enumerate(stats.cascade_histogram_counts)
    ]
    _table(
        lines,
        ("Total cascades", "Count", "% Sessions"),
        cascade_rows,
    )

    return "\n".join(lines)


def pretty_print_free_game(
    statistics: FeatureStatistics,
    file: TextIO | None = None,
):
    """Print a readable free-game session statistics report."""
    print(
        format_free_game_statistics(statistics),
        file=sys.stdout if file is None else file,
    )


def format_hold_and_spin_statistics(statistics: HoldAndSpinStatistics):
    """Return a readable Hold-and-Spin statistics report."""
    stats = statistics
    lines = [
        "HOLD AND SPIN STATISTICS",
        f"Source: {stats.source_path}",
        "",
        f"Sessions:                       {stats.session_count:,}",
        f"Respins:                        {stats.respin_count:,}",
        f"Stored steps:                   {stats.step_count:,}",
        f"Bet per session:                {stats.bet_per_session:,.4f}",
        f"Total bet:                      {stats.total_bet:,.4f}",
        f"Total win:                      {stats.total_win:,.4f}",
        f"Coin win:                       {stats.total_coin_win:,.4f}",
        f"Collector win:                  {stats.total_collector_win:,.4f}",
        f"Maximum session win:            {stats.maximum_win:,.4f}",
        f"RTP:                            {stats.rtp:.4%}",
        f"Winning-session rate:           {stats.hit_rate:.4%} "
        f"({stats.hit_count:,}/{stats.session_count:,})",
        f"Average win on winning session: {stats.average_win_on_hit:,.4f}",
        f"Return variance:                {stats.return_variance:,.6f}",
        f"Return standard deviation:      "
        f"{stats.return_standard_deviation:,.6f}",
        f"Average respins per session:    "
        f"{stats.average_respins_per_session:,.4f}",
        f"Maximum respins:                {stats.maximum_respins:,}",
        f"Average steps per respin:       "
        f"{stats.average_steps_per_respin:,.4f}",
        f"Maximum steps per respin:       {stats.maximum_steps_per_respin:,}",
        f"Respin reset rate:              {stats.reset_rate:.4%} "
        f"({stats.reset_count:,}/{stats.respin_count:,})",
        f"Average starting features:      "
        f"{stats.average_starting_symbols:,.4f}",
        "",
        "SESSION WIN HISTOGRAM (multiples of session bet)",
    ]

    win_rows = [
        (
            "0 (no win)",
            f"{stats.zero_win_count:,}",
            f"{stats.zero_win_count / stats.session_count:.4%}",
        )
    ]
    for index, count in enumerate(stats.win_histogram_counts):
        win_rows.append(
            (
                _win_range_label(
                    stats.win_histogram_edges[index],
                    stats.win_histogram_edges[index + 1],
                ),
                f"{int(count):,}",
                f"{int(count) / stats.session_count:.4%}",
            )
        )
    _table(lines, ("Session win", "Count", "% Sessions"), win_rows)

    lines.extend(("", "RESPIN HISTOGRAM"))
    respin_rows = [
        (
            str(respin_count),
            f"{int(count):,}",
            f"{int(count) / stats.session_count:.4%}",
        )
        for respin_count, count in enumerate(stats.respin_histogram_counts)
        if count
    ]
    _table(lines, ("Respins", "Count", "% Sessions"), respin_rows)

    feature_names = (
        "Splitter",
        "Grower",
        "Booster",
        "Multiplier",
        "Collector",
        "Expansion",
    )
    lines.extend(("", "FEATURE RESOLUTIONS"))
    feature_rows = [
        (name, f"{int(stats.feature_resolution_counts[index]):,}")
        for index, name in enumerate(feature_names)
    ]
    _table(lines, ("Feature", "Resolutions"), feature_rows)
    return "\n".join(lines)


def pretty_print_hold_and_spin(
    statistics: HoldAndSpinStatistics,
    file: TextIO | None = None,
):
    """Print a readable Hold-and-Spin statistics report."""
    print(
        format_hold_and_spin_statistics(statistics),
        file=sys.stdout if file is None else file,
    )
