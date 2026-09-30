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
        FullGameStatistics,
        HoldAndSpinStatistics,
    )


_SYMBOL_NAMES = {value: key for key, value in REEL_DICT.items()}
_JACKPOT_NAMES = {
    0: "Mini",
    1: "Minor",
    2: "Major",
    3: "Grand",
}
_HOLD_FEATURE_NAMES = {
    -2: "Initial window",
    -1: "Symbols landed",
    0: "Splitter",
    1: "Grower",
    2: "Booster",
    3: "Multiplier",
    4: "Collector",
    5: "Expansion",
}
_ROUTED_FEATURE_NAMES = (
    "Splitter",
    "Grow",
    "Boost",
    "Multiplier",
    "Collect",
    "Expansion",
    "Mega Combo",
)


def _number(value):
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _format_matrix(
    matrix,
    empty_marker,
    map_symbols=False,
    value_names=None,
):
    display_rows = [
        [
            empty_marker
            if value == -1
            else value_names.get(value, _number(value))
            if value_names is not None
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
    value_names=None,
):
    lines.append(f"    {label}:")
    lines.extend(
        f"      {line}"
        for line in _format_matrix(
            matrix,
            empty_marker,
            map_symbols=map_symbols,
            value_names=value_names,
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
            _append_matrix(
                lines,
                "Jackpot overlays",
                spin["jackpot_overlay_board"],
                empty_marker,
                value_names=_JACKPOT_NAMES,
            )
            before = spin["jackpot_values_before"]
            after = spin["jackpot_values_after"]
            increments = spin["jackpot_increment_counts"]
            jackpot_transitions = ", ".join(
                f"{_JACKPOT_NAMES.get(index, index)} "
                f"{_number(start)}->{_number(end)}"
                for index, (start, end) in enumerate(zip(before, after))
            )
            lines.append(
                f"      Jackpot values: {jackpot_transitions}"
            )
            applied_overlays = [
                f"{_JACKPOT_NAMES.get(index, index)}={count}"
                for index, count in enumerate(increments)
                if count
            ]
            lines.append(
                "      Jackpot overlay counts: "
                + (", ".join(applied_overlays) if applied_overlays else "none")
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
        final_meters = ", ".join(
            f"{_JACKPOT_NAMES.get(index, index)}={value}/3"
            for index, value in enumerate(session["jackpot_meters"])
        )
        awarded_names = [
            _JACKPOT_NAMES.get(index, str(index))
            for index, awarded in enumerate(session["jackpot_awards"])
            if awarded
        ]
        lines.append(f"  Final jackpot meters: {final_meters}")
        lines.append(
            "  Jackpot types awarded: "
            + (", ".join(awarded_names) if awarded_names else "none")
        )
        for respin_position, respin in enumerate(session["respins"], start=1):
            lines.append(
                f"  Respin {respin_position}/{session['total_respins']} | "
                f"remaining: {respin['remaining_before']} -> "
                f"{respin['remaining_after']} | "
                f"reset: {'yes' if respin['reset'] else 'no'}"
            )
            _append_matrix(
                lines,
                "Jackpot token overlays",
                respin["jackpot_overlay_board"],
                empty_marker,
                value_names=_JACKPOT_NAMES,
            )
            meter_transitions = ", ".join(
                f"{_JACKPOT_NAMES.get(index, index)} "
                f"{before}->{after}"
                for index, (before, after) in enumerate(
                    zip(
                        respin["jackpot_meters_before"],
                        respin["jackpot_meters_after"],
                    )
                )
            )
            respin_awards = [
                _JACKPOT_NAMES.get(index, str(index))
                for index, awarded in enumerate(respin["jackpot_awards"])
                if awarded
            ]
            lines.append(f"    Jackpot meters: {meter_transitions}")
            lines.append(
                "    Jackpot awards: "
                + (", ".join(respin_awards) if respin_awards else "none")
            )
            for step_position, step in enumerate(respin["steps"], start=1):
                feature_type = step["feature_type"]
                event_name = _HOLD_FEATURE_NAMES.get(
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


def _format_full_game_payload(payload, empty_marker):
    lines = [
        "FULL GAME",
        f"Rounds: {payload['round_count']} | Paid spins: "
        f"{payload['spin_count']} | Features: "
        f"{payload['feature_session_count']}",
    ]
    for round_position, round_result in enumerate(payload["rounds"], start=1):
        lines.append(
            f"Round {round_position} | total: "
            f"{_number(round_result['total_win'])} | base: "
            f"{_number(round_result['base_win'])} | feature: "
            f"{_number(round_result['feature_win'])} | jackpots: "
            f"{_number(round_result['jackpot_win'])}"
        )
        for spin_position, spin in enumerate(round_result["spins"], start=1):
            base = spin["base_spin"]
            lines.append(
                f"  Paid spin {spin_position}/{len(round_result['spins'])} | "
                f"total: {_number(spin['total_win'])} | "
                f"base: {_number(spin['base_win'])} | "
                f"feature: {_number(spin['feature_win'])} | "
                f"jackpots: {_number(spin['jackpot_win'])}"
            )
            _append_matrix(
                lines,
                "Pay window",
                base["board"],
                empty_marker,
                map_symbols=True,
            )
            _append_matrix(
                lines,
                "Base jackpot overlays",
                base["jackpot_overlay_board"],
                empty_marker,
                value_names=_JACKPOT_NAMES,
            )
            increments = [
                f"{_JACKPOT_NAMES.get(index, index)}={count}"
                for index, count in enumerate(base["jackpot_increment_counts"])
                if count
            ]
            lines.append(
                "      Base jackpot increments: "
                + (", ".join(increments) if increments else "none")
            )
            awards = [
                f"{_JACKPOT_NAMES.get(index, index)}="
                f"{_number(spin['jackpot_award_amounts'][index])}"
                for index, awarded in enumerate(spin["jackpot_awards"])
                if awarded
            ]
            lines.append(
                "      Jackpot awards: "
                + (", ".join(awards) if awards else "none")
            )
            transitions = ", ".join(
                f"{_JACKPOT_NAMES.get(index, index)} "
                f"{_number(before)}->{_number(after)}"
                for index, (before, after) in enumerate(
                    zip(
                        spin["jackpot_values_before_feature"],
                        spin["jackpot_values_after_feature"],
                    )
                )
            )
            lines.append(f"      Jackpot values after feature: {transitions}")
            if spin["feature_routes"]:
                lines.append(
                    "      Routed features: "
                    + ", ".join(spin["feature_routes"])
                )
    total_win = sum(result["total_win"] for result in payload["rounds"])
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
    if storage_type == "full_game":
        return _format_full_game_payload(payload, empty_marker)

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
        f"Jackpot-overlay spin rate: {stats.jackpot_overlay_spin_rate:>10.4%} "
        f"({stats.jackpot_overlay_spin_count:,}/{stats.spin_count:,})",
        f"Total jackpot overlays:    {stats.total_jackpot_overlays:,}",
        f"Average overlays per spin: "
        f"{stats.average_jackpot_overlays_per_spin:,.4f}",
        f"Maximum overlays on spin:  "
        f"{stats.maximum_jackpot_overlays_on_spin:,}",
        f"Total jackpot value added: {stats.total_jackpot_value_increment:,.4f}",
        f"Average value added/spin:  "
        f"{stats.average_jackpot_value_increment_per_spin:,.4f}",
        "Jackpot value additions are state changes, not base-game wins.",
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
    ]

    lines.extend(("", "JACKPOT OVERLAYS BY TYPE"))
    jackpot_rows = []
    for jackpot_type, count in enumerate(stats.jackpot_overlay_counts):
        share = (
            int(count) / stats.total_jackpot_overlays
            if stats.total_jackpot_overlays
            else 0.0
        )
        jackpot_rows.append(
            (
                _JACKPOT_NAMES.get(jackpot_type, f"Type {jackpot_type}"),
                f"{int(count):,}",
                f"{share:.4%}",
                f"{int(count) / stats.spin_count:.6f}",
                f"{stats.jackpot_value_increments[jackpot_type]:,.4f}",
            )
        )
    _table(
        lines,
        ("Jackpot", "Overlays", "% Overlays", "Per Spin", "Value Added"),
        jackpot_rows,
    )

    lines.extend(("", "JACKPOT OVERLAY COUNT HISTOGRAM"))
    overlay_count_rows = [
        (
            str(overlay_count),
            f"{int(count):,}",
            f"{int(count) / stats.spin_count:.4%}",
        )
        for overlay_count, count in enumerate(
            stats.jackpot_overlay_count_histogram
        )
    ]
    _table(lines, ("Overlays", "Spins", "% Spins"), overlay_count_rows)

    lines.extend(("", "WIN HISTOGRAM (multiples of bet)"))

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
        f"Jackpot-token respin rate:       {stats.jackpot_token_respin_rate:.4%} "
        f"({stats.jackpot_token_respin_count:,}/{stats.respin_count:,})",
        f"Total jackpot tokens:            {stats.total_jackpot_tokens:,}",
        f"Jackpot-award session rate:      {stats.jackpot_award_session_rate:.4%} "
        f"({stats.jackpot_award_session_count:,}/{stats.session_count:,})",
        "Jackpot awards are type events; monetary values are not included.",
    ]

    lines.extend(("", "JACKPOT COLLECTIONS BY TYPE"))
    jackpot_rows = [
        (
            _JACKPOT_NAMES.get(index, f"Type {index}"),
            f"{int(stats.jackpot_token_counts[index]):,}",
            f"{stats.average_final_jackpot_meters[index]:.4f}",
            f"{int(stats.jackpot_award_counts[index]):,}",
            f"{int(stats.jackpot_award_counts[index]) / stats.session_count:.4%}",
        )
        for index in range(len(stats.jackpot_token_counts))
    ]
    _table(
        lines,
        ("Jackpot", "Tokens", "Avg Final Meter", "Awards", "% Sessions"),
        jackpot_rows,
    )

    lines.extend(("", "SESSION WIN HISTOGRAM (multiples of session bet)"))

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


def format_full_game_statistics(statistics: FullGameStatistics):
    """Return a readable integrated full-game statistics report."""
    stats = statistics
    lines = [
        "FULL GAME STATISTICS",
        f"Source: {stats.source_path}",
        "",
        f"Rounds:                    {stats.round_count:,}",
        f"Paid spins:                {stats.spin_count:,}",
        f"Feature sessions:          {stats.feature_session_count:,}",
        f"Bet per paid spin:         {stats.bet_per_spin:,.4f}",
        f"Total bet:                 {stats.total_bet:,.4f}",
        f"Total win:                 {stats.total_win:,.4f}",
        f"Base win / RTP:            {stats.total_base_win:,.4f} / "
        f"{stats.base_rtp:.4%}",
        f"Feature win / RTP:         {stats.total_feature_win:,.4f} / "
        f"{stats.feature_rtp:.4%}",
        f"Jackpot win / RTP:         {stats.total_jackpot_win:,.4f} / "
        f"{stats.jackpot_rtp:.4%}",
        f"Overall RTP:               {stats.rtp:.4%}",
        f"Maximum paid-spin win:     {stats.maximum_spin_win:,.4f}",
        f"Maximum round win:         {stats.maximum_round_win:,.4f}",
        f"Hit rate:                  {stats.hit_rate:.4%} "
        f"({stats.hit_count:,}/{stats.spin_count:,})",
        f"Feature-trigger rate:      {stats.feature_trigger_rate:.4%} "
        f"({stats.feature_spin_count:,}/{stats.spin_count:,})",
        f"Jackpot awards:            {stats.total_jackpot_award_count:,}",
        f"Return variance:           {stats.return_variance:,.6f}",
        f"Return standard deviation: {stats.return_standard_deviation:,.6f}",
    ]
    if stats.feature_trigger_counts.any():
        lines.extend(("", "FEATURE RESULTS BY TYPE"))
        feature_rows = [
            (
                _ROUTED_FEATURE_NAMES[index],
                f"{int(stats.feature_trigger_counts[index]):,}",
                f"{int(stats.feature_spin_counts[index]):,}",
                f"{stats.feature_win_amounts[index]:,.4f}",
            )
            for index in range(len(_ROUTED_FEATURE_NAMES))
        ]
        _table(
            lines,
            ("Feature", "Sessions", "Feature Spins", "Win"),
            feature_rows,
        )
    lines.extend(("", "JACKPOT AWARDS BY TYPE"))
    rows = [
        (
            _JACKPOT_NAMES.get(index, f"Type {index}"),
            f"{int(count):,}",
            f"{stats.jackpot_award_amounts[index]:,.4f}",
        )
        for index, count in enumerate(stats.jackpot_award_counts)
    ]
    _table(lines, ("Jackpot", "Awards", "Amount"), rows)
    lines.extend(("", "PAID-SPIN WIN HISTOGRAM (multiples of bet)"))
    rows = [
        (
            "0 (no win)",
            f"{stats.zero_win_count:,}",
            f"{stats.zero_win_count / stats.spin_count:.4%}",
        )
    ]
    for index, count in enumerate(stats.win_histogram_counts):
        rows.append(
            (
                _win_range_label(
                    stats.win_histogram_edges[index],
                    stats.win_histogram_edges[index + 1],
                ),
                f"{int(count):,}",
                f"{int(count) / stats.spin_count:.4%}",
            )
        )
    _table(lines, ("Win", "Count", "% Spins"), rows)
    return "\n".join(lines)


def pretty_print_full_game(
    statistics: FullGameStatistics,
    file: TextIO | None = None,
):
    """Print a readable integrated full-game statistics report."""
    print(
        format_full_game_statistics(statistics),
        file=sys.stdout if file is None else file,
    )
