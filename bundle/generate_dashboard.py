#!/usr/bin/env python3
"""Generate the self-contained Gryphon implementation dashboard."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "dashboard_gryphon.html"
NPZ_DIR = ROOT / "Numba_Engine" / "output" / "npz_library"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _fmt(value):
    if isinstance(value, float):
        return f"{value:,.4f}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def _pct(value):
    return f"{value:.4%}"


def _table(headers, rows):
    head = "".join(f"<th>{escape(str(header))}</th>" for header in headers)
    body = []
    for row in rows:
        cells = "".join(
            f"<td>{escape(str(value))}</td>" for value in row
        )
        body.append(f"<tr>{cells}</tr>")
    return (
        '<div class="scroll"><table><thead><tr>'
        f"{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def _cards(items):
    return '<div class="cards">' + "".join(
        '<article class="card">'
        f'<div class="label">{escape(str(label))}</div>'
        f'<div class="value">{escape(str(value))}</div>'
        f'<div class="note">{escape(str(note))}</div>'
        "</article>"
        for label, value, note in items
    ) + "</div>"


def _bars(items):
    maximum = max((float(value) for _, value in items), default=1.0) or 1.0
    rows = []
    for label, value in items:
        width = max(0.8, 100.0 * float(value) / maximum) if value else 0.0
        rows.append(
            '<div class="bar-row">'
            f'<span>{escape(str(label))}</span>'
            '<div class="bar-track">'
            f'<i style="width:{width:.3f}%"></i></div>'
            f'<b>{float(value):.2%}</b></div>'
        )
    return '<div class="bars">' + "".join(rows) + "</div>"


def _section(section_id, title, description, content):
    return (
        f'<section id="{escape(section_id)}"><h2>{escape(title)}</h2>'
        f'<p class="section-note">{escape(description)}</p>{content}</section>'
    )


def _latest_archives():
    archives = {}
    required = {
        "base": {
            "wins",
            "round_count",
            "jackpot_overlay_boards",
            "jackpot_values_before",
            "jackpot_values_after",
            "jackpot_increment_counts",
        },
        "free": {
            "session_wins",
            "respin_jackpot_overlay_boards",
            "session_jackpot_meters",
            "session_jackpot_awards",
        },
        "full": {
            "spin_total_wins",
            "spin_jackpot_awards",
            "round_total_wins",
        },
    }
    if not NPZ_DIR.is_dir():
        return archives
    for path in NPZ_DIR.glob("*.npz"):
        try:
            with np.load(path, allow_pickle=False) as data:
                kind = str(data["storage_kind"]) if "storage_kind" in data else ""
                if kind == "full_game":
                    storage_type = "full"
                elif kind == "hold_and_spin":
                    storage_type = "free"
                elif "wins" in data and "round_count" in data:
                    storage_type = "base"
                else:
                    continue
                if not required[storage_type].issubset(data.files):
                    continue
        except (OSError, ValueError, KeyError):
            continue
        current = archives.get(storage_type)
        if current is None or path.stat().st_mtime > current.stat().st_mtime:
            archives[storage_type] = path
    return archives


def _statistics_section():
    from Numba_Engine.output.statistics import (
        store_base_game,
        store_full_game,
        store_hold_and_spin,
    )

    archives = _latest_archives()
    blocks = []
    if "base" in archives:
        stats = store_base_game(archives["base"], print_result=False)
        blocks.append(
            (
                "Base game",
                archives["base"].name,
                [
                    ("Paid spins", _fmt(stats.spin_count)),
                    ("RTP", _pct(stats.rtp)),
                    ("Line RTP contribution", _pct(stats.total_line_win / stats.total_bet)),
                    ("Collect RTP contribution", _pct(stats.total_collect_win / stats.total_bet)),
                    ("Hit rate", _pct(stats.hit_rate)),
                    ("Feature trigger rate", _pct(stats.trigger_rate)),
                    ("Jackpot overlays", _fmt(stats.total_jackpot_overlays)),
                ],
            )
        )
    if "free" in archives:
        stats = store_hold_and_spin(archives["free"], print_result=False)
        blocks.append(
            (
                "Free game",
                archives["free"].name,
                [
                    ("Sessions", _fmt(stats.session_count)),
                    ("Session RTP", _pct(stats.rtp)),
                    ("Average respins", _fmt(stats.average_respins_per_session)),
                    ("Winning sessions", _pct(stats.hit_rate)),
                    ("Jackpot tokens", _fmt(stats.total_jackpot_tokens)),
                    ("Jackpot-award sessions", _pct(stats.jackpot_award_session_rate)),
                ],
            )
        )
    if "full" in archives:
        stats = store_full_game(archives["full"], print_result=False)
        blocks.append(
            (
                "Full game",
                archives["full"].name,
                [
                    ("Paid spins", _fmt(stats.spin_count)),
                    ("Overall RTP", _pct(stats.rtp)),
                    ("Base RTP", _pct(stats.base_rtp)),
                    ("Feature RTP", _pct(stats.feature_rtp)),
                    ("Jackpot RTP", _pct(stats.jackpot_rtp)),
                    ("Feature trigger rate", _pct(stats.feature_trigger_rate)),
                    ("Jackpot awards", _fmt(stats.total_jackpot_award_count)),
                ],
            )
        )

    if not blocks:
        return (
            '<p class="empty">No current-schema simulation NPZ files are '
            'available yet. Run <code>./tests/run.sh</code> to create them.</p>'
        )
    html = ['<div class="stat-grid">']
    for title, source, rows in blocks:
        html.append(
            '<article class="stat-block">'
            f"<h3>{escape(title)}</h3><code>{escape(source)}</code>"
            + _table(("Metric", "Value"), rows)
            + "</article>"
        )
    missing = [name for name in ("base", "free", "full") if name not in archives]
    if missing:
        html.append(
            '<article class="stat-block missing"><h3>Missing results</h3>'
            f"<p>{escape(', '.join(missing))} statistics will appear after "
            '<code>tests/run.sh</code> produces their NPZ files.</p></article>'
        )
    html.append("</div>")
    return "".join(html)


def build_dashboard():
    from Numba_Engine.core.config import (
        BASE_GAME_CONFIG,
        BASE_JACKPOT_OVERLAY_CONFIG,
        BASE_PAY_TABLE,
        FULL_GAME_CONFIG,
        HOLD_AND_SPIN_CONFIG,
        JACKPOT_CONFIG,
        PAY_LINES,
    )
    from Numba_Engine.core.reels import REEL_DICT

    symbol_names = {value: key for key, value in REEL_DICT.items()}
    jackpot_names = ("Mini", "Minor", "Major", "Grand")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    source_rows = (
        ("Numba_Engine/core/config.py", "All base, feature, overlay and jackpot tuning variables"),
        ("Numba_Engine/core/reels.py", "Symbol IDs and CSV reel loading"),
        ("Numba_Engine/simulations/base_game.py", "Paid spins and walking Collectors"),
        ("Numba_Engine/core/hold_and_spin_kernels.py", "Bag resolution, respins and jackpot meters"),
        ("Numba_Engine/simulations/full_game.py", "Base/free integration, jackpot awards and resets"),
        ("Numba_Engine/output/statistics.py", "Base, feature and full-game aggregation"),
    )
    symbol_rows = []
    bag_ids = set(int(value) for value in HOLD_AND_SPIN_CONFIG.bag_symbols)
    paying_ids = set(range(BASE_GAME_CONFIG.num_paying_symbols))
    for name, symbol_id in sorted(REEL_DICT.items(), key=lambda item: item[1]):
        if symbol_id in paying_ids:
            role = "Paying symbol" if name != "WD" else "Wild substitute"
        elif symbol_id == BASE_GAME_CONFIG.coin_symbol:
            role = "Base credit coin"
        elif symbol_id == BASE_GAME_CONFIG.collect_symbol:
            role = "Walking Collector"
        elif symbol_id in bag_ids:
            role = "Base trigger and feature Bag"
        else:
            role = "Reserved"
        symbol_rows.append((symbol_id, name, role))

    paytable_rows = []
    for symbol_id, pays in enumerate(BASE_PAY_TABLE):
        paytable_rows.append(
            (symbol_names.get(symbol_id, symbol_id),) + tuple(f"{pay:.2f}" for pay in pays)
        )

    reel_paths = sorted(Path(BASE_GAME_CONFIG.reelset_path).glob("*.csv"))
    reel_rows = []
    for index, probability in enumerate(BASE_GAME_CONFIG.reelset_probabilities):
        name = reel_paths[index].name if index < len(reel_paths) else f"Reel set {index + 1}"
        reel_rows.append((name, _pct(float(probability))))

    overlay_rows = [
        (int(count), _pct(float(probability)))
        for count, probability in zip(
            BASE_JACKPOT_OVERLAY_CONFIG.count_values,
            BASE_JACKPOT_OVERLAY_CONFIG.count_probabilities,
        )
    ]
    jackpot_rows = []
    for index, name in enumerate(jackpot_names):
        seed = float(JACKPOT_CONFIG.seed_values[index])
        jackpot_rows.append(
            (
                name,
                f"{seed:g}x",
                f"{float(JACKPOT_CONFIG.increment_values[index]):g}x",
                f"{seed * JACKPOT_CONFIG.cap_multiplier:g}x",
                int(HOLD_AND_SPIN_CONFIG.jackpot_collection_targets[index]),
                _pct(float(HOLD_AND_SPIN_CONFIG.jackpot_type_probabilities[index])),
            )
        )

    bag_names = ("Splitter", "Grower", "Booster", "Multiplier", "Collector", "Expansion")
    action_names = ("Remain", "Disappear", "Convert to credit coin")
    effects = (
        "Copies selected active coin values into empty unlocked cells.",
        "Increases selected active coins by weighted increments.",
        "Adds one weighted boost to every active unlocked coin.",
        "Multiplies every active unlocked coin by a weighted value.",
        "Adds the active coin total to the Collector meter.",
        "Unlocks rows upward from the starting active area.",
    )
    bag_rows = []
    for order, bag_index in enumerate(HOLD_AND_SPIN_CONFIG.bag_resolution_order, start=1):
        index = int(bag_index)
        bag_rows.append(
            (
                order,
                symbol_names.get(int(HOLD_AND_SPIN_CONFIG.bag_symbols[index]), index),
                bag_names[index],
                action_names[int(HOLD_AND_SPIN_CONFIG.bag_symbol_actions[index])],
                effects[index],
            )
        )

    base_flow = (
        "Choose a configured reelset and one random stop per reel.",
        "Overlay walking Collectors carried from the previous paid spin.",
        "Evaluate paylines, assign visible coin credits, and pay Collectors.",
        "Draw 0-4 jackpot overlays on eligible symbols and increment their progressives.",
        "If Bag triggers are visible, the full game launches the feature immediately.",
        "Move active Collectors right and continue paid spins until all have exited.",
    )
    feature_flow = (
        "Pass every visible trigger Bag from the base result, preserving duplicates and board order.",
        "Place the starting Bags and credit coins in the initially unlocked rows.",
        "Land natural coins or Bags, then resolve Bags in configured order.",
        "Attach jackpot-token overlays only to eligible naturally landed feature coins.",
        "Collect session-local jackpot meters; reaching a target awards that jackpot type.",
        "The full game pays the current progressive and resets only the awarded jackpot to seed.",
    )

    sections = []
    sections.append(_section("sources", "Authoritative sources", "The dashboard is generated from the current engine, not from the retired Gryphon_Lovy snapshot.", _table(("Path", "Role"), source_rows)))
    sections.append(_section("statistics", "Latest simulation results", "The newest compatible NPZ archive for each storage type is selected automatically.", _statistics_section()))
    sections.append(_section("base", "Base game", "Current paid-game dimensions, reel weights, paylines and credit values.", _cards((("Grid", f"{BASE_GAME_CONFIG.num_reels} × {BASE_GAME_CONFIG.num_rows}", "reels × rows"), ("Paylines", len(PAY_LINES), "left-to-right evaluation"), ("Trigger requirement", BASE_GAME_CONFIG.free_game_trigger_count, "visible Bag symbol"), ("Max Collectors", BASE_GAME_CONFIG.max_active_collectors, "walking instances"))) + '<div class="two-col">' + _table(("Reelset", "Probability"), reel_rows) + _table(("Coin credit", "Probability"), [(int(v), _pct(float(p))) for v, p in zip(BASE_GAME_CONFIG.coin_credit_values, BASE_GAME_CONFIG.coin_credit_value_probabilities)]) + "</div>"))
    sections.append(_section("base-flow", "Base-game flow", "A round may contain multiple paid spins while walking Collectors remain active.", '<div class="flow">' + "".join(f'<div><b>{i}</b><span>{escape(step)}</span></div>' for i, step in enumerate(base_flow, 1)) + "</div>"))
    sections.append(_section("symbols", "Symbol map", "Numeric IDs and their current runtime roles.", _table(("ID", "Symbol", "Role"), symbol_rows)))
    sections.append(_section("paytable", "Base paytable", "Values are multiples of bet; columns correspond to the stored match-count indices.", _table(("Symbol", "1", "2", "3", "4", "5"), paytable_rows)))
    sections.append(_section("jackpots", "Progressive jackpots", "Base overlays increment monetary values. Free-game token meters decide which values are awarded.", _cards((("Token chance", _pct(HOLD_AND_SPIN_CONFIG.jackpot_token_probability), "per eligible natural feature coin"), ("Max tokens/respin", HOLD_AND_SPIN_CONFIG.max_jackpot_tokens_per_respin, "configured hard limit"), ("Cap multiplier", f"{JACKPOT_CONFIG.cap_multiplier:g}× seed", "applied independently"), ("Session meters", "Reset", "at each free-game start"))) + _table(("Jackpot", "Seed", "Increment", "Cap", "Tokens to award", "Token type weight"), jackpot_rows) + '<div class="two-col"><div><h3>Base overlay count</h3>' + _bars([(str(c), float(p)) for c, p in zip(BASE_JACKPOT_OVERLAY_CONFIG.count_values, BASE_JACKPOT_OVERLAY_CONFIG.count_probabilities)]) + '</div><div><h3>Overlay count table</h3>' + _table(("Count", "Probability"), overlay_rows) + "</div></div>"))
    sections.append(_section("feature", "Hold-and-Spin feature", "The feature owns coin awards and session-local jackpot meters; the full game owns monetary jackpot payment.", _cards((("Grid", f"{HOLD_AND_SPIN_CONFIG.num_reels} × {HOLD_AND_SPIN_CONFIG.num_rows}", "reels × rows"), ("Starting rows", HOLD_AND_SPIN_CONFIG.starting_rows, "unlocked"), ("Starting respins", HOLD_AND_SPIN_CONFIG.respin_reset_count, "reset on qualifying natural coin"), ("Max coin", HOLD_AND_SPIN_CONFIG.max_coin_value, "credit cap"))) + _table(("Order", "Symbol", "Bag", "Post-action", "Effect"), bag_rows)))
    sections.append(_section("feature-flow", "Full-game feature flow", "Trigger Bags are transferred directly from the triggering base window without deduplication.", '<div class="flow">' + "".join(f'<div><b>{i}</b><span>{escape(step)}</span></div>' for i, step in enumerate(feature_flow, 1)) + "</div>"))
    sections.append(_section("runtime", "Runtime and reporting", "Supported entry points and generated artifacts.", _table(("Command", "Purpose"), (("python -m Numba_Engine.simulations.base_game", "Base simulation and statistics"), ("python -m Numba_Engine.simulations.free_game", "Standalone Hold-and-Spin simulation"), ("python -m Numba_Engine.simulations.full_game", "Integrated base/free/jackpot simulation"), ("python -m Numba_Engine.simulations.per_spin", "Interactive base, feature, and persistent full-game session"), ("./tests/run.sh", "Run 10,000 base, free, and full-game samples"), ("python bundle/generate_dashboard.py", "Regenerate this dashboard")))))

    nav = "".join(
        f'<a href="#{section_id}">{escape(label)}</a>'
        for section_id, label in (
            ("sources", "Sources"), ("statistics", "Statistics"),
            ("base", "Base game"), ("base-flow", "Base flow"),
            ("symbols", "Symbols"), ("paytable", "Paytable"),
            ("jackpots", "Jackpots"), ("feature", "Feature"),
            ("feature-flow", "Full-game flow"), ("runtime", "Runtime"),
        )
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gryphon Engine Dashboard</title>
<style>
:root{{--bg:#0d1117;--panel:#151b23;--panel2:#1c2430;--line:#2b3543;--text:#eef2f7;--muted:#9ca9b8;--blue:#4c9aff;--green:#35c48d;--gold:#e7b75b;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;--sans:Inter,ui-sans-serif,system-ui,sans-serif}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 var(--sans)}}.shell{{display:grid;grid-template-columns:210px minmax(0,1fr);gap:28px;max-width:1450px;margin:auto;padding:28px 24px 64px}}nav{{position:sticky;top:24px;align-self:start;display:grid;gap:3px}}nav strong{{font:700 11px var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:0 8px 8px}}nav a{{color:var(--muted);text-decoration:none;padding:6px 9px;border-left:2px solid transparent;border-radius:5px}}nav a:hover{{color:var(--text);background:var(--panel);border-left-color:var(--blue)}}main{{min-width:0}}header{{margin-bottom:18px}}.eyebrow{{font:700 11px var(--mono);letter-spacing:.16em;text-transform:uppercase;color:var(--blue)}}h1{{font-size:32px;line-height:1.15;margin:7px 0}}header p,.section-note,.note,.empty{{color:var(--muted)}}.meta{{display:flex;gap:12px;flex-wrap:wrap;font:12px var(--mono);color:var(--muted)}}section{{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:19px 20px;margin-top:16px;scroll-margin-top:20px}}h2{{margin:0;font-size:18px}}h3{{margin:0 0 10px;font-size:14px}}.section-note{{margin:5px 0 16px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:10px;margin-bottom:15px}}.card,.stat-block{{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:13px}}.label{{font:700 10px var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}}.value{{font:700 23px var(--mono);margin-top:6px}}.note{{font-size:11px;margin-top:3px}}.two-col,.stat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:13px}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:12.5px}}th,td{{padding:7px 9px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}thead th{{font:700 10px var(--mono);letter-spacing:.07em;text-transform:uppercase;color:var(--muted)}}tbody tr:hover{{background:#222c39}}code{{font:12px var(--mono);color:#acd1ff}}.flow{{display:grid;grid-template-columns:repeat(auto-fit,minmax(205px,1fr));gap:9px}}.flow>div{{background:var(--panel2);border:1px solid var(--line);border-radius:9px;padding:12px}}.flow b{{display:block;color:var(--blue);font:700 11px var(--mono);margin-bottom:5px}}.flow span{{color:#cbd4df}}.bars{{display:grid;gap:7px}}.bar-row{{display:grid;grid-template-columns:45px 1fr 70px;gap:8px;align-items:center;font:12px var(--mono)}}.bar-row>span,.bar-row>b{{color:var(--muted);text-align:right}}.bar-track{{height:10px;border-left:1px solid var(--line)}}.bar-track i{{display:block;height:9px;background:linear-gradient(90deg,var(--blue),var(--green));border-radius:0 5px 5px 0}}footer{{color:var(--muted);padding:22px 4px;font-size:12px}}@media(max-width:800px){{.shell{{grid-template-columns:1fr;padding:18px 13px 45px}}nav{{position:static;display:flex;flex-wrap:wrap}}nav strong{{width:100%}}}}
</style></head><body><div class="shell"><nav><strong>Gryphon</strong>{nav}</nav><main>
<header><div class="eyebrow">Current implementation console</div><h1>Gryphon engine dashboard</h1><p>Generated directly from the active Python configuration and latest compatible simulation archives. No Gryphon_Lovy runtime code is used.</p><div class="meta"><span>Generated: {escape(generated)}</span><span>Config: FULL_GAME_CONFIG</span><span>Engine: Python + Numba</span></div></header>
{''.join(sections)}
<footer>Regenerate with <code>python bundle/generate_dashboard.py</code>. Running <code>./tests/run.sh</code> also refreshes this dashboard after producing new simulation statistics.</footer>
</main></div></body></html>"""


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_dashboard(), encoding="utf-8")
    print(f"Dashboard written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
