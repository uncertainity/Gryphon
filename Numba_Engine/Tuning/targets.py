"""Parse and reconcile the supplied 2026-10-02 tuning constraints."""

from __future__ import annotations

import csv
from pathlib import Path
import re

import numpy as np


NEW_TARGET_PATH = Path(__file__).with_name("rtp_targets_new.csv")
TOTAL_RTP_TARGET = 0.9402008490918528
BASE_RTP_FALLBACK = 0.55

# Expected RTP at the explicit hit rates, using progressive-meter award
# means measured during the calibration pass. The final certification run
# refreshes these references without changing the requested hit rates.
CALIBRATED_JACKPOT_RTPS = np.array(
    [
        0.058346802712472,
        0.016126207777574,
        0.009915923771689,
        0.000999859611653,
    ],
    dtype=np.float64,
)


def _odds_probability(value: str) -> float:
    """Parse values such as ``1 in 450 spins`` or ``1 in 15M spins``."""
    match = re.fullmatch(
        r"\s*1\s+in\s+([0-9]+(?:\.[0-9]+)?)\s*([kKmM]?)"
        r"(?:\s+.*)?",
        value,
    )
    if match is None:
        raise ValueError(f"Unsupported odds target: {value!r}")
    denominator = float(match.group(1))
    suffix = match.group(2).lower()
    if suffix == "k":
        denominator *= 1_000.0
    elif suffix == "m":
        denominator *= 1_000_000.0
    return 1.0 / denominator


def load_new_target_rows(path: Path = NEW_TARGET_PATH) -> dict[tuple[str, str], str]:
    """Return non-empty category/component target strings from the new CSV."""
    rows: dict[tuple[str, str], str] = {}
    with path.open(newline="", encoding="utf-8") as target_file:
        for row in csv.DictReader(target_file):
            category = (row.get("category") or "").strip()
            component = (row.get("component") or "").strip()
            value = (
                row.get("target_odds (1 in X spins/features)") or ""
            ).strip()
            if category and component and value:
                rows[(category, component)] = value
    return rows


def reconciled_feature_probabilities(
    path: Path = NEW_TARGET_PATH,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return the closest symmetric route budget honoring total and Mega.

    The supplied mutually exclusive route rates sum to 1.40%, while the
    supplied overall rate is 1.3333%. Plain is explicitly zero. The agreed
    fallback treats overall and Mega as hard constraints and divides the
    remaining probability equally across the six single-Bag routes.
    """
    rows = load_new_target_rows(path)
    overall = _odds_probability(rows[("feature", "any Hold-and-Spin")])
    mega = _odds_probability(rows[("feature", "Mega Combo")])
    requested_single = _odds_probability(rows[("feature", "Splitter")])
    requested_plain = float(rows[("feature", "Plain Hold-and-Spin")])
    requested_sum = 6.0 * requested_single + mega + requested_plain
    feasible_single = (overall - mega - requested_plain) / 6.0
    if feasible_single <= 0.0:
        raise ValueError("Overall H&S target leaves no single-Bag budget")

    # Engine order: SC1-SC6, Mega, Plain.
    probabilities = np.array(
        [feasible_single] * 6 + [mega, requested_plain],
        dtype=np.float64,
    )
    diagnostics = {
        "overall_probability": overall,
        "requested_single_probability": requested_single,
        "feasible_single_probability": feasible_single,
        "mega_probability": mega,
        "plain_probability": requested_plain,
        "requested_exclusive_sum": requested_sum,
        "requested_exclusive_odds": 1.0 / requested_sum,
    }
    return probabilities, diagnostics


def jackpot_award_rate_targets(
    path: Path = NEW_TARGET_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    """Return paid-spin and conditional-session award probabilities."""
    rows = load_new_target_rows(path)
    _, diagnostics = reconciled_feature_probabilities(path)
    overall = diagnostics["overall_probability"]
    mini_session_rate = _odds_probability(rows[("jackpot", "Mini")])
    paid_rates = np.array(
        [
            mini_session_rate * overall,
            _odds_probability(rows[("jackpot", "Minor")]),
            _odds_probability(rows[("jackpot", "Major")]),
            _odds_probability(rows[("jackpot", "Grand")]),
        ],
        dtype=np.float64,
    )
    session_rates = paid_rates / overall
    return paid_rates, session_rates


def reconciled_rtp_budget() -> dict[str, float]:
    """Return the fallback RTP budget implied by the calibrated jackpots."""
    jackpot_subtotal = float(CALIBRATED_JACKPOT_RTPS.sum())
    return {
        "Total RTP": TOTAL_RTP_TARGET,
        "Base subtotal RTP": BASE_RTP_FALLBACK,
        "Non-jackpot feature subtotal RTP": (
            TOTAL_RTP_TARGET - BASE_RTP_FALLBACK - jackpot_subtotal
        ),
        "Jackpot subtotal RTP": jackpot_subtotal,
        "Mini RTP": float(CALIBRATED_JACKPOT_RTPS[0]),
        "Minor RTP": float(CALIBRATED_JACKPOT_RTPS[1]),
        "Major RTP": float(CALIBRATED_JACKPOT_RTPS[2]),
        "Grand RTP": float(CALIBRATED_JACKPOT_RTPS[3]),
    }
