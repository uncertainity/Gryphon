"""Tune jackpot-token probabilities from production H&S opportunity traces.

The production kernel is first run with token capture forced on.  Its stored
respin overlays then reveal the number of jackpot-eligible newly landed coins
on every respin.  Candidate probabilities are evaluated analytically over
those observed opportunity profiles; feature mechanics are never recreated.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from ..core.config import FULL_GAME_CONFIG
from ..simulations.free_game import FEATURE_NAMES, _new_storage
from ..simulations.full_game import starting_bags_for_route
from .evaluate import (
    _evaluate_feature_kernel,
    _positive_count,
    load_feature_targets,
    load_game_targets,
)


JACKPOT_NAMES = ("Mini", "Minor", "Major", "Grand")

# Major and Grand are effectively capped before their rare awards.  Mini and
# Minor assumptions are initialized from integrated production simulations
# and are refined by the subsequent full-game validation pass.
DEFAULT_AVERAGE_AWARDS = np.array(
    [4.10, 22.70, 230.0, 23_000.0], dtype=np.float64
)


def _opportunity_rules(config=FULL_GAME_CONFIG):
    """Return rules that expose all token opportunities without awarding."""
    return config.hold_and_spin._replace(
        jackpot_token_probability=1.0,
        jackpot_type_probabilities=np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float64
        ),
        jackpot_collection_targets=np.full(4, 30_000, dtype=np.int16),
        max_jackpot_tokens_per_respin=(
            config.hold_and_spin.num_rows
            * config.hold_and_spin.num_reels
        ),
    )


def collect_opportunity_profiles(
    sessions_per_route: int,
    seed: int,
    config=FULL_GAME_CONFIG,
) -> tuple[Counter, dict]:
    """Collect a route-weighted distribution of per-respin opportunity counts."""
    target_probabilities, _ = load_feature_targets()
    conditional_route_weights = (
        target_probabilities / target_probabilities.sum()
    )
    rules = _opportunity_rules(config)
    weighted_profiles: Counter = Counter()
    diagnostics = {}

    for route_index, route_name in enumerate(FEATURE_NAMES):
        starting_bags = starting_bags_for_route(
            route_index,
            rules.bag_symbols,
        )
        storage = _new_storage(sessions_per_route, rules)
        _evaluate_feature_kernel(
            starting_bags,
            rules,
            sessions_per_route,
            storage,
            seed + route_index * 104_729,
        )

        route_profiles: Counter = Counter()
        total_opportunities = 0
        for session_index in range(storage.session_count):
            respin_start = int(
                storage.session_respin_offsets[session_index]
            )
            respin_end = int(
                storage.session_respin_offsets[session_index + 1]
            )
            profile = []
            for respin_index in range(respin_start, respin_end):
                opportunity_count = int(
                    np.count_nonzero(
                        storage.respin_jackpot_overlay_boards[
                            respin_index
                        ]
                        >= 0
                    )
                )
                profile.append(opportunity_count)
                total_opportunities += opportunity_count
            route_profiles[tuple(profile)] += 1

        route_weight = float(conditional_route_weights[route_index])
        for profile, count in route_profiles.items():
            weighted_profiles[profile] += (
                route_weight * count / sessions_per_route
            )
        diagnostics[route_name] = {
            "sessions": int(sessions_per_route),
            "unique_profiles": len(route_profiles),
            "average_opportunities": (
                total_opportunities / sessions_per_route
            ),
        }

    return weighted_profiles, diagnostics


def probability_of_three_tokens(
    profile: tuple[int, ...],
    token_probability: float,
    type_probability: float,
) -> float:
    """Return P(type token count >= 3) for one respin profile."""
    probability_0 = 1.0
    probability_1 = 0.0
    probability_2 = 0.0
    for opportunity_count in profile:
        if opportunity_count <= 0:
            continue
        any_token_probability = 1.0 - (
            1.0 - token_probability
        ) ** opportunity_count
        typed_probability = any_token_probability * type_probability
        next_2 = (
            probability_2 * (1.0 - typed_probability)
            + probability_1 * typed_probability
        )
        next_1 = (
            probability_1 * (1.0 - typed_probability)
            + probability_0 * typed_probability
        )
        probability_0 *= 1.0 - typed_probability
        probability_1 = next_1
        probability_2 = next_2
    return 1.0 - probability_0 - probability_1 - probability_2


def expected_award_probabilities(
    weighted_profiles: Counter,
    token_probability: float,
    type_probabilities: np.ndarray,
) -> np.ndarray:
    """Average each jackpot's award chance per H&S session."""
    result = np.zeros(len(type_probabilities), dtype=np.float64)
    for profile, profile_weight in weighted_profiles.items():
        for jackpot_index, type_probability in enumerate(
            type_probabilities
        ):
            result[jackpot_index] += profile_weight * (
                probability_of_three_tokens(
                    profile,
                    token_probability,
                    float(type_probability),
                )
            )
    return result


def _decode_candidate(parameters: np.ndarray) -> tuple[float, np.ndarray]:
    token_probability = 1.0 / (1.0 + np.exp(-parameters[0]))
    type_logits = np.append(parameters[1:], 0.0)
    type_logits -= type_logits.max()
    type_probabilities = np.exp(type_logits)
    type_probabilities /= type_probabilities.sum()
    return token_probability, type_probabilities


def tune(
    sessions_per_route: int,
    seed: int,
    average_awards=DEFAULT_AVERAGE_AWARDS,
    config=FULL_GAME_CONFIG,
) -> dict:
    """Fit token and type probabilities to jackpot RTP-derived hit rates."""
    profiles, diagnostics = collect_opportunity_profiles(
        sessions_per_route,
        seed,
        config,
    )
    game_targets = load_game_targets()
    target_paid_rates = np.array(
        [
            game_targets[f"{jackpot_name} RTP"] / average_awards[index]
            for index, jackpot_name in enumerate(JACKPOT_NAMES)
        ],
        dtype=np.float64,
    )
    overall_trigger_rate = game_targets["H&S trigger rate"]
    target_session_rates = target_paid_rates / overall_trigger_rate

    initial_type_probabilities = (
        config.hold_and_spin.jackpot_type_probabilities
    )
    initial_parameters = np.concatenate(
        (
            np.array(
                [
                    np.log(
                        config.hold_and_spin.jackpot_token_probability
                        / (
                            1.0
                            - config.hold_and_spin.jackpot_token_probability
                        )
                    )
                ]
            ),
            np.log(
                initial_type_probabilities[:3]
                / initial_type_probabilities[3]
            ),
        )
    )

    def residuals(parameters):
        token_probability, type_probabilities = _decode_candidate(parameters)
        observed = expected_award_probabilities(
            profiles,
            token_probability,
            type_probabilities,
        )
        return np.log(observed / target_session_rates)

    optimization = least_squares(
        residuals,
        initial_parameters,
        max_nfev=300,
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
    )
    token_probability, type_probabilities = _decode_candidate(
        optimization.x
    )
    predicted_session_rates = expected_award_probabilities(
        profiles,
        token_probability,
        type_probabilities,
    )
    predicted_paid_rates = (
        predicted_session_rates * overall_trigger_rate
    )
    predicted_rtps = predicted_paid_rates * average_awards

    return {
        "sessions_per_route": int(sessions_per_route),
        "seed": int(seed),
        "token_probability": float(token_probability),
        "type_probabilities": type_probabilities.tolist(),
        "average_award_assumptions": average_awards.tolist(),
        "target_paid_award_rates": target_paid_rates.tolist(),
        "predicted_paid_award_rates": predicted_paid_rates.tolist(),
        "predicted_rtps": predicted_rtps.tolist(),
        "optimization_success": bool(optimization.success),
        "optimization_cost": float(optimization.cost),
        "opportunity_profiles": {
            "unique_weighted_profiles": len(profiles),
            "weight_sum": float(sum(profiles.values())),
            "routes": diagnostics,
        },
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tune jackpot-token probabilities from production H&S traces."
        )
    )
    parser.add_argument(
        "--sessions-per-route",
        type=_positive_count,
        default=20_000,
    )
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(__file__).with_name("results")
        / "jackpot_opportunity_tuning.json",
    )
    return parser


def main(argv=None) -> dict:
    args = build_argument_parser().parse_args(argv)
    result = tune(args.sessions_per_route, args.seed)
    print(f"Token probability: {result['token_probability']:.12f}")
    print("Type probabilities:")
    for name, probability in zip(
        JACKPOT_NAMES, result["type_probabilities"]
    ):
        print(f"  {name:<6} {probability:.12f}")
    print("Predicted RTP:")
    for name, observed, target in zip(
        JACKPOT_NAMES,
        result["predicted_rtps"],
        (
            load_game_targets()[f"{name} RTP"]
            for name in JACKPOT_NAMES
        ),
    ):
        print(f"  {name:<6} {observed:.6%} (target {target:.6%})")

    output_path = args.json_output.resolve()
    tuning_root = Path(__file__).resolve().parent
    if tuning_root not in output_path.parents:
        raise ValueError("--json-output must be inside Numba_Engine/Tuning")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    main()
