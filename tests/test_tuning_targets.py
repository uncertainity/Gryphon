import numpy as np

from Numba_Engine.Tuning.targets import (
    BASE_RTP_FALLBACK,
    TOTAL_RTP_TARGET,
    jackpot_award_rate_targets,
    reconciled_feature_probabilities,
    reconciled_rtp_budget,
)


def test_new_trigger_targets_are_reconciled_without_a_plain_route():
    probabilities, diagnostics = reconciled_feature_probabilities()

    assert probabilities.shape == (8,)
    assert probabilities[7] == 0.0
    assert np.isclose(probabilities[6], 1.0 / 1500.0)
    assert np.allclose(probabilities[:6], probabilities[0])
    assert np.isclose(probabilities.sum(), 1.0 / 75.0)
    assert np.isclose(probabilities[0], 1.0 / 473.6842105263158)
    assert diagnostics["requested_exclusive_sum"] > 1.0 / 75.0


def test_new_jackpot_targets_use_the_correct_denominators():
    paid_rates, session_rates = jackpot_award_rate_targets()

    assert np.isclose(session_rates[0], 1.0 / 1.25)
    assert np.isclose(paid_rates[1], 1.0 / 1500.0)
    assert np.isclose(paid_rates[2], 1.0 / 50_000.0)
    assert np.isclose(paid_rates[3], 1.0 / 15_000_000.0)


def test_reconciled_rtp_budget_preserves_total_and_base_fallback():
    budget = reconciled_rtp_budget()

    assert np.isclose(budget["Total RTP"], TOTAL_RTP_TARGET)
    assert np.isclose(budget["Base subtotal RTP"], BASE_RTP_FALLBACK)
    assert np.isclose(
        budget["Base subtotal RTP"]
        + budget["Non-jackpot feature subtotal RTP"]
        + budget["Jackpot subtotal RTP"],
        TOTAL_RTP_TARGET,
    )
