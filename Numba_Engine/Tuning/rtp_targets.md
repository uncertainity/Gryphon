# Gryphon RTP and trigger targets

Updated: 2026-10-02

`rtp_targets_new.csv` is the supplied source. `rtp_targets.csv` is the
reconciled, machine-readable calibration target consumed by the evaluator.
All RTP values use paid spins as the denominator.

## Reconciled game-level target

| Metric | Target |
|---|---:|
| Total expected RTP | 94.0201% |
| Base RTP | 55.0000% |
| Base hit rate | 25%-30% |
| Natural Base Collect | 1 in 20-25 paid spins |
| Coins collected per Collector | 3.5-5.0 |
| Any Hold-and-Spin | 1 in 75 paid spins |
| Plain Hold-and-Spin | 0 |
| Mega Combo | 1 in 1,500 paid spins |
| Hard win cap | 15,000x per wager round |

The resulting RTP budget is:

| Component | Expected RTP |
|---|---:|
| Base | 55.0000% |
| Non-jackpot Hold-and-Spin | 30.4812% |
| Progressive jackpots | 8.5389% |
| **Total** | **94.0201%** |

The jackpot subtotal is derived from the requested award odds and the
progressive award means measured by the final production-kernel run. Grand is
modeled because a 1-in-15-million event is not estimable from a 10M run alone.

## Route-frequency reconciliation

The requested mutually exclusive routes are internally inconsistent. Six
single-Bag routes at 1 in 450 plus Mega at 1 in 1,500 sum to 1.40%, or 1 in
71.43, while the supplied overall rate is 1.3333%, or 1 in 75.

The agreed fallback keeps the overall rate, Mega rate, and zero Plain rate as
hard constraints. The remaining trigger budget is split equally over the six
single-Bag routes:

| Route | Paid-spin probability | Odds | Calibrated mean win |
|---|---:|---:|---:|
| Splitter | 0.211111% | 1 in 473.68 | 17.8443x |
| Grow | 0.211111% | 1 in 473.68 | 20.8814x |
| Boost | 0.211111% | 1 in 473.68 | 15.0000x |
| Multiplier | 0.211111% | 1 in 473.68 | 17.0231x |
| Collect | 0.211111% | 1 in 473.68 | 26.4341x |
| Expansion | 0.211111% | 1 in 473.68 | 31.0319x |
| Mega Combo | 0.066667% | 1 in 1,500 | 51.2047x |
| Plain | 0 | disabled | not live |

The Plain execution path remains in the Numba engine for compatibility and
experiments, but its production route-selection weight is zero.

## Jackpot targets and feasibility

| Jackpot | Requested award odds | Calibrated expected RTP |
|---|---:|---:|
| Mini | 1 in 93.75 paid spins | 5.8347% |
| Minor | 1 in 1,500 paid spins | 1.6126% |
| Major | 1 in 50,000 paid spins | 0.9916% |
| Grand | 1 in 15,000,000 paid spins | 0.1000% |

With one token attempt per respin and three matching tokens required, all four
award odds cannot be met simultaneously. The constrained opportunity-profile
fit matches Minor, Major, and Grand closely and maximizes Mini at approximately
1 in 108.3 paid spins. The engine mechanic was preserved; token eligibility
was not broadened merely to force an infeasible target.

The progressive configuration is:

| Jackpot | Seed | Increment | Cap |
|---|---:|---:|---:|
| Mini | 2x | 0.5x | 10x |
| Minor | 10x | 1x | 25x |
| Major | 100x | 2x | 500x |
| Grand | 10,000x | 10x | 15,000x |

## Mechanic-target interpretation

The 6x5 Expansion/Mega backing board starts with three active rows, so at most
three distinct rows can be unlocked. The supplied averages of 4-4.5 and 4.5-5
row unlocks are therefore impossible when “row unlocks” means distinct rows.
The engine reports distinct rows unlocked and all-rows-unlocked frequency.

Likewise, feature credit values are integer xBet units in the shared board
kernel. Requested average Grow/Boost increments below 1x cannot be represented
exactly without changing that monetary model; the closest supported tables are
used and the final feature RTP is corrected only by payout multipliers.

## Retained validation

The final checks use production Numba kernels rather than a tuning-only game
implementation.

| Check | Result |
|---|---:|
| Base, 5M paid spins | 55.0589% RTP; 29.8955% hit rate |
| Coins per Collector, 3M diagnostic | 3.5128 |
| Natural Base Collect | 1 in 24.83 exact reel-window odds |
| Routing, 10M paid spins | 1.33045% H&S; 0 Plain; 0.06639% Mega |
| Full game, 10M rounds | 10,770,660 paid spins |
| Full-game sampled RTP | 93.8741% with no Grand observed |
| Full-game sampled + modeled Grand | 93.9741% |
| Expected RTP from fixed base, exact routes, and modeled jackpots | 94.0260% |

The expected result is 0.0059 percentage points above the 94.0201% target.
The observed full-game result is lower because that finite run had a slightly
low feature-trigger sample and no Grand jackpot.

Retained artifacts:

- `results/new_targets_base_final_5m.json`
- `results/new_targets_routing_final_10m.json`
- `results/new_targets_features_20k.json`
- `results/new_targets_jackpot_fit_20k.json`
- `results/new_targets_full_final_10m.json`
