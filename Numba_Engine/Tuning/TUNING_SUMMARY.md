# Gryphon current tuning summary

Updated: 2026-10-01

## Engine boundary

The integrated game remains fully inside the Numba engine framework. Runtime
behavior is implemented by the existing base, routing, and shared
Hold-and-Spin kernels. The files in this directory only load targets, construct
candidate configuration values, collect production-kernel measurements, and
report results. Neither tuning evaluator contains a replacement game flow.

## Selected controls

| Control | Selected value |
|---|---:|
| Base reelset weights | 0.0472980782 / 0.9517882953 / 0.0009136265 |
| Plain route relative weight | 0.1571597724 |
| Single-Bag relative weights | 1 / 1 / 1 / 1 / 1 / 1 |
| Base paytable scale | 2.0621263051 |
| Base Coin-drop probabilities | 0.0868970066 / 0.1303455099 / 0.1882768476 |
| Feature multipliers: Splitter / Grow / Boost / Multiplier | 1.923051683 / 1.331275802 / 1.558106375 / 1.506248630 |
| Feature multipliers: Collect / Expansion / Mega / Plain | 2.679141934 / 2.689860064 / 0.600905123 / 3.155013434 |
| Jackpot token probability | 0.6543931954 |
| Jackpot type weights: Mini / Minor / Major / Grand | 0.6851984585 / 0.2279710187 / 0.0760905664 / 0.0107399564 |

`ReelSet_3.csv` is the rare Mega-capable reelset. It is selected through the
normal base reelset weights, produces a normal 3x5 base window, and then uses
the same independent generic-SC conversion and all-six detection as every
other base spin. There is no direct or tuning-only Mega launch.

## Route-frequency solve

The third reelset exposes 12 generic SCs. With equal SC1-SC6 conversion,
the exact probability that all six distinct Bags appear is
43.7815680621%. Its selected reel weight therefore gives the target 0.04%
paid-spin Mega rate. The remaining reel weights solve the exact 1.00% total
Hold-and-Spin rate. The Plain relative weight then solves 0.1248% Plain and
0.1392% for each symmetric single-Bag route.

A 2,000,000-spin production-kernel route check measured 1.00335% total H&S
and 0.03905% Mega, consistent with those exact probabilities.

## Conditional feature calibration

The eight payout multipliers were fitted from 50,000 production H&S sessions
per route and checked with an independent 50,000-session seed. Independent
conditional means were:

| Route | Observed mean | Target mean |
|---|---:|---:|
| Splitter | 29.7565x | 29.6087x |
| Grow | 34.5636x | 34.7665x |
| Boost | 24.7570x | 24.8021x |
| Multiplier | 28.4181x | 28.4246x |
| Collect | 44.4094x | 44.3604x |
| Expansion | 51.6223x | 51.4483x |
| Mega Combo | 84.4756x | 84.8922x |
| Plain | 35.6293x | 35.5684x |

Progressive awards are not multiplied by these values.

## Integrated validation

The retained final run used 10,000,000 rounds, 11,124,765 paid spins, and seed
`20261014`.

| Metric | Observed | Target |
|---|---:|---:|
| Total RTP (no Grand observed) | 93.9892% | 94.0201% |
| Total RTP plus modeled Grand contribution | 94.0493% | 94.0201% |
| Base RTP | 55.2860% | 55.1746% |
| Base hit rate | 23.7548% | 23.7585% |
| Non-jackpot feature RTP | 37.4351% | 37.5414% |
| Hold-and-Spin trigger rate | 0.9984% | 1.0000% |
| Mini RTP | 0.7203% | 0.7038% |
| Minor RTP | 0.3617% | 0.3550% |
| Major RTP | 0.1861% | 0.1852% |
| Grand RTP | 0 observed | 0.0601% modeled |

Grand is modeled at roughly one award per 38.25 million paid spins, so a
10-million-round run normally contains zero or one and cannot certify its RTP
by ordinary sampling. `tune_jackpots.py` evaluates its three-token collection
probability over production-generated per-respin opportunity profiles instead.

## Reproduction

```bash
python -m Numba_Engine.Tuning.evaluate --routing-only --routing-spins 2m --seed 20261001
python -m Numba_Engine.Tuning.evaluate --features-only --feature-sessions 50k --seed 20261007
python -m Numba_Engine.Tuning.tune_jackpots --sessions-per-route 20k --seed 20261013
python -m Numba_Engine.Tuning.evaluate --full-only --full-rounds 10m --seed 20261014
```

Final machine-readable evidence is retained in:

- `results/routing_2m.json`
- `results/base_tuned_10m.json`
- `results/features_tuned_50k.json`
- `results/jackpot_opportunity_tuning.json`
- `results/full_tuned_10m.json`
